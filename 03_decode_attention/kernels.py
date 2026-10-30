"""任务三：decode attention Triton kernels（连续 / paged KV + split 合并）。

结构
- `_decode_attn_contig_kernel`：grid = (B*HQ, NUM_SPLITS)，online softmax，写 partial。
- `_decode_attn_paged_kernel`：同样结构，但按 page 读取 KV。
- `_combine_kernel`：把 NUM_SPLITS 份 (m, l, acc) 用 log-sum-exp 合并。

NUM_SPLITS=1 时即为非 split 版本；所有 kernel 与 SDPA/eager 独立。
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False
    triton = None  # type: ignore
    tl = None  # type: ignore


if _HAS_TRITON:

    @triton.jit
    def _decode_attn_contig_kernel(
        Q,
        K,
        V,
        PartAcc,
        PartM,
        PartL,
        HQ,
        S,
        D,
        stride_qb,
        stride_qh,
        stride_kb,
        stride_ks,
        stride_kh,
        scale,
        GROUP: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_split = tl.program_id(1)
        b = pid_bh // HQ
        hq = pid_bh % HQ
        hkv = hq // GROUP

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q = tl.load(
            Q + b * stride_qb + hq * stride_qh + offs_d, mask=mask_d, other=0.0
        ).to(tl.float32)

        chunk = tl.cdiv(S, NUM_SPLITS)
        start = pid_split * chunk
        end = tl.minimum(start + chunk, S)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        offs_s = tl.arange(0, BLOCK_S)

        for s0 in range(start, end, BLOCK_S):
            sidx = s0 + offs_s
            mask_s = sidx < end
            k = tl.load(
                K
                + b * stride_kb
                + sidx[:, None] * stride_ks
                + hkv * stride_kh
                + offs_d[None, :],
                mask=mask_s[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * scale
            s = tl.where(mask_s, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            v = tl.load(
                V
                + b * stride_kb
                + sidx[:, None] * stride_ks
                + hkv * stride_kh
                + offs_d[None, :],
                mask=mask_s[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        base = pid_bh * NUM_SPLITS + pid_split
        tl.store(PartAcc + base * BLOCK_D + offs_d, acc, mask=mask_d)
        tl.store(PartM + base, m_i)
        tl.store(PartL + base, l_i)

    @triton.jit
    def _decode_attn_paged_kernel(
        Q,
        K,
        V,
        PAGE_TABLE,
        PartAcc,
        PartM,
        PartL,
        HQ,
        S,
        D,
        stride_qb,
        stride_qh,
        stride_kp,
        stride_kr,
        stride_kh,
        stride_pt,
        scale,
        GROUP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_split = tl.program_id(1)
        b = pid_bh // HQ
        hq = pid_bh % HQ
        hkv = hq // GROUP

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q = tl.load(
            Q + b * stride_qb + hq * stride_qh + offs_d, mask=mask_d, other=0.0
        ).to(tl.float32)

        pages_per_split = tl.cdiv(tl.cdiv(S, PAGE_SIZE), NUM_SPLITS)
        start = pid_split * pages_per_split * PAGE_SIZE
        end = tl.minimum(start + pages_per_split * PAGE_SIZE, S)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        within = tl.arange(0, PAGE_SIZE)

        for s0 in range(start, end, PAGE_SIZE):
            logical_page = s0 // PAGE_SIZE
            phys = tl.load(PAGE_TABLE + b * stride_pt + logical_page).to(tl.int64)
            sidx = s0 + within
            mask_s = sidx < end
            k = tl.load(
                K
                + phys * stride_kp
                + within[:, None] * stride_kr
                + hkv * stride_kh
                + offs_d[None, :],
                mask=mask_s[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * scale
            s = tl.where(mask_s, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            v = tl.load(
                V
                + phys * stride_kp
                + within[:, None] * stride_kr
                + hkv * stride_kh
                + offs_d[None, :],
                mask=mask_s[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        base = pid_bh * NUM_SPLITS + pid_split
        tl.store(PartAcc + base * BLOCK_D + offs_d, acc, mask=mask_d)
        tl.store(PartM + base, m_i)
        tl.store(PartL + base, l_i)

    @triton.jit
    def _combine_kernel(
        PartAcc,
        PartM,
        PartL,
        Out,
        HQ,
        D,
        stride_ob,
        stride_oh,
        NUM_SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // HQ
        hq = pid % HQ
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        m_max = float("-inf")
        for s in range(NUM_SPLITS):
            m_max = tl.maximum(m_max, tl.load(PartM + pid * NUM_SPLITS + s))

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        l = 0.0
        for s in range(NUM_SPLITS):
            m_s = tl.load(PartM + pid * NUM_SPLITS + s)
            alpha = tl.exp(m_s - m_max)
            acc += (
                tl.load(
                    PartAcc + (pid * NUM_SPLITS + s) * BLOCK_D + offs_d,
                    mask=mask_d,
                    other=0.0,
                )
                * alpha
            )
            l += tl.load(PartL + pid * NUM_SPLITS + s) * alpha

        out = acc / l
        tl.store(
            Out + b * stride_ob + hq * stride_oh + offs_d,
            out.to(Out.dtype.element_ty),
            mask=mask_d,
        )


def _check_available() -> None:
    if not _HAS_TRITON:
        raise RuntimeError("未安装 triton，无法执行 GPU kernel")
    if not torch.cuda.is_available():
        raise RuntimeError("无可用 CUDA 设备")


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor | None = None,
    seq_len: int | None = None,
    page_size: int = 16,
    num_splits: int = 1,
    block_s: int = 64,
) -> torch.Tensor:
    """单 token decode attention。

    Args:
        q: [B, HQ, D]
        k_cache/v_cache: dense [B, S, HKV, D] 或 paged [num_pages, page_size, HKV, D]
        page_table: paged 布局时必填，[B, max_pages] int32
        seq_len: paged 布局的逻辑序列长度（dense 时自动推断）
        num_splits: 序列切分数（必须是 2 的幂；1 表示不切分）
        block_s: dense 路径的 token block 大小
    """
    _check_available()
    if q.dim() != 3:
        raise ValueError("q 必须是 [B, HQ, D]")
    if not _is_pow2(num_splits):
        raise ValueError("num_splits 必须是 2 的幂")
    if not q.is_contiguous():
        raise ValueError("q 必须是 contiguous")

    b, hq, d = q.shape
    paged = page_table is not None
    hkv = k_cache.shape[2]
    if hq % hkv != 0:
        raise ValueError(f"HQ={hq} 不能被 HKV={hkv} 整除")

    if paged:
        if seq_len is None:
            raise ValueError("paged KV 必须显式传入 seq_len")
        s = int(seq_len)
        if not _is_pow2(page_size):
            raise ValueError("page_size 必须是 2 的幂")
    else:
        s = int(k_cache.shape[1])

    if s == 0:
        return torch.zeros_like(q)

    out = torch.empty_like(q)
    bh = b * hq
    block_d = triton.next_power_of_2(d)
    part_acc = torch.empty((bh, num_splits, block_d), dtype=torch.float32, device=q.device)
    part_m = torch.empty((bh, num_splits), dtype=torch.float32, device=q.device)
    part_l = torch.empty((bh, num_splits), dtype=torch.float32, device=q.device)

    scale = 1.0 / math.sqrt(d)
    group = hq // hkv
    grid = (bh, num_splits)

    if paged:
        _decode_attn_paged_kernel[grid](
            q,
            k_cache,
            v_cache,
            page_table,
            part_acc,
            part_m,
            part_l,
            hq,
            s,
            d,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            page_table.stride(0),
            scale,
            GROUP=group,
            PAGE_SIZE=page_size,
            BLOCK_D=block_d,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )
    else:
        _decode_attn_contig_kernel[grid](
            q,
            k_cache,
            v_cache,
            part_acc,
            part_m,
            part_l,
            hq,
            s,
            d,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            scale,
            GROUP=group,
            BLOCK_S=block_s,
            BLOCK_D=block_d,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )

    _combine_kernel[(bh,)](
        part_acc,
        part_m,
        part_l,
        out,
        hq,
        d,
        out.stride(0),
        out.stride(1),
        NUM_SPLITS=num_splits,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out
