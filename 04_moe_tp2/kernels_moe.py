"""任务四-A：MoE grouped GEMM Triton kernels。

- `_grouped_gemm_kernel`：通用 grouped GEMM，一次 launch 覆盖所有专家；
- `_grouped_gemm_silu_kernel`：gate/up 融合，x 只读一次，epilogue 直接 SwiGLU；
- tile map 由 host 预处理，避免 `E × max_m_tiles` 的空 tile。
"""

from __future__ import annotations

import numpy as np
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
    def _grouped_gemm_kernel(
        A,
        W,
        C,
        OFFS,
        TILES,
        K,
        N,
        stride_am,
        stride_cm,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        e = tl.load(TILES + pid * 3)
        mt = tl.load(TILES + pid * 3 + 1)
        nt = tl.load(TILES + pid * 3 + 2)

        row_start = tl.load(OFFS + e)
        row_end = tl.load(OFFS + e + 1)
        m_local = row_end - row_start

        offs_m = mt * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < m_local
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)

        e64 = e.to(tl.int64)
        row = row_start + offs_m
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kmask = k0 + offs_k < K
            a = tl.load(
                A + row[:, None] * stride_am + (k0 + offs_k)[None, :],
                mask=mask_m[:, None] & kmask[None, :],
                other=0.0,
            )
            b = tl.load(
                W
                + e64 * (N * K)
                + offs_n[None, :] * K
                + (k0 + offs_k)[:, None],
                mask=mask_n[None, :] & kmask[:, None],
                other=0.0,
            )
            acc = tl.dot(a, b, acc)

        tl.store(
            C + row[:, None] * stride_cm + offs_n[None, :],
            acc.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _grouped_gemm_silu_kernel(
        A,
        WG,
        WU,
        C,
        OFFS,
        TILES,
        K,
        N,
        stride_am,
        stride_cm,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """gate/up 融合：一份 x，两份权重，输出 silu(gate) * up。"""
        pid = tl.program_id(0)
        e = tl.load(TILES + pid * 3)
        mt = tl.load(TILES + pid * 3 + 1)
        nt = tl.load(TILES + pid * 3 + 2)

        row_start = tl.load(OFFS + e)
        row_end = tl.load(OFFS + e + 1)
        m_local = row_end - row_start

        offs_m = mt * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < m_local
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)

        e64 = e.to(tl.int64)
        row = row_start + offs_m
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kmask = k0 + offs_k < K
            a = tl.load(
                A + row[:, None] * stride_am + (k0 + offs_k)[None, :],
                mask=mask_m[:, None] & kmask[None, :],
                other=0.0,
            )
            b_g = tl.load(
                WG
                + e64 * (N * K)
                + offs_n[None, :] * K
                + (k0 + offs_k)[:, None],
                mask=mask_n[None, :] & kmask[:, None],
                other=0.0,
            )
            b_u = tl.load(
                WU
                + e64 * (N * K)
                + offs_n[None, :] * K
                + (k0 + offs_k)[:, None],
                mask=mask_n[None, :] & kmask[:, None],
                other=0.0,
            )
            acc_g = tl.dot(a, b_g, acc_g)
            acc_u = tl.dot(a, b_u, acc_u)

        out = acc_g * tl.sigmoid(acc_g) * acc_u
        tl.store(
            C + row[:, None] * stride_cm + offs_n[None, :],
            out.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _check_available() -> None:
    if not _HAS_TRITON:
        raise RuntimeError("未安装 triton，无法执行 GPU kernel")
    if not torch.cuda.is_available():
        raise RuntimeError("无可用 CUDA 设备")


def build_tile_map(
    offsets: torch.Tensor,
    n_cols: int,
    block_m: int,
    block_n: int,
) -> torch.Tensor:
    """构造 (expert, m_tile, n_tile) 列表；跳过 0 token 的专家。"""
    offs = offsets.detach().cpu().tolist()
    tiles = []
    for e in range(len(offs) - 1):
        m = int(offs[e + 1]) - int(offs[e])
        if m <= 0:
            continue
        for mt in range((m + block_m - 1) // block_m):
            for nt in range((n_cols + block_n - 1) // block_n):
                tiles.append((e, mt, nt))
    if not tiles:
        return torch.empty((0, 3), dtype=torch.int32, device=offsets.device)
    return torch.tensor(tiles, dtype=torch.int32, device=offsets.device)


def _validate(sorted_x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor) -> None:
    if sorted_x.dim() != 2 or w.dim() != 3:
        raise ValueError("expect sorted_x [M,K], W [E,N,K]")
    if offsets.dim() != 1 or offsets.numel() != w.shape[0] + 1:
        raise ValueError("offsets 应为 [E+1]")
    if not sorted_x.is_contiguous() or not w.is_contiguous() or not offsets.is_contiguous():
        raise ValueError("输入必须是 contiguous")


def _launch(
    a: torch.Tensor,
    w: torch.Tensor,
    c: torch.Tensor,
    offsets: torch.Tensor,
    n_cols: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> None:
    tiles = build_tile_map(offsets, n_cols, block_m, block_n)
    if tiles.shape[0] == 0:
        return
    grid = (tiles.shape[0],)
    _grouped_gemm_kernel[grid](
        a,
        w,
        c,
        offsets,
        tiles,
        a.shape[1],
        n_cols,
        a.stride(0),
        c.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def moe_gate_up_plain(
    sorted_x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    offsets: torch.Tensor,
    block_m: int = 32,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """非融合 grouped gate/up，输出 [M, 2I]（两个 launch）。"""
    _check_available()
    _validate(sorted_x, w_gate, offsets)
    _validate(sorted_x, w_up, offsets)
    m, k = sorted_x.shape
    e, i, _ = w_gate.shape
    out = torch.empty(m, 2 * i, dtype=sorted_x.dtype, device=sorted_x.device)
    _launch(
        sorted_x, w_gate, out[:, :i], offsets, i, block_m, block_n, block_k, num_warps
    )
    _launch(
        sorted_x, w_up, out[:, i:], offsets, i, block_m, block_n, block_k, num_warps
    )
    return out


def moe_gate_up_silu(
    sorted_x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    offsets: torch.Tensor,
    block_m: int = 32,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """gate/up 融合 + SwiGLU，输出 [M, I]。"""
    _check_available()
    _validate(sorted_x, w_gate, offsets)
    _validate(sorted_x, w_up, offsets)
    m, _ = sorted_x.shape
    e, i, _ = w_gate.shape
    out = torch.empty(m, i, dtype=sorted_x.dtype, device=sorted_x.device)
    tiles = build_tile_map(offsets, i, block_m, block_n)
    if tiles.shape[0] == 0:
        return out
    _grouped_gemm_silu_kernel[(tiles.shape[0],)](
        sorted_x,
        w_gate,
        w_up,
        out,
        offsets,
        tiles,
        sorted_x.shape[1],
        i,
        sorted_x.stride(0),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def moe_down(
    sorted_y: torch.Tensor,
    w_down: torch.Tensor,
    offsets: torch.Tensor,
    block_m: int = 32,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """grouped down 投影，输出 [M, H]。"""
    _check_available()
    _validate(sorted_y, w_down, offsets)
    m, _ = sorted_y.shape
    e, h, _ = w_down.shape
    out = torch.empty(m, h, dtype=sorted_y.dtype, device=sorted_y.device)
    _launch(
        sorted_y, w_down, out, offsets, h, block_m, block_n, block_k, num_warps
    )
    return out
