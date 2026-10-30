"""任务四-B：双 GPU 拆分投影的本地小 GEMM kernel。

每张卡只持有 K 维的一半：x_shard [T, I/2]、w_shard [H, I/2]，
本地算 partial y [T, H]（fp32），再通过 all-reduce / P2P 合并。
"""

from __future__ import annotations

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
    def _small_gemm_kernel(
        A,
        W,
        C,
        M,
        N,
        K,
        stride_am,
        stride_wn,
        stride_cm,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kmask = k0 + offs_k < K
            a = tl.load(
                A + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :],
                mask=mask_m[:, None] & kmask[None, :],
                other=0.0,
            )
            b = tl.load(
                W + offs_n[None, :] * stride_wn + (k0 + offs_k)[:, None],
                mask=mask_n[None, :] & kmask[:, None],
                other=0.0,
            )
            acc = tl.dot(a, b, acc)

        tl.store(
            C + offs_m[:, None] * stride_cm + offs_n[None, :],
            acc.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _check_available() -> None:
    if not _HAS_TRITON:
        raise RuntimeError("未安装 triton，无法执行 GPU kernel")
    if not torch.cuda.is_available():
        raise RuntimeError("无可用 CUDA 设备")


def local_projection(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    block_m: int = 16,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
) -> None:
    """out = x @ wᵀ（本地 partial），写 fp32 out。

    支持非连续 view：x/w/out 的 data_ptr 偏移已经包含切片偏移，
    stride 由 host 显式传入。
    """
    _check_available()
    if x.dim() != 2 or w.dim() != 2 or out.dim() != 2:
        raise ValueError("x/w/out 必须是二维")
    if w.shape[1] != x.shape[1]:
        raise ValueError("K 维不一致")
    if out.shape != (x.shape[0], w.shape[0]):
        raise ValueError(f"out 形状应为 {(x.shape[0], w.shape[0])}")
    if block_m < 16:
        raise ValueError("tl.dot 要求 BLOCK_M >= 16")
    m, k = x.shape
    n = w.shape[0]
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _small_gemm_kernel[grid](
        x,
        w,
        out,
        m,
        n,
        k,
        x.stride(0),
        w.stride(0),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
