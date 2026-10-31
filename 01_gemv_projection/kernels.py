"""任务一：Triton GEMV kernels（naive / tiled / split-K + reduce）。

设计要点
- 权重按 K 维连续加载，天然合并访存；x 的 tile 在 program 内复用。
- 结果统一 fp32 累加，最后转回输出 dtype。
- split-K 写 fp32 partial 再由 reduce kernel 求和，结果确定、可复现。
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only 环境
    _HAS_TRITON = False
    triton = None  # type: ignore
    tl = None  # type: ignore


if _HAS_TRITON:

    @triton.jit
    def _gemv_naive_kernel(
        W,
        X,
        Y,
        K,
        stride_wn,
        stride_wk,
        BLOCK_K: tl.constexpr,
    ):
        """v0：每个 program 处理 1 行，沿 K 循环，寄存器内累加。"""
        row = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            mask_k = k < K
            w = tl.load(
                W + row * stride_wn + k * stride_wk, mask=mask_k, other=0.0
            )
            x = tl.load(X + k, mask=mask_k, other=0.0)
            acc += tl.sum(w.to(tl.float32) * x.to(tl.float32), axis=0)
        tl.store(Y + row, acc.to(Y.dtype.element_ty))

    @triton.jit
    def _gemv_tiled_kernel(
        W,
        X,
        Y,
        N,
        K,
        stride_wn,
        stride_wk,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """v1：每个 program 处理 BLOCK_N 行，复用 x 的 tile。"""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            mask_k = k < K
            w = tl.load(
                W + offs_n[:, None] * stride_wn + k[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            x = tl.load(X + k, mask=mask_k, other=0.0)
            partial = tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
            acc += partial
        tl.store(Y + offs_n, acc.to(Y.dtype.element_ty), mask=mask_n)

    @triton.jit
    def _gemv_splitk_kernel(
        W,
        X,
        Partial,
        N,
        K,
        stride_wn,
        stride_wk,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """v2：K 维切 SPLIT_K 份，各份写 fp32 partial。"""
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        chunk = tl.cdiv(K, SPLIT_K)
        k_start = pid_k * chunk
        k_end = tl.minimum(k_start + chunk, K)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, chunk, BLOCK_K):
            k = k_start + k0 + offs_k
            mask_k = k < k_end
            w = tl.load(
                W + offs_n[:, None] * stride_wn + k[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            x = tl.load(X + k, mask=mask_k, other=0.0)
            partial = tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
            acc += partial
        tl.store(Partial + pid_k * N + offs_n, acc, mask=mask_n)

    @triton.jit
    def _splitk_reduce_kernel(
        Partial,
        Y,
        N,
        SPLIT_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """把 [SPLIT_K, N] 的 fp32 partial 求和并转成输出 dtype。"""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for s in range(SPLIT_K):
            acc += tl.load(Partial + s * N + offs_n, mask=mask_n, other=0.0)
        tl.store(Y + offs_n, acc.to(Y.dtype.element_ty), mask=mask_n)


def _check_available() -> None:
    if not _HAS_TRITON:
        raise RuntimeError("未安装 triton，无法执行 GPU kernel")
    if not torch.cuda.is_available():
        raise RuntimeError("无可用 CUDA 设备")


def _check_inputs(W: torch.Tensor, x: torch.Tensor) -> None:
    if W.dim() != 2 or x.dim() != 1:
        raise ValueError(f"expect W[N,K], x[K]; got {tuple(W.shape)}, {tuple(x.shape)}")
    if W.shape[1] != x.shape[0]:
        raise ValueError(f"shape mismatch: {tuple(W.shape)} vs {tuple(x.shape)}")
    if not W.is_contiguous() or not x.is_contiguous():
        raise ValueError("W 与 x 必须是 contiguous")


def _check_block_params(block_n: int | None = None, block_k: int | None = None) -> None:
    if block_n is not None and block_n <= 0:
        raise ValueError("block_n 必须 > 0")
    if block_k is not None:
        if block_k <= 0:
            raise ValueError("block_k 必须 > 0")
        # fp16/bf16 向量化常见对齐要求；也避免极小 tile 导致 launch 噪声占主导
        if block_k % 8 != 0:
            raise ValueError("block_k 建议为 8 的倍数")


def gemv_naive(W: torch.Tensor, x: torch.Tensor, block_k: int = 256) -> torch.Tensor:
    """v0：一行一 program。"""
    _check_available()
    _check_inputs(W, x)
    _check_block_params(block_k=block_k)
    n, k = W.shape
    y = torch.empty(n, dtype=W.dtype, device=W.device)
    _gemv_naive_kernel[(n,)](
        W,
        x,
        y,
        k,
        W.stride(0),
        W.stride(1),
        BLOCK_K=block_k,
        num_warps=4,
    )
    return y


def gemv_tiled(
    W: torch.Tensor,
    x: torch.Tensor,
    block_n: int = 8,
    block_k: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """v1：BLOCK_N 行一 program，复用 x tile。"""
    _check_available()
    _check_inputs(W, x)
    _check_block_params(block_n=block_n, block_k=block_k)
    if num_warps <= 0:
        raise ValueError("num_warps 必须 > 0")
    n, k = W.shape
    y = torch.empty(n, dtype=W.dtype, device=W.device)
    grid = (triton.cdiv(n, block_n),)
    _gemv_tiled_kernel[grid](
        W,
        x,
        y,
        n,
        k,
        W.stride(0),
        W.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return y


def gemv_splitk(
    W: torch.Tensor,
    x: torch.Tensor,
    block_n: int = 8,
    block_k: int = 256,
    split_k: int = 4,
    num_warps: int = 4,
) -> torch.Tensor:
    """v2：split-K + fp32 partial reduce。"""
    _check_available()
    _check_inputs(W, x)
    _check_block_params(block_n=block_n, block_k=block_k)
    if num_warps <= 0:
        raise ValueError("num_warps 必须 > 0")
    n, k = W.shape
    if split_k <= 0:
        raise ValueError("split_k 必须 > 0")
    # split 数量超过 K 没有意义，会带来额外 partial 流量与 launch 开销
    split_k = min(split_k, k)
    y = torch.empty(n, dtype=W.dtype, device=W.device)
    partial = torch.empty((split_k, n), dtype=torch.float32, device=W.device)
    grid = (triton.cdiv(n, block_n), split_k)
    _gemv_splitk_kernel[grid](
        W,
        x,
        partial,
        n,
        k,
        W.stride(0),
        W.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=num_warps,
    )
    _splitk_reduce_kernel[(triton.cdiv(n, 256),)](
        partial,
        y,
        n,
        SPLIT_K=split_k,
        BLOCK_N=256,
        num_warps=4,
    )
    return y
