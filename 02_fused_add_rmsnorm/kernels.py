"""任务二：单 kernel 融合 Residual Add + RMSNorm。"""

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
    def _fused_add_rmsnorm_kernel(
        X,
        R,
        W,
        OUT,
        RES_OUT,
        H,
        eps,
        stride_x,
        stride_r,
        stride_o,
        stride_ro,
        BLOCK_H: tl.constexpr,
        WRITE_RES: tl.constexpr,
    ):
        """一行一 program：读 x/residual → 加法 → fp32 平方和 → 归一化 → 写回。"""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
        h = x + r

        if WRITE_RES:
            tl.store(
                RES_OUT + row * stride_ro + offs,
                h.to(RES_OUT.dtype.element_ty),
                mask=mask,
            )

        # 用真实的 H 而不是 BLOCK_H 做 mean，避免 padding 影响
        var = tl.sum(h * h, axis=0) / H
        rrms = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        out = h * rrms * w
        tl.store(OUT + row * stride_o + offs, out.to(OUT.dtype.element_ty), mask=mask)


def _check_available() -> None:
    if not _HAS_TRITON:
        raise RuntimeError("未安装 triton，无法执行 GPU kernel")
    if not torch.cuda.is_available():
        raise RuntimeError("无可用 CUDA 设备")


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    write_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """融合 Residual Add + RMSNorm。

    Returns:
        out:  归一化结果
        h:    x + residual（write_residual=True 时返回，供残差流继续使用）
    """
    _check_available()
    if x.shape != residual.shape or x.dim() != 2:
        raise ValueError(f"expect x/residual [T,H]; got {tuple(x.shape)}, {tuple(residual.shape)}")
    if weight.dim() != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight 必须是 [H]")
    if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
        raise ValueError("x/residual/weight 必须是 contiguous")

    t, h = x.shape
    if h > 32768:
        raise ValueError(f"H={h} 超过单 program 寄存器预算，请改用分块两遍规约实现")

    out = torch.empty_like(x)
    res_out = torch.empty_like(x) if write_residual else out

    block_h = triton.next_power_of_2(h)
    num_warps = 4 if block_h <= 2048 else 8
    _fused_add_rmsnorm_kernel[(t,)](
        x,
        residual,
        weight,
        out,
        res_out,
        h,
        eps,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        res_out.stride(0),
        BLOCK_H=block_h,
        WRITE_RES=write_residual,
        num_warps=num_warps,
    )
    return out, (res_out if write_residual else None)
