"""任务二：Residual Add + RMSNorm 的 baseline、float64 参考与输入构造。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from common.utils import get_device


def torch_baseline(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """当前实现：2 个 kernel，中间张量 h 落 HBM。"""
    h = x + residual
    out = F.rms_norm(h, (h.shape[-1],), weight, eps)
    return out, h


def torch_eager_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐算子 eager 写法，仅用于对照（非融合 baseline 的另一种形态）。"""
    h = x + residual
    hf = h.float()
    out = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(
        x.dtype
    )
    return out, h


def make_inputs(
    t: int,
    h: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = device or get_device()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(t, h, generator=gen).to(dtype).to(device)
    residual = torch.randn(t, h, generator=gen).to(dtype).to(device)
    weight = (1.0 + 0.1 * torch.randn(h, generator=gen)).to(dtype).to(device)
    return x, residual, weight


def float64_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    xd, rd, wd = x.detach().double().cpu(), residual.detach().double().cpu(), weight.detach().double().cpu()
    h = xd + rd
    out = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * wd
    return out, h


def baseline_bytes(t: int, h: int, itemsize: int = 2) -> int:
    """2 kernel 版本：add(读2写1) + rmsnorm(读1写1) + weight 读 1 次。"""
    return 5 * t * h * itemsize + h * itemsize


def fused_bytes(t: int, h: int, itemsize: int = 2, write_residual: bool = True) -> int:
    """单 kernel：读 x/residual，写 out（以及可选 h），weight 读 1 次。"""
    return (2 + (2 if write_residual else 1)) * t * h * itemsize + h * itemsize
