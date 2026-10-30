"""任务一：Decode 投影 GEMV 的 baseline、float64 参考与输入构造。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.utils import get_device


def torch_baseline(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """当前实现：cuBLAS / MKL GEMV（M=1）。"""
    return torch.nn.functional.linear(x, W)


def make_inputs(
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造接近 LLM 权重量级的输入：W ~ N(0, 0.02²)，x ~ N(0, 1)。"""
    device = device or get_device()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.randn(n, k, generator=gen) * 0.02).to(dtype).to(device)
    x = torch.randn(k, generator=gen).to(dtype).to(device)
    return W, x


def float64_reference(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """以 float64 计算 ground truth（返回 CPU 张量）。"""
    return W.detach().double().cpu() @ x.detach().double().cpu()


def gemv_bytes(n: int, k: int, itemsize: int = 2, out_itemsize: int = 2) -> int:
    """必须发生的 HBM 流量：读 W、读 x、写 y。"""
    return (n * k + k) * itemsize + n * out_itemsize
