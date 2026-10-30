"""任务四：MoE 路由构造、per-expert baseline、float64 参考。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.utils import get_device


def make_moe_inputs(
    tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int = 2,
    dtype: torch.dtype = torch.float16,
    device: torch.device | None = None,
    seed: int = 0,
    drop_expert: int | None = None,
) -> dict[str, torch.Tensor]:
    """构造 token、专家权重和路由结果（按 expert 排序）。"""
    device = device or get_device()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(tokens, hidden, generator=gen) * 0.5).to(dtype).to(device)
    w_gate = (torch.randn(experts, intermediate, hidden, generator=gen) * 0.02).to(dtype).to(device)
    w_up = (torch.randn(experts, intermediate, hidden, generator=gen) * 0.02).to(dtype).to(device)
    w_down = (torch.randn(experts, hidden, intermediate, generator=gen) * 0.02).to(dtype).to(device)

    logits = torch.randn(tokens, experts, generator=gen)
    if drop_expert is not None:
        logits[:, drop_expert] = float("-inf")
    topk = logits.topk(top_k, dim=-1).indices  # [T, top_k]

    expert_ids = topk.reshape(-1)
    token_ids = torch.arange(tokens).repeat_interleave(top_k)
    order = torch.argsort(expert_ids, stable=True)
    sorted_token_ids = token_ids[order]
    sorted_expert_ids = expert_ids[order]

    counts = torch.bincount(sorted_expert_ids, minlength=experts)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(torch.int32)
    sorted_x = x[sorted_token_ids].contiguous()

    return {
        "x": x,
        "sorted_x": sorted_x,
        "token_ids": sorted_token_ids.to(device),
        "offsets": offsets.to(device),
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "topk": topk,
        "counts": counts,
    }


def moe_gate_up_baseline(
    sorted_x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    offsets: torch.Tensor,
    act: bool = True,
) -> torch.Tensor:
    """per-expert 循环：每个专家一次小 GEMM（当前实现）。"""
    experts = w_gate.shape[0]
    outs = []
    for e in range(experts):
        lo = int(offsets[e].item())
        hi = int(offsets[e + 1].item())
        if hi <= lo:
            continue
        x_e = sorted_x[lo:hi]
        g = torch.matmul(x_e, w_gate[e].t())
        u = torch.matmul(x_e, w_up[e].t())
        outs.append(torch.nn.functional.silu(g) * u if act else torch.cat([g, u], dim=-1))
    if not outs:
        n = w_gate.shape[1] if act else 2 * w_gate.shape[1]
        return torch.empty(0, n, dtype=sorted_x.dtype, device=sorted_x.device)
    return torch.cat(outs, dim=0)


def moe_down_baseline(
    sorted_y: torch.Tensor,
    w_down: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    experts = w_down.shape[0]
    outs = []
    for e in range(experts):
        lo = int(offsets[e].item())
        hi = int(offsets[e + 1].item())
        if hi <= lo:
            continue
        outs.append(torch.matmul(sorted_y[lo:hi], w_down[e].t()))
    if not outs:
        return torch.empty(0, w_down.shape[1], dtype=sorted_y.dtype, device=sorted_y.device)
    return torch.cat(outs, dim=0)


def moe_gate_up_reference(
    sorted_x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    offsets: torch.Tensor,
    act: bool = True,
) -> torch.Tensor:
    """float64 参考（调用方负责传入 double 张量）。"""
    return moe_gate_up_baseline(sorted_x, w_gate, w_up, offsets, act)


def moe_down_reference(
    sorted_y: torch.Tensor,
    w_down: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return moe_down_baseline(sorted_y, w_down, offsets)


def moe_bytes(
    m_tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    act: bool = True,
    itemsize: int = 2,
) -> int:
    """gate/up：读 token、读全部专家权重、写输出。"""
    n_out = intermediate if act else 2 * intermediate
    return (m_tokens * hidden + 2 * experts * intermediate * hidden + m_tokens * n_out) * itemsize
