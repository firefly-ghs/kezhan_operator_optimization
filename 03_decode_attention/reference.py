"""任务三：decode attention 的 baseline、float64 参考、KV cache 构造。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from common.utils import get_device


def group_size(num_heads: int, num_kv_heads: int) -> int:
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads 必须能被 num_kv_heads 整除")
    return num_heads // num_kv_heads


def torch_eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """eager 当前实现：matmul + softmax + matmul（多个 kernel，物化 [B,H,S] scores）。

    q: [B, HQ, D]; k/v: [B, S, HKV, D]
    """
    b, hq, d = q.shape
    _, s, hkv, _ = k.shape
    scale = scale or (1.0 / math.sqrt(d))
    if s == 0:
        return torch.zeros_like(q)
    g = group_size(hq, hkv)
    qg = q.float().reshape(b, hkv, g, d)
    kt = k.float().transpose(1, 2)  # [B, HKV, S, D]
    vt = v.float().transpose(1, 2)
    scores = torch.einsum("bhgd,bhsd->bhgs", qg, kt) * scale
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhgs,bhsd->bhgd", p, vt)
    return out.reshape(b, hq, d).to(q.dtype)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """SDPA 当前实现（GPU 上走 flash / mem-efficient backend）。

    q: [B, HQ, D]; k/v: [B, S, HKV, D]
    """
    if q.shape[1] == 0 or k.shape[1] == 0:
        return torch.zeros_like(q)
    q4 = q[:, :, None, :]                     # [B, HQ, 1, D]
    k4 = k.transpose(1, 2)                    # [B, HKV, S, D]
    v4 = v.transpose(1, 2)
    try:
        out = F.scaled_dot_product_attention(
            q4, k4, v4, scale=scale, enable_gqa=(q.shape[1] != k.shape[2])
        )
    except TypeError:
        g = group_size(q.shape[1], k.shape[2])
        k4 = k4.repeat_interleave(g, dim=1)
        v4 = v4.repeat_interleave(g, dim=1)
        out = F.scaled_dot_product_attention(q4, k4, v4, scale=scale)
    return out[:, :, 0, :]


def float64_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """float64 ground truth；k/v 为 dense [B, S, HKV, D]。"""
    b, hq, d = q.shape
    s, hkv = k.shape[1], k.shape[2]
    if s == 0:
        return torch.zeros_like(q, dtype=torch.float64)
    scale = scale or (1.0 / math.sqrt(d))
    g = group_size(hq, hkv)
    qd = q.detach().double().cpu().reshape(b, hkv, g, d)
    kd = k.detach().double().cpu().transpose(1, 2)  # [B, HKV, S, D]
    vd = v.detach().double().cpu().transpose(1, 2)
    scores = torch.einsum("bhgd,bhsd->bhgs", qd, kd) * scale
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhgs,bhsd->bhgd", p, vd)
    return out.reshape(b, hq, d)


def make_inputs(
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q: [B, HQ, D]（单 token），k/v: [B, S, HKV, D]。"""
    device = device or get_device()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(batch, num_heads, head_dim, generator=gen).to(dtype).to(device)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, generator=gen).to(dtype).to(device)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, generator=gen).to(dtype).to(device)
    return q, k, v


def pack_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 dense KV 打包成 paged 布局，页号按“页块优先、batch 交错”分配。

    Returns:
        k_cache/v_cache: [num_pages, page_size, HKV, D]
        page_table:      [B, max_pages] int32
    """
    b, s, hkv, d = k.shape
    max_pages = (s + page_size - 1) // page_size
    total_pages = b * max_pages
    kc = torch.zeros(total_pages, page_size, hkv, d, dtype=k.dtype, device=k.device)
    vc = torch.zeros_like(kc)
    table = torch.empty(b, max_pages, dtype=torch.int32, device=k.device)
    for bi in range(b):
        for t in range(max_pages):
            phys = t * b + bi
            table[bi, t] = phys
            lo, hi = t * page_size, min((t + 1) * page_size, s)
            if lo < hi:
                kc[phys, : hi - lo] = k[bi, lo:hi]
                vc[phys, : hi - lo] = v[bi, lo:hi]
    return kc, vc, table


def paged_to_dense(
    kc: torch.Tensor, vc: torch.Tensor, table: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 page table 还原 dense KV（用于参考实现与 baseline）。"""
    num_pages, page_size, hkv, d = kc.shape
    b = table.shape[0]
    k = torch.zeros(b, seq_len, hkv, d, dtype=kc.dtype, device=kc.device)
    v = torch.zeros_like(k)
    for bi in range(b):
        for t in range(table.shape[1]):
            phys = int(table[bi, t])
            lo, hi = t * page_size, min((t + 1) * page_size, seq_len)
            if lo < hi:
                k[bi, lo:hi] = kc[phys, : hi - lo]
                v[bi, lo:hi] = vc[phys, : hi - lo]
    return k, v


def attention_bytes(
    batch: int, num_heads: int, num_kv_heads: int, seq_len: int, head_dim: int, itemsize: int = 2
) -> int:
    """必须发生的 HBM 流量：读 K/V（每个 kv head 一份）+ 读写 q/out。"""
    return (
        2 * batch * seq_len * num_kv_heads * head_dim * itemsize
        + 2 * batch * num_heads * head_dim * itemsize
    )
