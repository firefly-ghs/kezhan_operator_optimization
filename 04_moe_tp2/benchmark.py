"""任务四-A benchmark：per-expert baseline vs grouped GEMM / 融合 SwiGLU。

用法:
    python benchmark.py --tokens 2048 --experts 8 --hidden 4096 --intermediate 14336 --repeat 3
CPU 环境只跑 per-expert baseline 并打印 padding 统计。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.utils import (
    CUDA_AVAILABLE,
    TRITON_AVAILABLE,
    bench_callable,
    check_error,
    cv_percent,
    device_info,
    dtype_from_name,
    dtype_name,
    gbs,
    get_device,
    measure_peak_bandwidth,
    measure_peak_memory,
    peak_ratio,
    print_table,
    save_result,
    seed_all,
    tensor_bytes,
)
from reference import (
    make_moe_inputs,
    moe_bytes,
    moe_down_baseline,
    moe_gate_up_baseline,
)

import kernels_moe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE grouped GEMM benchmark")
    p.add_argument("--tokens", type=int, default=None)
    p.add_argument("--experts", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--intermediate", type=int, default=None)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--block-m", type=int, default=32)
    p.add_argument("--block-n", type=int, default=128)
    p.add_argument("--block-k", type=int, default=64)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_all(0)
    device = get_device()
    has_gpu_kernels = CUDA_AVAILABLE and TRITON_AVAILABLE

    tokens = args.tokens if args.tokens is not None else (2048 if has_gpu_kernels else 128)
    experts = args.experts if args.experts is not None else (8 if has_gpu_kernels else 4)
    hidden = args.hidden if args.hidden is not None else (4096 if has_gpu_kernels else 512)
    intermediate = args.intermediate if args.intermediate is not None else (14336 if has_gpu_kernels else 1024)

    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print(f"[info] CPU 环境将 dtype 从 {dtype_name(dtype)} 切换为 fp32")
        dtype = torch.float32

    data = make_moe_inputs(
        tokens, hidden, intermediate, experts, args.top_k, dtype, device
    )
    sx, offs = data["sorted_x"], data["offsets"]
    wg, wu, wd = data["w_gate"], data["w_up"], data["w_down"]
    m = sx.shape[0]
    counts = data["counts"]
    itemsize = torch.tensor([], dtype=dtype).element_size()

    print(f"device={device} dtype={dtype_name(dtype)}")
    print(f"T={tokens} E={experts} top_k={args.top_k} H={hidden} I={intermediate} M={m}")
    print(
        f"每专家 token 数: min={int(counts.min())} max={int(counts.max())} "
        f"avg={float(counts.float().mean()):.1f}"
    )
    pad_waste = sum(
        (int(math.ceil(c / args.block_m)) * args.block_m - c) if c > 0 else 0
        for c in counts.tolist()
    ) / max(1, m)
    print(f"BLOCK_M={args.block_m} 时 M 方向 padding 浪费: {pad_waste * 100:.1f}%")

    # tile 数与波数（评估 SM 利用率是否足够）
    sm = torch.cuda.get_device_properties(0).multi_processor_count if CUDA_AVAILABLE else None
    tiles_up = kernels_moe.build_tile_map(offs, intermediate, args.block_m, args.block_n).shape[0]
    tiles_up_plain = 2 * tiles_up  # 非融合版本两次 launch，每次 N=I
    tiles_down = kernels_moe.build_tile_map(offs, hidden, args.block_m, args.block_n).shape[0]
    print(f"tiles: gate_up_silu={tiles_up} gate_up_plain={tiles_up_plain} down={tiles_down}")
    if sm:
        print(
            f"waves(SM={sm}): gate_up_silu={math.ceil(tiles_up / sm)} "
            f"gate_up_plain={math.ceil(tiles_up_plain / sm)} down={math.ceil(tiles_down / sm)}"
        )

    y_gate_up = moe_gate_up_baseline(sx, wg, wu, offs, act=True)
    up_impls = {
        "baseline_silu": lambda: moe_gate_up_baseline(sx, wg, wu, offs, act=True),
        "baseline_plain": lambda: moe_gate_up_baseline(sx, wg, wu, offs, act=False),
    }
    down_impls = {"baseline_down": lambda: moe_down_baseline(y_gate_up, wd, offs)}
    if has_gpu_kernels:
        up_impls["grouped_silu"] = lambda: kernels_moe.moe_gate_up_silu(
            sx, wg, wu, offs, args.block_m, args.block_n, args.block_k
        )
        up_impls["grouped_plain"] = lambda: kernels_moe.moe_gate_up_plain(
            sx, wg, wu, offs, args.block_m, args.block_n, args.block_k
        )
        down_impls["grouped_down"] = lambda: kernels_moe.moe_down(
            y_gate_up, wd, offs, args.block_m, args.block_n, args.block_k
        )
        # 正确性：grouped 与 baseline 同数学比较
        for name, fn in (("grouped_silu", up_impls["grouped_silu"]), ("grouped_plain", up_impls["grouped_plain"])):
            ref = up_impls["baseline_silu" if name == "grouped_silu" else "baseline_plain"]()
            ok, _ = check_error(fn(), ref, 5e-3, 1e-2, name)
            if not ok:
                print(f"[error] {name} 未通过正确性检查，停止。")
                return 1
        ok, _ = check_error(down_impls["grouped_down"](), down_impls["baseline_down"](), 5e-3, 1e-2, "grouped_down")
        if not ok:
            print("[error] grouped_down 未通过正确性检查，停止。")
            return 1
    else:
        print("[skip] 未检测到 GPU + Triton，仅计时 per-expert baseline。")

    peak = measure_peak_bandwidth(device)
    bytes_up = moe_bytes(m, hidden, intermediate, experts, act=True, itemsize=itemsize)
    bytes_down = (m * intermediate + experts * hidden * intermediate + m * hidden) * itemsize

    static_mem = {
        "sorted_x_mb": tensor_bytes(sx) / 1e6,
        "w_gate_mb": tensor_bytes(wg) / 1e6,
        "w_up_mb": tensor_bytes(wu) / 1e6,
        "w_down_mb": tensor_bytes(wd) / 1e6,
        "y_gate_up_mb": tensor_bytes(y_gate_up) / 1e6,
    }
    print("静态显存分解(MB): " + "  ".join(f"{k}={v:.2f}" for k, v in static_mem.items()))
    # 峰值显存（仅 CUDA；在计时之外单独测量）
    peak_mem_mb: dict[str, float] = {}
    if device.type == "cuda":
        for group, impls, _, _ in (
            ("gate_up", up_impls, bytes_up, "baseline_silu"),
            ("down", down_impls, bytes_down, "baseline_down"),
        ):
            for name, fn in impls.items():
                key = f"{group}/{name}"
                peak_mem_mb[key] = measure_peak_memory(fn, device)["extra_mb"]
                print(f"peak_extra_mem {key}: {peak_mem_mb[key]:.2f} MB")

    results = {}
    for group, impls, nbytes, baseline in (
        ("gate_up", up_impls, bytes_up, "baseline_silu"),
        ("down", down_impls, bytes_down, "baseline_down"),
    ):
        rounds = []
        latency_detail: dict[str, dict[str, float]] = {}
        for rr in range(args.repeat):
            lat = {}
            for name, fn in impls.items():
                stat = bench_callable(fn, warmup=args.warmup, rep=args.rep, device=device)
                lat[name] = stat["ms"]
                if rr == 0:
                    latency_detail[name] = {
                        k: float(stat[k])
                        for k in ("mean_ms", "min_ms", "max_ms", "std_ms", "inner_iters", "samples")
                    }
            rounds.append(lat)
            print(f"[{group} round {rr + 1}/{args.repeat}] " + "  ".join(f"{k}={v:.4f}ms" for k, v in lat.items()))
        print(f"[{group}] 延迟分布(首轮采样, ms): " + "  ".join(
            f"{name}[mean={d['mean_ms']:.4f},min={d['min_ms']:.4f},max={d['max_ms']:.4f},std={d['std_ms']:.4f}]"
            for name, d in latency_detail.items()
        ))

        base_ms = [rd[baseline] for rd in rounds]
        rows, metrics = [], {}
        for name in impls:
            ms = [rd[name] for rd in rounds]
            median_ms = sorted(ms)[len(ms) // 2]
            speedups = [b / v for b, v in zip(base_ms, ms)]
            median_speedup = sorted(speedups)[len(speedups) // 2]
            rows.append(
                [
                    name,
                    f"{median_ms:.4f}",
                    f"{gbs(nbytes, median_ms):.1f}",
                    f"{peak_ratio(nbytes, median_ms, peak) * 100:.1f}%",
                    f"{tokens * 1e3 / median_ms:.0f}",
                    f"{peak_mem_mb.get(f'{group}/{name}', 0.0):.1f}",
                    f"{median_speedup:.2f}x",
                    f"{cv_percent(ms):.2f}%",
                ]
            )
            metrics[name] = {
                "median_ms": median_ms,
                "gbs": gbs(nbytes, median_ms),
                "peak_ratio": peak_ratio(nbytes, median_ms, peak),
                "tokens_per_s": tokens * 1e3 / median_ms,  # 每次迭代处理 T 个原始 token
                "peak_extra_mem_mb": peak_mem_mb.get(f"{group}/{name}", 0.0),
                "latency_detail_ms": latency_detail.get(name, {}),
                "speedup": median_speedup,
                "cv_percent": cv_percent(ms),
            }
        print(f"\n[{group}] 结果（中位数，{args.repeat} 轮），bytes/iter ≈ {nbytes / 1e6:.1f} MB")
        print_table(["impl", "ms", "GB/s", "peak%", "tok/s", "memMB", "speedup", "CV"], rows)
        results[group] = {
            "bytes_mb": nbytes / 1e6,
            "metrics": metrics,
            "rounds": rounds,
        }

    if args.save:
        save_result(
            "task4_moe",
            {
                "config": {
                    "tokens": tokens,
                    "experts": experts,
                    "top_k": args.top_k,
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "dtype": dtype_name(dtype),
                    "block_m": args.block_m,
                    "block_n": args.block_n,
                    "block_k": args.block_k,
                },
                "per_expert_tokens": {
                    "min": int(counts.min()),
                    "max": int(counts.max()),
                    "avg": float(counts.float().mean()),
                },
                "padding_waste_ratio": pad_waste,
                "peak_bw_gbs": peak,
                "memory_static_mb": static_mem,
                "device": device_info(),
                "results": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
