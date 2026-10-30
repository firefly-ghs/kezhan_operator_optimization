"""任务一 benchmark：baseline vs Triton naive / tiled / split-K。

用法:
    python benchmark.py --N 4096 --K 4096 --repeat 3
GPU 环境自动测量峰值带宽并计算带宽利用率；CPU 环境只跑 baseline 作为参考。
"""

from __future__ import annotations

import argparse
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
from reference import float64_reference, gemv_bytes, make_inputs, torch_baseline

import kernels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEMV decode projection benchmark")
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--block-n", type=int, default=8)
    p.add_argument("--block-k", type=int, default=256)
    p.add_argument("--split-k", type=int, default=4)
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--repeat", type=int, default=3, help="完整重复轮数，用于评估可复现性")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_all(0)
    device = get_device()

    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print(f"[info] CPU 环境将 dtype 从 {dtype_name(dtype)} 切换为 fp32 以获得有意义的 baseline 计时")
        dtype = torch.float32

    n, k = args.N, args.K
    W, x = make_inputs(n, k, dtype=dtype, device=device)
    ref = float64_reference(W, x)
    itemsize = torch.tensor([], dtype=dtype).element_size()
    nbytes = gemv_bytes(n, k, itemsize=itemsize, out_itemsize=itemsize)

    has_gpu_kernels = CUDA_AVAILABLE and TRITON_AVAILABLE
    impls = {"torch_baseline": lambda: torch_baseline(W, x)}
    if has_gpu_kernels:
        impls["triton_naive"] = lambda: kernels.gemv_naive(W, x, block_k=args.block_k)
        impls["triton_tiled"] = lambda: kernels.gemv_tiled(
            W, x, args.block_n, args.block_k, args.num_warps
        )
        impls["triton_splitk"] = lambda: kernels.gemv_splitk(
            W, x, args.block_n, args.block_k, args.split_k, args.num_warps
        )
    else:
        print("[skip] 未检测到 GPU + Triton，仅计时 torch baseline。")

    # 先做一次正确性检查，避免把错误实现拿来比较性能
    for name, fn in impls.items():
        ok, _ = check_error(fn(), ref, 1e-2, 1e-2, name=name)
        if not ok:
            print(f"[error] {name} 未通过正确性检查，停止。")
            return 1

    peak = measure_peak_bandwidth(device)
    flops = 2 * n * k
    static_mem = {"W_mb": tensor_bytes(W) / 1e6, "x_mb": tensor_bytes(x) / 1e6}
    if has_gpu_kernels:
        static_mem["splitk_partial_mb"] = (args.split_k * n * 4) / 1e6
    print(f"\nshape: W[{n},{k}] x[{k}]  dtype={dtype_name(dtype)}  device={device}")
    if device.type == "cuda":
        print(f"device_info: {device_info()}")
    print(f"peak_bw(measured)={peak:.1f} GB/s  bytes_moved={nbytes / 1e6:.1f} MB")
    print(f"FLOPs={flops / 1e9:.2f} GFLOP  AI={flops / nbytes:.2f} FLOP/Byte（远低于 ridge，带宽受限）")
    print("静态显存分解(MB): " + "  ".join(f"{k}={v:.2f}" for k, v in static_mem.items()))

    # 显存峰值（仅 CUDA；会重置进程级 peak 统计，放在计时之外）
    peak_mem_mb: dict[str, float] = {}
    if device.type == "cuda":
        for name, fn in impls.items():
            peak_mem_mb[name] = measure_peak_memory(fn, device)["extra_mb"]
            print(f"peak_extra_mem {name}: {peak_mem_mb[name]:.2f} MB")

    rounds: list[dict[str, float]] = []
    latency_detail: dict[str, dict[str, float]] = {}
    for r in range(args.repeat):
        lat: dict[str, float] = {}
        for name, fn in impls.items():
            stat = bench_callable(fn, warmup=args.warmup, rep=args.rep, device=device)
            lat[name] = stat["ms"]
            if r == 0:
                latency_detail[name] = {
                    k: float(stat[k])
                    for k in ("mean_ms", "min_ms", "max_ms", "std_ms", "inner_iters", "samples")
                }
        rounds.append(lat)
        print(f"[round {r + 1}/{args.repeat}] " + "  ".join(f"{k_}={v:.4f}ms" for k_, v in lat.items()))
    print("延迟分布(首轮采样, ms): " + "  ".join(
        f"{name}[mean={d['mean_ms']:.4f},min={d['min_ms']:.4f},max={d['max_ms']:.4f},std={d['std_ms']:.4f}]"
        for name, d in latency_detail.items()
    ))

    names = list(impls.keys())
    base_ms = [rd["torch_baseline"] for rd in rounds]
    rows = []
    summary = {
        "shape": {"N": n, "K": k, "dtype": dtype_name(dtype)},
        "peak_bw_gbs": peak,
        "memory_static_mb": static_mem,
    }
    for name in names:
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
                f"{1e3 / median_ms:.0f}",
                f"{peak_mem_mb.get(name, 0.0):.1f}",
                f"{median_speedup:.2f}x",
                f"{cv_percent(ms):.2f}%",
            ]
        )
        summary[name] = {
            "median_ms": median_ms,
            "gbs": gbs(nbytes, median_ms),
            "peak_ratio": peak_ratio(nbytes, median_ms, peak),
            "tokens_per_s": 1e3 / median_ms,  # 每次迭代 1 个 token
            "peak_extra_mem_mb": peak_mem_mb.get(name, 0.0),
            "latency_detail_ms": latency_detail.get(name, {}),
            "speedup": median_speedup,
            "cv_percent": cv_percent(ms),
        }

    print("\n结果（中位数，%d 轮）:" % args.repeat)
    from common.utils import print_table

    print_table(["impl", "ms", "GB/s", "peak%", "tok/s", "memMB", "speedup", "CV"], rows)

    worst_cv = max(summary[name]["cv_percent"] for name in names)
    if worst_cv > 5.0:
        print(f"[warn] 轮间变异系数最大 {worst_cv:.2f}% > 5%，请排查计时稳定性。")

    if args.save:
        summary["rounds"] = rounds
        summary["device"] = device_info()
        summary["cv_percent_max"] = worst_cv
        save_result("task1_gemv", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
