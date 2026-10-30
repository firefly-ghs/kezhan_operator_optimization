"""任务二 benchmark：2-kernel baseline vs 单 kernel 融合（两种输出模式）。"""

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
from reference import (
    baseline_bytes,
    float64_reference,
    fused_bytes,
    make_inputs,
    torch_baseline,
)

import kernels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fused residual-add + RMSNorm benchmark")
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--H", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_all(0)
    device = get_device()
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print(f"[info] CPU 环境将 dtype 从 {dtype_name(dtype)} 切换为 fp32")
        dtype = torch.float32

    t, h = args.T, args.H
    x, r, w = make_inputs(t, h, dtype=dtype, device=device)
    ref_out, _ = float64_reference(x, r, w)
    itemsize = torch.tensor([], dtype=dtype).element_size()

    has_gpu_kernels = CUDA_AVAILABLE and TRITON_AVAILABLE
    impls = {
        "baseline_2kernel": (
            lambda: torch_baseline(x, r, w),
            baseline_bytes(t, h, itemsize),
        ),
    }
    if has_gpu_kernels:
        impls["fused_write_res"] = (
            lambda: kernels.fused_add_rmsnorm(x, r, w, 1e-6, True),
            fused_bytes(t, h, itemsize, True),
        )
        impls["fused_norm_only"] = (
            lambda: kernels.fused_add_rmsnorm(x, r, w, 1e-6, False),
            fused_bytes(t, h, itemsize, False),
        )
    else:
        print("[skip] 未检测到 GPU + Triton，仅计时 baseline。")

    atol = 5e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-2 if dtype == torch.float16 else 1e-5
    for name, (fn, _) in impls.items():
        result = fn()
        out = result[0] if isinstance(result, tuple) else result
        ok, _ = check_error(out, ref_out, atol, rtol, name=name)
        if not ok:
            print(f"[error] {name} 未通过正确性检查，停止。")
            return 1

    peak = measure_peak_bandwidth(device)
    static_mem = {
        "x_mb": tensor_bytes(x) / 1e6,
        "residual_mb": tensor_bytes(r) / 1e6,
        "weight_mb": tensor_bytes(w) / 1e6,
        "out_mb": tensor_bytes(x) / 1e6,
        "res_out_mb": tensor_bytes(x) / 1e6,
    }
    print(f"\nshape: x[{t},{h}] dtype={dtype_name(dtype)} device={device}")
    if device.type == "cuda":
        print(f"device_info: {device_info()}")
    print(f"peak_bw(measured)={peak:.1f} GB/s")
    print("静态显存分解(MB): " + "  ".join(f"{k}={v:.2f}" for k, v in static_mem.items()))

    peak_mem_mb: dict[str, float] = {}
    if device.type == "cuda":
        for name, (fn, _) in impls.items():
            peak_mem_mb[name] = measure_peak_memory(fn, device)["extra_mb"]
            print(f"peak_extra_mem {name}: {peak_mem_mb[name]:.2f} MB")

    rounds: list[dict[str, float]] = []
    latency_detail: dict[str, dict[str, float]] = {}
    for rr in range(args.repeat):
        lat = {}
        for name, (fn, _) in impls.items():
            stat = bench_callable(fn, warmup=args.warmup, rep=args.rep, device=device)
            lat[name] = stat["ms"]
            if rr == 0:
                latency_detail[name] = {
                    k: float(stat[k])
                    for k in ("mean_ms", "min_ms", "max_ms", "std_ms", "inner_iters", "samples")
                }
        rounds.append(lat)
        print(f"[round {rr + 1}/{args.repeat}] " + "  ".join(f"{k}={v:.4f}ms" for k, v in lat.items()))
    print("延迟分布(首轮采样, ms): " + "  ".join(
        f"{name}[mean={d['mean_ms']:.4f},min={d['min_ms']:.4f},max={d['max_ms']:.4f},std={d['std_ms']:.4f}]"
        for name, d in latency_detail.items()
    ))

    names = list(impls.keys())
    base_ms = [rd["baseline_2kernel"] for rd in rounds]
    rows, metrics = [], {}
    for name in names:
        ms = [rd[name] for rd in rounds]
        median_ms = sorted(ms)[len(ms) // 2]
        nbytes = impls[name][1]
        speedups = [b / v for b, v in zip(base_ms, ms)]
        median_speedup = sorted(speedups)[len(speedups) // 2]
        rows.append(
            [
                name,
                f"{nbytes / 1e6:.1f}",
                f"{median_ms:.4f}",
                f"{gbs(nbytes, median_ms):.1f}",
                f"{peak_ratio(nbytes, median_ms, peak) * 100:.1f}%",
                f"{t * 1e3 / median_ms:.0f}",
                f"{peak_mem_mb.get(name, 0.0):.1f}",
                f"{median_speedup:.2f}x",
                f"{cv_percent(ms):.2f}%",
            ]
        )
        metrics[name] = {
            "bytes_mb": nbytes / 1e6,
            "median_ms": median_ms,
            "gbs": gbs(nbytes, median_ms),
            "peak_ratio": peak_ratio(nbytes, median_ms, peak),
            "tokens_per_s": t * 1e3 / median_ms,  # 每次迭代处理 T 行
            "peak_extra_mem_mb": peak_mem_mb.get(name, 0.0),
            "latency_detail_ms": latency_detail.get(name, {}),
            "speedup": median_speedup,
            "cv_percent": cv_percent(ms),
        }

    print("\n结果（中位数，%d 轮）:" % args.repeat)
    print_table(["impl", "MB", "ms", "GB/s", "peak%", "tok/s", "memMB", "speedup", "CV"], rows)

    worst_cv = max(m["cv_percent"] for m in metrics.values())
    if worst_cv > 5.0:
        print(f"[warn] 轮间变异系数最大 {worst_cv:.2f}% > 5%。")

    if args.save:
        save_result(
            "task2_add_rmsnorm",
            {
                "shape": {"T": t, "H": h, "dtype": dtype_name(dtype)},
                "peak_bw_gbs": peak,
                "memory_static_mb": static_mem,
                "device": device_info(),
                "metrics": metrics,
                "rounds": rounds,
                "cv_percent_max": worst_cv,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
