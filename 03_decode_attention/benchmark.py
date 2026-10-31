"""任务三 benchmark：eager / SDPA / Triton(split) / paged，扫描多个上下文长度。"""

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
    attention_bytes,
    float64_reference,
    make_inputs,
    pack_paged,
    sdpa_attention,
    torch_eager_attention,
)

import kernels


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _prev_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="decode attention benchmark")
    p.add_argument("--contexts", type=int, nargs="+", default=None)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--splits", type=int, default=0, help="0 表示按 SM 数自适应")
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

    has_gpu_kernels = CUDA_AVAILABLE and TRITON_AVAILABLE
    contexts = args.contexts or ([512, 2048, 8192, 32768] if has_gpu_kernels else [256, 1024])

    atol = 2e-2 if dtype == torch.float16 else 1e-4
    rtol = 2e-2 if dtype == torch.float16 else 1e-4
    peak = measure_peak_bandwidth(device)
    print(f"device={device} dtype={dtype_name(dtype)} peak_bw={peak:.1f} GB/s")
    if device.type == "cuda":
        print(f"device_info: {device_info()}")

    results: dict[int, dict] = {}
    for s in contexts:
        print(f"\n===== context S={s} =====")
        q, k, v = make_inputs(
            args.batch, args.heads, args.kv_heads, s, args.dim, dtype=dtype, device=device
        )
        ref = float64_reference(q, k, v)
        nbytes = attention_bytes(
            args.batch, args.heads, args.kv_heads, s, args.dim,
            itemsize=torch.tensor([], dtype=dtype).element_size(),
        )

        impls = {
            "eager": lambda: torch_eager_attention(q, k, v),
            "sdpa": lambda: sdpa_attention(q, k, v),
        }
        if has_gpu_kernels:
            bh = args.batch * args.heads
            splits = args.splits
            if splits == 0:
                sm = torch.cuda.get_device_properties(0).multi_processor_count
                splits = min(32, _next_pow2(max(1, math.ceil(2 * sm / bh))))
            # paged 路径按 page 切分；split 超过 page 数只会产生空 split
            max_reasonable = _prev_pow2(max(1, (s + 16 - 1) // 16))
            if splits > max_reasonable:
                print(f"[info] splits 从 {splits} 裁剪到 {max_reasonable}（避免空 split）")
            splits = max(1, min(splits, max_reasonable))
            kc, vc, table = pack_paged(k, v, page_size=16)
            impls["triton_split1"] = lambda: kernels.decode_attention(
                q, k, v, num_splits=1, validate=False
            )
            impls[f"triton_split{splits}"] = lambda: kernels.decode_attention(
                q, k, v, num_splits=splits, validate=False
            )
            impls[f"paged_split{splits}"] = lambda: kernels.decode_attention(
                q,
                kc,
                vc,
                page_table=table,
                seq_len=s,
                page_size=16,
                num_splits=splits,
                validate=False,
            )
            print(f"num_splits={splits}")
        else:
            print("[skip] 未检测到 GPU + Triton，仅比较 eager / SDPA。")

        for name, fn in impls.items():
            out = fn()
            ok, _ = check_error(out, ref, atol, rtol, name=name)
            if not ok:
                print(f"[error] {name} 未通过正确性检查，停止。")
                return 1

        # 显存：静态分解（work buffer 只对 triton 路径计 part_acc/m/l）
        static_mem = {
            "q_mb": tensor_bytes(q) / 1e6,
            "k_cache_mb": tensor_bytes(k) / 1e6,
            "v_cache_mb": tensor_bytes(v) / 1e6,
            "out_mb": tensor_bytes(q) / 1e6,
        }
        if has_gpu_kernels:
            # part_acc [BH,splits,BLOCK_D] fp32 + part_m/part_l [BH,splits] fp32
            partial_mb = (bh * splits * (_next_pow2(args.dim) + 2) * 4) / 1e6
            static_mem[f"partial_mb(splits={splits})"] = partial_mb
        print("静态显存分解(MB): " + "  ".join(f"{k}={v:.2f}" for k, v in static_mem.items()))
        peak_mem_mb: dict[str, float] = {}
        if device.type == "cuda":
            for name, fn in impls.items():
                peak_mem_mb[name] = measure_peak_memory(fn, device)["extra_mb"]
                print(f"peak_extra_mem {name}: {peak_mem_mb[name]:.2f} MB")

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
            print(f"[round {rr + 1}/{args.repeat}] " + "  ".join(f"{k_}={v_:.4f}ms" for k_, v_ in lat.items()))
        print("延迟分布(首轮采样, ms): " + "  ".join(
            f"{name}[mean={d['mean_ms']:.4f},min={d['min_ms']:.4f},max={d['max_ms']:.4f},std={d['std_ms']:.4f}]"
            for name, d in latency_detail.items()
        ))

        names = list(impls.keys())
        base = [rd["eager"] for rd in rounds]
        rows, metrics = [], {}
        for name in names:
            ms = [rd[name] for rd in rounds]
            median_ms = sorted(ms)[len(ms) // 2]
            speedups = [b / v_ for b, v_ in zip(base, ms)]
            median_speedup = sorted(speedups)[len(speedups) // 2]
            tokens_per_s = args.batch * 1e3 / median_ms  # decode 每步生成 B 个 token
            rows.append(
                [
                    name,
                    f"{median_ms:.4f}",
                    f"{gbs(nbytes, median_ms):.1f}",
                    f"{peak_ratio(nbytes, median_ms, peak) * 100:.1f}%",
                    f"{tokens_per_s:.0f}",
                    f"{peak_mem_mb.get(name, 0.0):.1f}",
                    f"{median_speedup:.2f}x",
                    f"{cv_percent(ms):.2f}%",
                ]
            )
            metrics[name] = {
                "median_ms": median_ms,
                "gbs": gbs(nbytes, median_ms),
                "peak_ratio": peak_ratio(nbytes, median_ms, peak),
                "tokens_per_s": tokens_per_s,
                "peak_extra_mem_mb": peak_mem_mb.get(name, 0.0),
                "latency_detail_ms": latency_detail.get(name, {}),
                "speedup_vs_eager": median_speedup,
                "cv_percent": cv_percent(ms),
            }
        print_table(["impl", "ms", "GB/s", "peak%", "tok/s", "memMB", "vs_eager", "CV"], rows)
        results[s] = {
            "bytes_mb": nbytes / 1e6,
            "memory_static_mb": static_mem,
            "metrics": metrics,
            "rounds": rounds,
        }

    if args.save:
        save_result(
            "task3_decode_attention",
            {
                "config": {
                    "batch": args.batch,
                    "heads": args.heads,
                    "kv_heads": args.kv_heads,
                    "dim": args.dim,
                    "dtype": dtype_name(dtype),
                },
                "peak_bw_gbs": peak,
                "device": device_info(),
                "contexts": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
