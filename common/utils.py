"""跨任务公共工具：设备探测、Triton 可用性、稳定计时、误差统计、峰值带宽实测。

所有脚本在 CPU-only 环境下也应可导入、可运行；GPU/Triton 路径通过
``require_triton_gpu()`` 显式跳过。
"""

from __future__ import annotations

import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

try:  # Triton 只在 GPU 机器上需要；CPU 环境缺少它也不影响参考实现
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    TRITON_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# 设备与运行环境
# --------------------------------------------------------------------------- #
def get_device(prefer_cuda: bool = True) -> torch.device:
    """返回当前可用设备（优先 CUDA）。"""
    return torch.device("cuda" if (prefer_cuda and CUDA_AVAILABLE) else "cpu")


def device_info() -> dict[str, Any]:
    """采集环境信息，用于写入 benchmark 结果。"""
    info: dict[str, Any] = {
        "device": str(get_device()),
        "torch": torch.__version__,
        "triton_available": TRITON_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
    }
    if CUDA_AVAILABLE:
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "sm_count": props.multi_processor_count,
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "l2_cache_mb": round(
                    getattr(props, "L2_cache_size", 0) / (1024 * 1024), 2
                ),
            }
        )
    return info


def require_triton_gpu(feature: str = "该实现") -> bool:
    """检查 Triton + GPU 是否可用；不可用时打印原因并返回 False。"""
    if CUDA_AVAILABLE and TRITON_AVAILABLE:
        return True
    reasons = []
    if not CUDA_AVAILABLE:
        reasons.append("无可用 CUDA 设备")
    if not TRITON_AVAILABLE:
        reasons.append("未安装 triton")
    print(f"[skip] {feature} 需要 GPU + Triton（{'，'.join(reasons)}），已跳过。")
    return False


def seed_all(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device | None = None) -> None:
    device = device or get_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
def _median(samples: Sequence[float]) -> float:
    return float(statistics.median(samples))


def _calibrate_inner_iters(fn: Callable[[], Any], device: torch.device) -> int:
    """让每次采样至少约 1 ms，避免短 kernel 被计时开销淹没。"""
    probe = max(3, 5)
    times = []
    for _ in range(probe):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    base = _median(times)
    if base <= 0:
        return 1
    inner = int(math.ceil(1.0e-3 / base))
    return max(1, min(inner, 10_000))


def bench_callable(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    device: torch.device | None = None,
    inner_iters: int | None = None,
) -> dict[str, float]:
    """稳定计时。

    GPU 使用 CUDA Event；CPU 使用 perf_counter。返回中位数等统计量（毫秒）。
    若单次调用过短，会自动增加内层循环次数（``inner_iters``）再除以次数。
    """
    device = device or get_device()

    for _ in range(warmup):
        fn()
    sync(device)

    if inner_iters is None:
        inner_iters = _calibrate_inner_iters(fn, device)

    samples: list[float] = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(rep):
            start.record()
            for _ in range(inner_iters):
                fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / inner_iters)
    else:
        for _ in range(rep):
            t0 = time.perf_counter()
            for _ in range(inner_iters):
                fn()
            samples.append((time.perf_counter() - t0) / inner_iters * 1e3)

    samples_t = torch.tensor(samples, dtype=torch.float64)
    return {
        "ms": _median(samples),
        "mean_ms": float(samples_t.mean()),
        "min_ms": float(samples_t.min()),
        "max_ms": float(samples_t.max()),
        "std_ms": float(samples_t.std(unbiased=False)),
        "inner_iters": float(inner_iters),
        "samples": float(rep),
    }


def measure_peak_bandwidth(
    device: torch.device | None = None,
    size_mb: int = 256,
    dtype: torch.dtype = torch.float32,
) -> float:
    """用 device-to-device copy 实测峰值带宽（GB/s，十进制）。"""
    device = device or get_device()
    itemsize = torch.tensor([], dtype=dtype).element_size()
    n = int(size_mb * 1024 * 1024 / itemsize)
    src = torch.empty(n, dtype=dtype, device=device)
    dst = torch.empty(n, dtype=dtype, device=device)
    src.fill_(1.0)
    stats = bench_callable(lambda: dst.copy_(src), warmup=10, rep=30, device=device)
    nbytes = 2 * n * itemsize
    return nbytes / (stats["ms"] * 1e-3) / 1e9


# --------------------------------------------------------------------------- #
# 显存
# --------------------------------------------------------------------------- #
def tensor_bytes(t) -> int:
    """张量占用的字节数。"""
    return int(t.numel() * t.element_size())


def measure_peak_memory(
    fn: Callable[[], Any],
    device: torch.device | None = None,
    warmup: int = 10,
) -> dict[str, float]:
    """实测一次调用相对当前 allocated 的峰值显存增量（MB）。

    CPU 或不可用环境下返回全 0。注意：会重置进程级峰值统计
    （``torch.cuda.reset_peak_memory_stats``），只应在 timer 之外使用。
    """
    device = device or get_device()
    if device.type != "cuda" or not CUDA_AVAILABLE:
        return {"base_mb": 0.0, "peak_mb": 0.0, "extra_mb": 0.0}
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "base_mb": base / 1e6,
        "peak_mb": peak / 1e6,
        "extra_mb": max(0.0, (peak - base) / 1e6),
    }


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def gbs(nbytes: int | float, latency_ms: float) -> float:
    """有效带宽（GB/s）。"""
    return float(nbytes) / (latency_ms * 1e-3) / 1e9


def peak_ratio(nbytes: int | float, latency_ms: float, peak_gbs: float) -> float:
    """理论最优延迟 / 实际延迟，即带宽利用率。"""
    ideal_ms = float(nbytes) / (peak_gbs * 1e9) * 1e3
    return ideal_ms / latency_ms


def error_stats(actual, reference) -> dict[str, float]:
    """与 float64 参考比较，返回 max_abs / max_rel / cosine 等指标。"""

    def _f64(x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.detach().to("cpu", torch.float64).reshape(-1)

    a = _f64(actual)
    b = _f64(reference)
    if a.numel() != b.numel():
        raise ValueError(f"shape mismatch: {a.numel()} vs {b.numel()}")
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    denom_norm = b.norm()
    cosine = (
        float((a * b).sum() / (a.norm() * denom_norm))
        if float(a.norm()) > 0 and float(denom_norm) > 0
        else 1.0
    )
    return {
        "max_abs": float(diff.max()),
        "max_rel": float((diff / denom).max()),
        "mean_abs": float(diff.mean()),
        "ref_max_abs": float(b.abs().max()),
        "cosine": cosine,
    }


def check_error(
    actual,
    reference,
    atol: float,
    rtol: float,
    name: str = "",
) -> tuple[bool, dict[str, float]]:
    """混合容差判定：max_abs <= atol + rtol * max|ref|。"""
    stats = error_stats(actual, reference)
    bound = atol + rtol * stats["ref_max_abs"]
    ok = stats["max_abs"] <= bound
    tag = f"[{name}] " if name else ""
    status = "PASS" if ok else "FAIL"
    print(
        f"  {tag}{status} max_abs={stats['max_abs']:.3e} "
        f"(bound={bound:.3e}) max_rel={stats['max_rel']:.3e} "
        f"cos={stats['cosine']:.6f}"
    )
    return ok, stats


def cv_percent(samples: Sequence[float]) -> float:
    """变异系数（百分比）。"""
    s = list(samples)
    if not s:
        return 0.0
    m = statistics.mean(s)
    if m == 0:
        return 0.0
    return statistics.pstdev(s) / m * 100.0


# --------------------------------------------------------------------------- #
# 格式化与结果落盘
# --------------------------------------------------------------------------- #
def print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """打印对齐的文本表格。"""
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for row in str_rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def save_result(task: str, payload: dict[str, Any], outdir: str | Path | None = None) -> Path:
    """把 benchmark 结果写入 ``results/<task>_<timestamp>.json``。"""
    root = Path(__file__).resolve().parents[1]
    outdir = Path(outdir) if outdir is not None else root / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
    path = outdir / f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {path}")
    return path


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[key]


def dtype_name(dtype: torch.dtype) -> str:
    return {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}.get(
        dtype, str(dtype)
    )
