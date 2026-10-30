"""任务一正确性测试：与 float64 ground truth 比较，覆盖非 2 的幂形状。

CPU-only 环境下只验证 torch baseline；Triton kernel 自动跳过。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.utils import (
    CUDA_AVAILABLE,
    check_error,
    get_device,
    require_triton_gpu,
    seed_all,
)
from reference import float64_reference, make_inputs, torch_baseline

import kernels

ATOL = 1e-2
RTOL = 1e-2


def main() -> int:
    seed_all(0)
    device = get_device()
    has_gpu = CUDA_AVAILABLE and require_triton_gpu("Triton GEMV kernels")

    full_cases = [(2048, 2048), (4096, 4096), (11008, 4096), (2048, 8192)]
    small_cases = [(256, 512), (100, 1000), (513, 777), (2048, 2048)]
    cases = full_cases if has_gpu else small_cases

    dtype = torch.float16 if has_gpu else torch.float32
    print(f"device={device}, dtype={dtype}, cases={len(cases)}")

    failures = 0
    for n, k in cases:
        print(f"\n[case] N={n}, K={k}")
        W, x = make_inputs(n, k, dtype=dtype, device=device)
        ref = float64_reference(W, x)

        y = torch_baseline(W, x)
        ok, _ = check_error(y, ref, ATOL, RTOL, name="torch_baseline")
        failures += int(not ok)

        if not has_gpu:
            continue

        configs = [
            ("naive", lambda: kernels.gemv_naive(W, x, block_k=256)),
            ("tiled_n4_k256", lambda: kernels.gemv_tiled(W, x, 4, 256)),
            ("tiled_n8_k256", lambda: kernels.gemv_tiled(W, x, 8, 256)),
            ("tiled_n16_k512", lambda: kernels.gemv_tiled(W, x, 16, 512, num_warps=8)),
            ("splitk_n8_k256_s4", lambda: kernels.gemv_splitk(W, x, 8, 256, 4)),
            ("splitk_n8_k256_s8", lambda: kernels.gemv_splitk(W, x, 8, 256, 8)),
        ]
        for name, fn in configs:
            ok, _ = check_error(fn(), ref, ATOL, RTOL, name=name)
            failures += int(not ok)

    print("\n" + ("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
