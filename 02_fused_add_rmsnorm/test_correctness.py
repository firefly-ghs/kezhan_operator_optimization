"""任务二正确性测试：融合 kernel vs float64，以及 fused vs unfused 的一致性。"""

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


def main() -> int:
    seed_all(0)
    device = get_device()
    has_gpu = CUDA_AVAILABLE and require_triton_gpu("Triton fused add+rmsnorm")
    dtype = torch.float16 if has_gpu else torch.float32
    cases = (
        [(1, 2048), (8, 4096), (64, 4096), (8, 8192)] if has_gpu else [(4, 512), (3, 777), (8, 2048)]
    )
    atol = 5e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-2 if dtype == torch.float16 else 1e-5

    print(f"device={device}, dtype={dtype}, cases={len(cases)}")
    failures = 0
    for t, h in cases:
        print(f"\n[case] T={t}, H={h}, eps=1e-6")
        x, r, w = make_inputs(t, h, dtype=dtype, device=device)
        ref_out, ref_h = float64_reference(x, r, w)

        out, h_out = torch_baseline(x, r, w)
        failures += int(not check_error(out, ref_out, atol, rtol, "baseline_out")[0])
        failures += int(not check_error(h_out, ref_h, 1e-3 if dtype == torch.float16 else 1e-6, rtol, "baseline_h")[0])

        if not has_gpu:
            continue

        for write_res in (True, False):
            tag = "write_res" if write_res else "norm_only"
            f_out, f_h = kernels.fused_add_rmsnorm(x, r, w, 1e-6, write_res)
            failures += int(not check_error(f_out, ref_out, atol, rtol, f"fused_{tag}_out")[0])
            if write_res:
                failures += int(not check_error(f_h, ref_h, 1e-3, rtol, "fused_write_res_h")[0])
            # 融合实现应与未融合 baseline 数值一致（同 dtype 下差异应很小）
            failures += int(not check_error(f_out, out, 2e-3, 1e-2, f"fused_{tag}_vs_baseline")[0])

        # 额外边界：eps=1e-5
        f_out, _ = kernels.fused_add_rmsnorm(x, r, w, 1e-5, True)
        ref_out2, _ = float64_reference(x, r, w, 1e-5)
        failures += int(not check_error(f_out, ref_out2, atol, rtol, "fused_eps1e-5")[0])

    print("\n" + ("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
