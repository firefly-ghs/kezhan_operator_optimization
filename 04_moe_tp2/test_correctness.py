"""任务四-A 正确性测试：MoE grouped GEMM vs per-expert baseline / float64。"""

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
from reference import (
    make_moe_inputs,
    moe_down_baseline,
    moe_down_reference,
    moe_gate_up_baseline,
    moe_gate_up_reference,
)

import kernels_moe


def main() -> int:
    seed_all(0)
    device = get_device()
    has_gpu = CUDA_AVAILABLE and require_triton_gpu("Triton grouped MoE kernels")
    dtype = torch.float16 if has_gpu else torch.float32
    atol = 2e-2 if dtype == torch.float16 else 1e-4
    rtol = 2e-2 if dtype == torch.float16 else 1e-4

    gpu_cases = [
        (8, 256, 1024, 2048, 2, None),
        (16, 512, 2048, 2048, 2, 3),
    ]
    cpu_cases = [
        (4, 32, 128, 256, 2, None),
        (5, 16, 96, 192, 2, 1),
    ]
    cases = gpu_cases if has_gpu else cpu_cases

    print(f"device={device}, dtype={dtype}, cases={len(cases)}")
    failures = 0
    for e, t, h, i, topk, drop in cases:
        print(f"\n[case] E={e} T={t} H={h} I={i} top_k={topk} drop={drop}")
        data = make_moe_inputs(t, h, i, e, topk, dtype, device, drop_expert=drop)
        sx, offs = data["sorted_x"], data["offsets"]
        wg, wu, wd = data["w_gate"], data["w_up"], data["w_down"]
        print(f"  counts={data['counts'].tolist()}  offsets={offs.tolist()}")

        # float64 参考（把权重搬到 CPU double；测试尺寸较小）
        sx64, offs64 = sx.double().cpu(), offs.cpu()
        wg64, wu64, wd64 = wg.double().cpu(), wu.double().cpu(), wd.double().cpu()
        ref_act = moe_gate_up_reference(sx64, wg64, wu64, offs64, act=True)
        ref_plain = moe_gate_up_reference(sx64, wg64, wu64, offs64, act=False)

        base_act = moe_gate_up_baseline(sx, wg, wu, offs, act=True)
        failures += int(not check_error(base_act, ref_act, atol, rtol, "baseline_gate_up_silu")[0])
        base_plain = moe_gate_up_baseline(sx, wg, wu, offs, act=False)
        failures += int(not check_error(base_plain, ref_plain, atol, rtol, "baseline_gate_up_plain")[0])

        ref_down = moe_down_reference(base_act.double().cpu(), wd64, offs64)
        base_down = moe_down_baseline(base_act, wd, offs)
        failures += int(not check_error(base_down, ref_down, atol, rtol, "baseline_down")[0])

        if not has_gpu:
            continue

        tri_silu = kernels_moe.moe_gate_up_silu(sx, wg, wu, offs)
        failures += int(not check_error(tri_silu, ref_act, atol, rtol, "triton_gate_up_silu")[0])
        failures += int(not check_error(tri_silu, base_act, 5e-3, 1e-2, "triton_vs_baseline_silu")[0])

        tri_plain = kernels_moe.moe_gate_up_plain(sx, wg, wu, offs)
        failures += int(not check_error(tri_plain, ref_plain, atol, rtol, "triton_gate_up_plain")[0])

        tri_down = kernels_moe.moe_down(base_act, wd, offs)
        failures += int(not check_error(tri_down, ref_down, atol, rtol, "triton_down")[0])
        failures += int(not check_error(tri_down, base_down, 5e-3, 1e-2, "triton_vs_baseline_down")[0])

    print("\n" + ("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
