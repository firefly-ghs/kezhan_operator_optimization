"""任务三正确性测试：eager / SDPA / Triton(split) / paged KV vs float64。"""

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
    float64_reference,
    make_inputs,
    pack_paged,
    paged_to_dense,
    sdpa_attention,
    torch_eager_attention,
)

import kernels


def main() -> int:
    seed_all(0)
    device = get_device()
    has_gpu = CUDA_AVAILABLE and require_triton_gpu("Triton decode attention")
    dtype = torch.float16 if has_gpu else torch.float32
    atol = 2e-2 if dtype == torch.float16 else 1e-4
    rtol = 2e-2 if dtype == torch.float16 else 1e-4

    gpu_cases = [
        (1, 32, 8, 512, 128),
        (1, 32, 8, 2048, 128),
        (1, 32, 8, 8192, 128),
        (2, 16, 4, 1024, 64),
        (1, 32, 32, 777, 128),
    ]
    cpu_cases = [
        (2, 8, 2, 512, 64),
        (1, 4, 4, 1000, 128),
        (1, 4, 2, 257, 64),
        (1, 2, 1, 1, 64),
        (1, 2, 1, 0, 64),
    ]
    cases = gpu_cases if has_gpu else cpu_cases

    print(f"device={device}, dtype={dtype}, cases={len(cases)}")
    failures = 0
    for b, hq, hkv, s, d in cases:
        print(f"\n[case] B={b} HQ={hq} HKV={hkv} S={s} D={d}")
        q, k, v = make_inputs(b, hq, hkv, s, d, dtype=dtype, device=device)
        ref = float64_reference(q, k, v)

        failures += int(not check_error(torch_eager_attention(q, k, v), ref, atol, rtol, "eager")[0])
        failures += int(not check_error(sdpa_attention(q, k, v), ref, atol, rtol, "sdpa")[0])
        if s == 0:
            continue

        if has_gpu:
            for splits in (1, 4, 8):
                if splits > s:
                    continue
                failures += int(
                    not check_error(
                        kernels.decode_attention(q, k, v, num_splits=splits),
                        ref,
                        atol,
                        rtol,
                        f"triton_split{splits}",
                    )[0]
                )

        # paged KV：与 dense Triton / float64 比较
        kc, vc, table = pack_paged(k, v, page_size=16)
        dense_k, dense_v = paged_to_dense(kc, vc, table, s)
        failures += int(
            not check_error(torch_eager_attention(q, dense_k, dense_v), ref, atol, rtol, "paged_dense_roundtrip")[0]
        )
        if has_gpu:
            out_paged = kernels.decode_attention(
                q,
                kc,
                vc,
                page_table=table,
                seq_len=s,
                page_size=16,
                num_splits=4,
            )
            failures += int(not check_error(out_paged, ref, atol, rtol, "triton_paged_split4")[0])
            out_dense = kernels.decode_attention(q, k, v, num_splits=4)
            failures += int(
                not check_error(out_paged, out_dense, 2e-3, 1e-2, "paged_vs_dense")[0]
            )

    print("\n" + ("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
