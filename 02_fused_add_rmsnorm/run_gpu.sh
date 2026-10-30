#!/usr/bin/env bash
# 任务二：融合 Residual Add + RMSNorm 的 GPU 实测脚本
# 用法（Linux GPU 机器 / WSL2 + CUDA）：bash run_gpu.sh
# 可覆盖：REPEAT（默认 3）、DTYPE（默认 fp16）
set -euo pipefail
cd "$(dirname "$0")"

REPEAT=${REPEAT:-3}
DTYPE=${DTYPE:-fp16}
mkdir -p results

python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"[env] GPU={p.name} SM={p.multi_processor_count} "
      f"mem={p.total_memory / 2**30:.0f}GB torch={torch.__version__}")
try:
    import triton
    print(f"[env] triton={triton.__version__}")
except Exception as exc:
    print(f"[warn] triton 不可用: {exc}")
PY

echo "===== [0/3] 正确性 ====="
python test_correctness.py

echo "===== [1/3] 主矩阵 T × H（${DTYPE}） ====="
for t in 1 8 64; do
  for h in 2048 4096 8192; do
    echo "--- T=${t} H=${h} ---"
    python benchmark.py --T "$t" --H "$h" --dtype "$DTYPE" --repeat "$REPEAT" --save
  done
done

echo "===== [2/3] fp32 精度对照（T=8 H=4096） ====="
python benchmark.py --T 8 --H 4096 --dtype fp32 --repeat "$REPEAT" --save

echo "===== [3/3] decode 小 T 补充（T=1 H=8192） ====="
python benchmark.py --T 1 --H 8192 --dtype "$DTYPE" --repeat "$REPEAT" --save

echo
echo "[done] 结果 JSON 已写入 results/；把终端中位数填入 RESULTS_TEMPLATE.md"
