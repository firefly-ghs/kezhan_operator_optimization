#!/usr/bin/env bash
# 任务三：Decode Attention 的 GPU 实测脚本
# 用法（Linux GPU 机器 / WSL2 + CUDA）：bash run_gpu.sh
# 可覆盖：REPEAT（默认 3）、DTYPE（默认 fp16）、SPLITS（默认 0=按 SM 自适应）
set -euo pipefail
cd "$(dirname "$0")"

REPEAT=${REPEAT:-3}
DTYPE=${DTYPE:-fp16}
SPLITS=${SPLITS:-0}
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

echo "===== [0/5] 正确性 ====="
python test_correctness.py

echo "===== [1/5] GQA 4:1 长上下文（B=1, HQ=32, HKV=8, D=128） ====="
for s in 512 2048 8192 32768; do
  echo "--- S=${s} ---"
  python benchmark.py --contexts "$s" --batch 1 --heads 32 --kv-heads 8 --dim 128 \
    --dtype "$DTYPE" --splits "$SPLITS" --repeat "$REPEAT" --save
done

echo "===== [2/5] 多 batch（B=8, 32/8, D=128） ====="
for s in 2048 8192; do
  echo "--- B=8 S=${s} ---"
  python benchmark.py --contexts "$s" --batch 8 --heads 32 --kv-heads 8 --dim 128 \
    --dtype "$DTYPE" --splits "$SPLITS" --repeat "$REPEAT" --save
done

echo "===== [3/5] MHA 对照（HQ=HKV=32, B=1, D=128） ====="
for s in 2048 8192; do
  echo "--- MHA S=${s} ---"
  python benchmark.py --contexts "$s" --batch 1 --heads 32 --kv-heads 32 --dim 128 \
    --dtype "$DTYPE" --splits "$SPLITS" --repeat "$REPEAT" --save
done

echo "===== [4/5] D=64 对照（B=1, 32/8） ====="
for s in 2048 8192; do
  echo "--- D=64 S=${s} ---"
  python benchmark.py --contexts "$s" --batch 1 --heads 32 --kv-heads 8 --dim 64 \
    --dtype "$DTYPE" --splits "$SPLITS" --repeat "$REPEAT" --save
done

echo "===== [5/5] num_splits 扫描（B=1, 32/8, D=128, S=8192/32768） ====="
for s in 8192 32768; do
  for sp in 1 4 8 16 32; do
    echo "--- S=${s} num_splits=${sp} ---"
    python benchmark.py --contexts "$s" --batch 1 --heads 32 --kv-heads 8 --dim 128 \
      --dtype "$DTYPE" --splits "$sp" --repeat "$REPEAT" --save
  done
done

echo
echo "[done] 结果 JSON 已写入 results/；把终端中位数填入 RESULTS_TEMPLATE.md"
