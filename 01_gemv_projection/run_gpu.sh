#!/usr/bin/env bash
# 任务一：GEMV 优化的 GPU 实测脚本
# 用法（Linux GPU 机器 / WSL2 + CUDA）：bash run_gpu.sh
# 可覆盖的环境变量：REPEAT（默认 3）、DTYPE（默认 fp16）
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

echo "===== [0/5] 正确性 ====="
python test_correctness.py

echo "===== [1/5] 主形状（${DTYPE}） ====="
while read -r n k; do
  echo "--- N=${n} K=${k} ---"
  python benchmark.py --N "$n" --K "$k" --dtype "$DTYPE" --repeat "$REPEAT" --save
done <<'SHAPES'
4096 4096
11008 4096
2048 8192
SHAPES

echo "===== [2/5] fp32 对照（N=4096 K=4096） ====="
python benchmark.py --N 4096 --K 4096 --dtype fp32 --repeat "$REPEAT" --save

echo "===== [3/5] BLOCK_N 扫描（N=4096 K=4096, BLOCK_K=256, SPLIT_K=4） ====="
for bn in 1 2 4 8 16; do
  echo "--- BLOCK_N=${bn} ---"
  python benchmark.py --N 4096 --K 4096 --dtype "$DTYPE" \
    --block-n "$bn" --block-k 256 --split-k 4 --repeat "$REPEAT" --save
done

echo "===== [4/5] BLOCK_K 扫描（N=4096 K=4096, BLOCK_N=8, SPLIT_K=4） ====="
for bk in 128 256 512 1024; do
  echo "--- BLOCK_K=${bk} ---"
  python benchmark.py --N 4096 --K 4096 --dtype "$DTYPE" \
    --block-n 8 --block-k "$bk" --split-k 4 --repeat "$REPEAT" --save
done

echo "===== [5/5] SPLIT_K 扫描（N=4096 K=4096, BLOCK_N=8, BLOCK_K=256） ====="
for sk in 1 2 4 8; do
  echo "--- SPLIT_K=${sk} ---"
  python benchmark.py --N 4096 --K 4096 --dtype "$DTYPE" \
    --block-n 8 --block-k 256 --split-k "$sk" --repeat "$REPEAT" --save
done

echo
echo "[done] 结果 JSON 已写入 results/；把终端中位数填入 RESULTS_TEMPLATE.md"
