#!/usr/bin/env bash
# 任务四：MoE 专家计算 + 双 GPU 拆分投影的 GPU 实测脚本
# 用法（Linux GPU 机器，T4-B 需要 2 张同型号卡）：bash run_gpu.sh
# 可覆盖：REPEAT（默认 3）、DTYPE（默认 fp16）、
#         MOE_TOKENS/MOE_EXPERTS/MOE_HIDDEN/MOE_INTERMEDIATE（主配置，默认 2048/64/4096/14336）
# 24GB 显存提示：E=64、I=14336 时三份权重约 22GB，建议 MOE_EXPERTS=8 再跑主配置。
set -euo pipefail
cd "$(dirname "$0")"

REPEAT=${REPEAT:-3}
DTYPE=${DTYPE:-fp16}
TOKENS=${MOE_TOKENS:-2048}
EXPERTS=${MOE_EXPERTS:-64}
HIDDEN=${MOE_HIDDEN:-4096}
INTER=${MOE_INTERMEDIATE:-14336}
mkdir -p results

python - <<'PY'
import torch
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"[env] cuda:{i} GPU={p.name} SM={p.multi_processor_count} "
          f"mem={p.total_memory / 2**30:.0f}GB")
print(f"[env] torch={torch.__version__} device_count={n}")
try:
    import triton
    print(f"[env] triton={triton.__version__}")
except Exception as exc:
    print(f"[warn] triton 不可用: {exc}")
PY

echo "===== [0/6] 正确性（MoE） ====="
python test_correctness.py

echo "===== [1/6] T4-A 主配置（T=${TOKENS} E=${EXPERTS} H=${HIDDEN} I=${INTER}） ====="
python benchmark.py --tokens "$TOKENS" --experts "$EXPERTS" --hidden "$HIDDEN" \
  --intermediate "$INTER" --dtype "$DTYPE" --repeat "$REPEAT" --save

echo "===== [2/6] T4-A 小配置（T=512 E=8） ====="
python benchmark.py --tokens 512 --experts 8 --hidden "$HIDDEN" --intermediate "$INTER" \
  --dtype "$DTYPE" --repeat "$REPEAT" --save

echo "===== [3/6] T4-A BLOCK_M 扫描（T=512 E=8） ====="
for bm in 16 32 64; do
  echo "--- BLOCK_M=${bm} ---"
  python benchmark.py --tokens 512 --experts 8 --hidden "$HIDDEN" --intermediate "$INTER" \
    --dtype "$DTYPE" --block-m "$bm" --repeat "$REPEAT" --save
done

echo "===== [4/6] T4-A BLOCK_N 扫描（T=512 E=8, BLOCK_M=32） ====="
for bn in 64 128 256; do
  echo "--- BLOCK_N=${bn} ---"
  python benchmark.py --tokens 512 --experts 8 --hidden "$HIDDEN" --intermediate "$INTER" \
    --dtype "$DTYPE" --block-m 32 --block-n "$bn" --repeat "$REPEAT" --save
done

run2() {
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node=2 run_two_gpu.py "$@"
  else
    echo "[warn] 未找到 torchrun，回退 python --nproc 2"
    python run_two_gpu.py --nproc 2 "$@"
  fi
}

echo "===== [5/6] T4-B 端到端（K 切分 + 各通信模式，T=1/8/64/512） ====="
for t in 1 8 64 512; do
  echo "--- T=${t} modes=all ---"
  run2 --backend nccl --tokens "$t" --hidden 4096 --intermediate 14336 \
    --dtype "$DTYPE" --modes all --repeat "$REPEAT" --save
done

echo "===== [6/6] T4-B chunk 扫描（T=512, chunked_overlap） ====="
for ch in 1 2 4 8; do
  echo "--- chunks=${ch} ---"
  run2 --backend nccl --tokens 512 --hidden 4096 --intermediate 14336 \
    --dtype "$DTYPE" --modes chunked_overlap --chunks "$ch" --repeat "$REPEAT" --save
done

echo
echo "[done] 结果 JSON 已写入 results/；把终端中位数填入 RESULTS_TEMPLATE.md"
