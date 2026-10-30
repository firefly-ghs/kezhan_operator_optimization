# 任务三：Decode Attention 优化

## 1. 任务定义

Decode 时 `q` 只有 1 个 token，但必须读完整段 KV cache：

```text
scores = q · Kᵀ / sqrt(D)      # [HQ, S]
p      = softmax(scores)
out    = p · V                 # [HQ, D]
```

| 项 | 内容 |
| --- | --- |
| 输入 | `q [B, HQ, D]`；dense KV `[B, S, HKV, D]` 或 paged KV `[num_pages, page_size, HKV, D]` + `page_table [B, max_pages]` |
| 输出 | `out [B, HQ, D]` |
| Baseline | eager（`q@Kᵀ` → `softmax` → `@V`，3 个以上 kernel）与 `F.scaled_dot_product_attention` |
| 优化目标 | S ≥ 2048 时 ≥ 3× eager；≥ 0.9× SDPA；S=32768 时 split ≥ 1.4× 非 split；有效带宽 ≥ 峰值 60%；CV ≤ 5% |

## 2. 文件

| 文件 | 内容 |
| --- | --- |
| `reference.py` | eager / SDPA / float64 参考、dense KV 构造、`pack_paged`、`paged_to_dense`、`attention_bytes` |
| `kernels.py` | 连续 KV 单 split / 多 split、paged KV、combine kernel |
| `test_correctness.py` | GQA/MHA、非对齐长度、`S=0/1` 等边界；无 GPU 自动 skip |
| `benchmark.py` | 按上下文长度扫描 eager / SDPA / triton split1 / splitN / paged |

## 3. 瓶颈分析

**访存**：

```text
bytes = 2·B·S·HKV·D·itemsize + 2·B·HQ·D·itemsize
```

KV 读取占绝对主导，且只与 `HKV` 有关——GQA 下 `HQ/HKV` 个 q head 共享同一份 KV，复用次数 = `HQ/HKV`。

**算术强度**：`FLOPs = 4·B·HQ·S·D`（含 score 与 context 两次乘加）。

- `HQ/HKV=4, D=128`：`AI ≈ 2 FLOP/Byte`，远低于 GPU 脊点，**带宽受限**；
- 但真正的限制往往不是带宽，而是 **并行度**：grid 只有 `B × HQ`（`B=1, HQ=32` → 32 个 CTA），A100 有 108 SM，长上下文时 GPU 大部分 SM 空闲；
- 因此长上下文必须做 **flash-decoding 式 split**：把 S 切成 `num_splits` 段，每段独立做 online softmax 得到 `(m_i, l_i, acc_i)`，再用 combine kernel 合并；
- paged KV 额外引入 page table 间接寻址，读放大与地址计算是次要开销，但 page 边界会破坏连续性，需要按页粒度切分。

**split 的拐点**：当 `B×HQ < 2×SM` 时，非 split 的 CTA 数不足以填满 GPU，split 收益随 S 线性增长；当 `B×HQ ≥ 2×SM` 时，split 只增加 partial 写读与 combine 开销，应保持 `num_splits=1`。

## 4. 实现方案

### v0 单 program per (b, h)（`_decode_attn_contig_kernel`，`num_splits=1`）

- grid `(B×HQ, num_splits)`，`q` 一次加载后常驻寄存器；
- 沿 S 以 `BLOCK_S` 流式读 K/V，online softmax 维护 `m_i`、`l_i`、`acc[D]`，不物化 `[S]` scores；
- `tl.exp` 全部 fp32，越界 token 的 score 置 `-inf`；
- GQA：`hkv = hq // GROUP`，每个 program 只读自己 kv head 的 K/V。

### v1 flash-decoding split（同一 kernel，`num_splits > 1`）

- `chunk = cdiv(S, num_splits)`，每个 split 处理 `[pid_split·chunk, min(...))`；
- 各 split 写 fp32 的 `partial_acc [BH, splits, BLOCK_D]`、`partial_m [BH, splits]`、`partial_l [BH, splits]`；
- `_combine_kernel`：先求 `m_max`，再按 `exp(m_i - m_max)` 加权累加 `acc` 与 `l`，最后 `acc / l` 并转输出 dtype；
- 只支持 2 的幂的 `num_splits`，便于 combine 的循环展开。

### v2 paged KV（`_decode_attn_paged_kernel`）

- 先按 **page 数** 切分：`pages_per_split = cdiv(cdiv(S, PAGE_SIZE), NUM_SPLITS)`，保证 split 边界与页对齐；
- split 内逐页循环：`logical_page → page_table[b, logical_page] → phys`，按 `phys` 读 K/V 页；
- 页内用 `tl.arange(0, PAGE_SIZE)` 展开，尾部 token 用 `sidx < end` 掩码；
- 页号转 `int64` 再算偏移，避免大 cache 下的 32 位溢出。

### 参数

| 参数 | 默认值 | 依据 |
| --- | --- | --- |
| `BLOCK_S` | 64 | dense 路径一次处理 64 个 token，兼顾加载合并度与寄存器 |
| `BLOCK_D` | `next_pow2(D)` | D=128 时正好 128；不 padding 到 256 |
| `num_warps` | 4 | 单 program 只处理 1 个 q head，4 warps 足够 |
| `num_splits` | 1（kernel）/ 自适应（benchmark） | benchmark 用 `min(32, next_pow2(2·SM/(B·HQ)))`，即目标约 2 个 CTA/SM |

## 5. 正确性

- 参考：float64 全量 attention；
- fp16 容差 `atol=rtol=2e-2`，fp32 容差 `1e-4`；
- 覆盖：GQA `(B=2,HQ=8,HKV=2,S=512,D=64)`、MHA `(1,4,4,1000,128)`、非对齐 `(1,4,2,257,64)`、`S=1`、`S=0`；GPU 上另有 `S=8192` 与 `HQ/HKV=32/8` 等形状；
- paged 与 dense 的往返一致性单独校验。

本机 CPU 结果：全部 `PASS`（含 `S=0` 返回全零、`S=1` 退化为单点），`max_abs ≤ 2.7e-7`。

## 6. 本机实测（CPU-only，fp32）

运行：`python 03_decode_attention/benchmark.py --contexts 256 1024 --repeat 3 --save`

| S | impl | ms | GB/s | peak% | vs_eager | CV |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | eager | 0.1781 | 12.0 | 81.8% | 1.00x | 16.01% |
| 256 | sdpa | 0.1164 | 18.3 | 125.1% | 1.62x | 8.68% |
| 1024 | eager | 0.4635 | 18.2 | 124.3% | 1.00x | 5.45% |
| 1024 | sdpa | 0.4112 | 20.5 | 140.1% | 1.10x | 6.64% |

说明：

- 本机无 GPU/Triton，Triton 实现与 paged 路径不参与计时，仅做 import/skip；
- CPU 上的 `peak%` 同样受 copy 峰值口径影响，只作参考；
- `S=256` 的 CV 达到 16%，是 CPU 小张量调度噪声；GPU 验收时必须重新确认 ≤ 5%。

## 7. GPU 结果（待填）

一键执行：`bash 03_decode_attention/run_gpu.sh`（GQA/MHA/D=64/B=8 + num_splits 扫描，最长 S=32768）。
完整表格模板见 [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md)。

`--repeat 3 --save` 后补齐（benchmark 会自动加入 `triton_split1`、`triton_splitN`、`paged_splitN`）：

| B | HQ/HKV | D | S | impl | ms | GB/s | peak% | vs_eager | vs_sdpa | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32/8 | 128 | 512 | triton_split1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 32/8 | 128 | 2048 | triton_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 32/8 | 128 | 8192 | triton_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 32/8 | 128 | 32768 | triton_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 32/8 | 128 | 32768 | paged_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

同时需回答任务书 5.6 的 4 个问题，并给出 `latency–context` 与 `bandwidth–context` 曲线、split 收益拐点、paged 额外开销百分比。

## 8. 复现命令

```bash
python 03_decode_attention/test_correctness.py
python 03_decode_attention/benchmark.py --contexts 256 1024 --repeat 3 --save
# GPU 机器
python 03_decode_attention/benchmark.py --contexts 512 2048 8192 32768 --repeat 3 --save
python 03_decode_attention/benchmark.py --contexts 32768 --batch 8 --heads 32 --kv-heads 8 --dim 128 --repeat 3
python 03_decode_attention/benchmark.py --contexts 2048 --heads 32 --kv-heads 32 --dim 64 --repeat 3
```

## 9. 与规范的偏差

| 项 | 规范要求 | 实际实现 | 原因 |
| --- | --- | --- | --- |
| 同组 q head 共享 KV tile | 可选进阶 | 未实现 | 先保证每个 head 的 split/combine 正确性；GQA 复用需要把 `GROUP` 个 q head 放进同一 program，属于第二阶段优化 |
| `tl.dot` 计算 score | 可选 | 未使用 | `M=1` 时 tensor core 利用率极低，`tl.sum` 的逐元素乘加已足够 |
| dense split 边界 | 按 `BLOCK_S` 对齐 | 按 `cdiv(S, splits)` 切分，循环内用 `BLOCK_S` 步进 | 尾部由 mask 处理，避免为对齐而引入空 split；paged 路径则强制按 page 对齐 |
