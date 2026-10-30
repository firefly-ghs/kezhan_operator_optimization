# 任务四 GPU 实测结果（表格模板）

> 用法：2 卡 GPU 机器上执行 `bash run_gpu.sh`，把终端中位数填入本表；
> 完整数据在 `results/task4_moe_*.json` 与 `results/task4_tp2_*.json`。
> T4-B 需要 2 张同型号 GPU；单卡机器可跳过第 5~8 节。未测项保留 `待填`。
> 吞吐口径：T4-A `tok/s = T / latency`（T 个原始 token 走 top-k 个专家）；T4-B `tok/s = T / total_ms`。

## 三指标速查

### T4-A 主配置（T=2048, E=64, H=4096, I=14336, fp16）

| group | impl | 显存占用 peak MB | 吞吐 tok/s | 延迟 median ms | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | baseline_silu | 待填 | 待填 | 待填 | 1.00x | 待填 |
| gate_up | grouped_silu | 待填 | 待填 | 待填 | 待填 | 待填 |
| gate_up | grouped_plain | 待填 | 待填 | 待填 | 待填 | 待填 |
| down | baseline_down | 待填 | 待填 | 待填 | 1.00x | 待填 |
| down | grouped_down | 待填 | 待填 | 待填 | 待填 | 待填 |

> 常驻显存（权重）与实现无关：三份权重 ≈ 22.5 GB（详见 §1.1）；上表为单次调用的峰值增量。

### T4-B 通信关键场景（T=512, H=4096, I=14336, 2 卡）

| mode | 每卡权重+out MB | 吞吐 tok/s | 延迟 total ms | comm% | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- |
| allreduce_torch | 58.5+8.4 | 待填 | 待填 | 待填 | 待填 | 待填 |
| allreduce_triton | 58.5+8.4 | 待填 | 待填 | 待填 | 待填 | 待填 |
| chunked_overlap | 58.5+8.4 | 待填 | 待填 | 待填 | 待填 | 待填 |
| p2p_reduce | 58.5+8.4 | 待填 | 待填 | 待填 | 待填 | 待填 |
| column_parallel | 58.5+4.2 | 待填 | 待填 | 待填 | 待填 | 待填 |

> `tok/s = T / total_ms`；`speedup = single_gpu_ms / total_ms`；显存列 = 切分后每卡权重 + fp32 输出缓冲（MB，详见 §7）。

## 0. 环境信息

| 项 | 值 |
| --- | --- |
| GPU 型号 × 数量 | 待填 |
| 单卡 SM 数 | 待填 |
| 单卡显存 | 待填 |
| 卡间互联（NVLink / PCIe Genx） | 待填 |
| 实测峰值带宽 BW_peak (GB/s) | 待填 |
| CUDA / Driver | 待填 |
| PyTorch / Triton | 待填 |

---

# 第一部分：T4-A MoE 专家计算

## 1. 显存占用

### 1.1 静态分解（fp16）

| 张量 | 公式 | 主配置 T=2048,E=64,H=4096,I=14336 | 小配置 T=512,E=8 |
| --- | --- | --- | --- |
| sorted_x [M,H], M=T·top_k | M·H·2B | 4096·4096·2B = 33.6 MB | 8.4 MB |
| w_gate [E,I,H] | E·I·H·2B | ≈ 7.5 GB | ≈ 0.94 GB |
| w_up [E,I,H] | E·I·H·2B | ≈ 7.5 GB | ≈ 0.94 GB |
| w_down [E,H,I] | E·H·I·2B | ≈ 7.5 GB | ≈ 0.94 GB |
| y_gate_up 融合 [M,I] | M·I·2B | 117 MB | 15 MB |
| y_gate_up 非融合 [M,2I] | M·2I·2B | 235 MB | 29 MB |

> **主配置三份权重合计 ≈ 22.5 GB（十进制）**，超出 24GB 卡的可用空间时，用 `MOE_EXPERTS=8 bash run_gpu.sh` 或减半 I/H。三份权重是常驻显存，与实现无关；融合实现的差异只在激活（y_gate_up）与临时输出。

### 1.2 实测峰值显存增量（单次调用，MB，终端 `peak_extra_mem` 行）

| group | impl | 主配置 | 小配置 T=512,E=8 |
| --- | --- | --- | --- |
| gate_up | baseline_silu | 待填 | 待填 |
| gate_up | baseline_plain | 待填 | 待填 |
| gate_up | grouped_silu | 待填 | 待填 |
| gate_up | grouped_plain | 待填 | 待填 |
| down | baseline_down | 待填 | 待填 |
| down | grouped_down | 待填 | 待填 |

## 2. 延迟分布（首轮 100 次采样，ms；小配置 T=512,E=8）

| group | impl | mean | min | max | std |
| --- | --- | --- | --- | --- | --- |
| gate_up | baseline_silu | 待填 | 待填 | 待填 | 待填 |
| gate_up | grouped_silu | 待填 | 待填 | 待填 | 待填 |
| gate_up | grouped_plain | 待填 | 待填 | 待填 | 待填 |
| down | baseline_down | 待填 | 待填 | 待填 | 待填 |
| down | grouped_down | 待填 | 待填 | 待填 | 待填 |

## 3. 主配置结果（T=2048, E=64, H=4096, I=14336, fp16）

每专家 token 数：min=待填 / max=待填 / avg=待填
padding 浪费（BLOCK_M=32）：待填
tiles：gate_up_silu=待填 / gate_up_plain=待填 / down=待填
waves（SM=待填）：gate_up_silu=待填 / gate_up_plain=待填 / down=待填

| group | impl | MB/iter | ms | GB/s | peak% | tok/s | memMB | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | baseline_silu | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| gate_up | baseline_plain | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| gate_up | grouped_silu | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| gate_up | grouped_plain | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| down | baseline_down | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| down | grouped_down | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 4. 小配置结果（T=512, E=8, H=4096, I=14336）

| group | impl | ms | GB/s | peak% | tok/s | memMB | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | baseline_silu | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| gate_up | grouped_silu | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| gate_up | grouped_plain | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| down | baseline_down | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| down | grouped_down | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 5. BLOCK_M / BLOCK_N 扫描（T=512, E=8，`grouped_silu`；tile 数与 padding 浪费在终端输出）

| BLOCK_M | ms | GB/s | peak% | tok/s | CV | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 待填 | 待填 | 待填 | 待填 | 待填 | tl.dot 下限 |
| 32 | 待填 | 待填 | 待填 | 待填 | 待填 | 默认 |
| 64 | 待填 | 待填 | 待填 | 待填 | 待填 | |

| BLOCK_N | ms | GB/s | peak% | tok/s | CV |
| --- | --- | --- | --- | --- | --- |
| 64 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 128 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 256 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 6. T4-A 验收检查

| 项 | 目标 | 实测 | 是否达标 |
| --- | --- | --- | --- |
| grouped vs per-expert（T=2048,E=64） | ≥ 1.5× | 待填 | 待填 |
| SM 非空率（主要 wave） | ≥ 60% | 待填 | 待填 |
| 3 轮 CV | ≤ 5% | 待填 | 待填 |

---

# 第二部分：T4-B 双 GPU 拆分投影

## 7. 显存占用（单卡静态分解，H=4096, I=14336, fp16）

| 张量 | T=1 | T=8 | T=64 | T=512 |
| --- | --- | --- | --- | --- |
| x [T,I]（激活广播） | 29 KB | 229 KB | 1.8 MB | 14.7 MB |
| W [H,I]（单卡全量基线） | 117 MB | 117 MB | 117 MB | 117 MB |
| K 切分：x_shard / w_shard | 各半份 | 各半份 | 各半份 | 各半份 |
| 列切分：w_rows | 58.5 MB | 58.5 MB | 58.5 MB | 58.5 MB |
| out [T,H]（fp32 partial） | 16 KB | 131 KB | 1 MB | 8.4 MB |
| 通信缓冲（all_reduce/chunk） | 与 out 同规模 | 同左 | 同左 | 同左 |

> 收益口径：K 切分把每卡权重常驻从 117 MB 减半（半精度下如此；若用 fp16 权重 + fp32 累加，激活部分以 out 为准）。

## 8. T4-B 端到端结果（H=4096, I=14336, world_size=2）

| T | mode | local_ms | comm_ms | total_ms | comm% | tok/s | single_gpu_ms | speedup | eff | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | allreduce_torch | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | p2p_reduce | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | column_parallel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | allreduce_torch | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | p2p_reduce | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | column_parallel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | allreduce_torch | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | p2p_reduce | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | column_parallel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | allreduce_torch | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | p2p_reduce | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | column_parallel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

> `local_ms / comm_ms` 单独计时，`total_ms` 是端到端（重叠模式下 total < local + comm）。
> `speedup = single_gpu_ms / total_ms`，`eff = speedup / 2`，`comm% = comm_ms / total_ms`。

**延迟分布（total_ms 首轮采样，终端「延迟分布」行，ms）**

| T | mode | mean | min | max | std | median(=total_ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 |

## 9. chunk 扫描与吞吐汇总

| chunks（T=512） | comm% | speedup | 相对 chunks=1 | CV |
| --- | --- | --- | --- | --- |
| 1 | 待填 | 待填 | 1.00x | 待填 |
| 2 | 待填 | 待填 | 待填 | 待填 |
| 4 | 待填 | 待填 | 待填 | 待填 |
| 8 | 待填 | 待填 | 待填 | 待填 |

吞吐（chunked_overlap 最佳配置，tok/s）：T=1 待填 / T=8 待填 / T=64 待填 / T=512 待填

## 10. T4-B 验收检查

| 项 | 目标 | 实测 | 是否达标 |
| --- | --- | --- | --- |
| T=1/8 端到端 | ≥ 1.6× 单卡 | 待填 | 待填 |
| T=64/512 overlap vs non-overlap | ≥ 1.2× | 待填 | 待填 |
| 通信占比（T=64/512） | ≤ 30% | 待填 | 待填 |
| 3 轮 CV | ≤ 5% | 待填 | 待填 |

---

## 11. 结论与必答问题

### T4-A

#### Q1 小 M GEMM 为什么利用率低？tensor core 最小 tile 粒度与 padding 浪费？

答：待填

#### Q2 tile map 相比 `grid=(E, max_m_tiles, n_tiles)` 省掉多少空 tile？

答：待填

#### Q3 gate/up 融合省了多少 x 读取和输出写入？理论流量比？

答：待填

#### Q4 `[E, N, K]` 与 `[E, K, N]` 的合并访存差异？

答：待填

### T4-B

#### Q1 K 切分 vs N 切分的通信差异与适用并行策略？

答：待填

#### Q2 2 卡 ring all-reduce 的实际通信量与 T·H 的关系？

答：待填

#### Q3 T=1 通信几乎免费、T=512 必须重叠的临界点估算？

答：待填

#### Q4 chunk 数如何选？过小/过大的问题？

答：待填