# 任务二 GPU 实测结果（表格模板）

> 用法：GPU 机器上执行 `bash run_gpu.sh`，把终端中位数填入本表；完整数据在 `results/task2_add_rmsnorm_*.json`。
> `fused_*` 行只在检测到 GPU + Triton 时出现；未测项保留 `待填`。
> 列含义：`MB`=每次迭代访存流量，`tok/s`=吞吐（每次迭代处理 T 行），`memMB`=单次调用峰值显存增量。

## 三指标速查（Memory / Throughput / Latency · 主配置 T=8, H=4096, fp16）

| impl | 显存占用 peak MB | 吞吐 tok/s | 延迟 median ms | 延迟 mean ms | 延迟 min–max ms | 轮间 CV |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

> 数据来源：`peak_extra_mem`（实测峰值显存增量）│ `tok/s`（每迭代 T 行）│ 3 轮中位数与首轮采样 mean/min/max（终端「延迟分布」行）。

## 0. 环境信息

| 项 | 值 |
| --- | --- |
| GPU 型号 / 显存 | 待填 |
| SM 数 | 待填 |
| 实测峰值带宽 BW_peak (GB/s) | 待填 |
| CUDA / Driver | 待填 |
| PyTorch / Triton | 待填 |

## 1. 显存占用

### 1.1 静态分解（T×H×2B，fp16）

| 张量 | T=1,H=4096 | T=8,H=4096 | T=64,H=8192 |
| --- | --- | --- | --- |
| x / residual / out / res_out（各） | 8 KB | 64 KB | 1 MB |
| weight [H] | 8 KB | 8 KB | 16 KB |
| baseline 中间 h（未融合时） | 8 KB | 64 KB | 1 MB |
| fused 额外分配 | 0（原地） | 0（原地） | 0（原地） |

> fused 相比 baseline 省掉的显存 = 中间张量 `h`（T×H×2B）；无临时缓冲区。

### 1.2 实测峰值显存增量（单次调用，MB，终端 `peak_extra_mem` 行）

| impl | T=1,H=4096 | T=8,H=4096 | T=64,H=8192 |
| --- | --- | --- | --- |
| baseline_2kernel | 待填 | 待填 | 待填 |
| fused_write_res | 待填 | 待填 | 待填 |
| fused_norm_only | 待填 | 待填 | 待填 |

## 2. 延迟分布（首轮 100 次采样，ms，终端「延迟分布」行）

| T | H | impl | mean | min | max | std |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 |
| 1 | 4096 | fused_write_res | 待填 | 待填 | 待填 | 待填 |
| 1 | 4096 | fused_norm_only | 待填 | 待填 | 待填 | 待填 |
| 8 | 4096 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 |
| 8 | 4096 | fused_write_res | 待填 | 待填 | 待填 | 待填 |
| 8 | 4096 | fused_norm_only | 待填 | 待填 | 待填 | 待填 |

## 3. 主矩阵（fp16）

| T | H | impl | MB/iter | ms | GB/s | peak% | tok/s | memMB | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 1 | 2048 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 2048 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 4096 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 1 | 4096 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 4096 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 8192 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 1 | 8192 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 8192 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | 2048 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 8 | 2048 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | 2048 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | 4096 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 8 | 4096 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | 4096 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | 8192 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 8 | 8192 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | 8192 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 2048 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 64 | 2048 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 2048 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 4096 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 64 | 4096 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 4096 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 8192 | baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 64 | 8192 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 8192 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 4. 吞吐量汇总（tok/s）

| impl | T=1,H=4096 | T=8,H=4096 | T=64,H=8192 |
| --- | --- | --- | --- |
| baseline_2kernel | 待填 | 待填 | 待填 |
| fused_write_res | 待填 | 待填 | 待填 |
| fused_norm_only | 待填 | 待填 | 待填 |

## 5. fp32 精度对照（T=8, H=4096）

| impl | MB/iter | ms | GB/s | peak% | tok/s | memMB | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_2kernel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 6. 验收检查

| 项 | 目标 | 实测 | 是否达标 |
| --- | --- | --- | --- |
| `fused_norm_only` 加速比 | ≥ 1.5× | 待填 | 待填 |
| `fused_write_res` 加速比 | ≥ 1.25× | 待填 | 待填 |
| 有效带宽（最佳） | ≥ 70% BW_peak | 待填 | 待填 |
| 3 轮 CV | ≤ 5% | 待填 | 待填 |
| 数值精度 | 误差 ≤ 1.5× baseline 相对 fp64 误差 | 待填 | 待填 |
| 显存 | fused 不再持有中间张量 h（省 T×H×2B/iter） | 待填 | 待填 |

## 7. 结论与必答问题

### Q1 实测是否达到 3TH/4TH 的 roofline？差在哪？

答：待填

### Q2 为什么 mean 必须在 fp32 下做？fp16 累加的误差量级？

答：待填

### Q3 H=4096 时 BLOCK_H=next_pow2(H) 的寄存器压力与 num_warps 选择

答：待填

### Q4 残差流必须写回时还有哪些优化空间？

答：待填