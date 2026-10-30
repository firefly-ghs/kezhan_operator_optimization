# 任务二：融合 Residual Add + RMSNorm

## 1. 任务定义

Transformer 每个 block 的残差流更新：

```text
h   = x + residual
out = h * rsqrt(mean(h²) + eps) * g
```

| 项 | 内容 |
| --- | --- |
| 输入 | `x [T, H]`、`residual [T, H]`、`weight [H]`，均为 fp16 |
| 输出 | `out [T, H]`；`write_residual=True` 时额外输出 `h [T, H]` |
| Baseline | `h = x + r` 与 `F.rms_norm(h, (H,), weight, eps)` 两个 kernel |
| 优化目标 | norm-only ≥ 1.5×，带残差写回 ≥ 1.25×；有效带宽 ≥ 峰值 70%；3 轮 CV ≤ 5% |

## 2. 文件

| 文件 | 内容 |
| --- | --- |
| `reference.py` | 两 kernel baseline、`F.rms_norm` eager 参考、float64 ground truth、`baseline_bytes` / `fused_bytes` |
| `kernels.py` | 单 kernel 融合实现，`WRITE_RES` 编译期分支 |
| `test_correctness.py` | `T ∈ {1,8,64}`、`H ∈ {2048,4096,8192}`、两种模式、`eps ∈ {1e-6,1e-5}` |
| `benchmark.py` | baseline vs fused(write_res=True) vs fused(norm-only)，按各自口径计算 GB/s |

## 3. 瓶颈分析

以每个元素 2B 计，T×H 个元素：

| 实现 | 读 | 写 | 总流量 | 理论加速比 |
| --- | --- | --- | --- | --- |
| baseline（2 kernel） | 2TH + TH + H | TH + TH | ≈ 5TH | 1.00× |
| fused（norm-only） | 2TH + H | TH | ≈ 3TH | 1.67× |
| fused（写回残差） | 2TH + H | 2TH | ≈ 4TH | 1.25× |

- 中间张量 `h` 的“写一次 + 读一次”是纯浪费，融合直接消掉 `2TH`；
- 算术强度极低（每元素几次加法/乘法），**纯带宽受限**；
- decode 时 `T` 很小（1–8）：`T=1, H=4096` 只有 1 个 CTA，此时瓶颈从带宽转为 **kernel 启动开销 + CTA 数不足**；
- 未融合路径的两个 kernel 之间还有一次隐式同步，端到端延迟更高。

## 4. 实现方案

单 kernel `_fused_add_rmsnorm_kernel`：

1. `row = program_id(0)`，一个 program 处理一行，`BLOCK_H = next_power_of_2(H)`；
2. 一次加载 `x`、`residual`（`mask = offs < H`），在寄存器中转 fp32 并相加；
3. `write_residual` 为 `tl.constexpr`：为真时立刻把 `h` 写回残差流；为假时该分支在编译期被消除，不产生任何写流量；
4. `var = tl.sum(h * h) / H`，**除数是真实的 H，不是 BLOCK_H**，避免 padding 影响均值；
5. `rrms = 1.0 / tl.sqrt(var + eps)`（fp32 `sqrt` 而非近似 `rsqrt`，保证与参考一致）；
6. `out = h * rrms * w`，转回输出 dtype 写出。

Host 侧：

- `num_warps = 4`（`BLOCK_H ≤ 2048`）/ `8`（`BLOCK_H > 2048`）；
- `H > 32768` 直接报错并提示改用“分块 + 两遍规约”，不做静默降级；
- 输出与残差写回张量在计时区间外预分配。

## 5. 数值精度

- 加法、平方、均值、`sqrt` 全部 fp32；
- 若用 fp16 累加，`sum(h²)` 在 `H=4096` 时会有明显的精度损失（fp16 只有 10 位尾数，累加 4096 项后相对误差可达 1e-2 量级），因此必须在 fp32 下做；
- 验收：fp16 `atol=5e-3, rtol=1e-2`；fp32 `rtol=1e-5`；并要求与 baseline 的差异不超过 baseline 相对 fp64 误差的 1.5 倍。

本机 CPU 测试覆盖 `(T=4,H=512)`、`(T=3,H=777)`、`(T=8,H=2048)`，`out` 与 `h` 全部 `PASS`，`max_abs ≤ 5.8e-7`。

## 6. 本机实测（CPU-only，fp32）

运行：`python 02_fused_add_rmsnorm/benchmark.py --T 8 --H 4096 --repeat 3 --save`

| impl | MB/iter | ms | GB/s | peak% | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_2kernel | 0.67 | 0.0866 | 7.8 | 53.4% | 1.00x | 4.84% |

说明：

- 本机无 GPU/Triton，融合实现不会出现在表中，GPU 表见第 7 节；
- `T=8` 只有 8 个 CTA，CPU 上也无法体现带宽优势，此数字仅证明框架可跑、baseline 可复现；
- `CV = 4.84%`，接近 5% 阈值，属于小张量在 CPU 上的正常噪声。

## 7. GPU 结果（待填）

一键执行：`bash 02_fused_add_rmsnorm/run_gpu.sh`（`T × H` 全矩阵 + fp32 精度对照）。
完整表格模板见 [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md)。

`--repeat 3 --save` 后补齐：

| T | H | impl | MB/iter | ms | GB/s | peak% | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 1 | 4096 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | 4096 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | 4096 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 8192 | fused_norm_only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | 8192 | fused_write_res | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

需同时记录 GPU 型号与实测峰值带宽，并回答任务书 4.7 的 4 个问题（尤其“为何达不到 3TH/4TH 的理论上限”）。

## 8. 复现命令

```bash
python 02_fused_add_rmsnorm/test_correctness.py
python 02_fused_add_rmsnorm/benchmark.py --T 1  --H 4096 --repeat 3 --save
python 02_fused_add_rmsnorm/benchmark.py --T 8  --H 4096 --repeat 3 --save
python 02_fused_add_rmsnorm/benchmark.py --T 64 --H 8192 --repeat 3 --save
```

## 9. 与规范的偏差

| 项 | 规范要求 | 实际实现 | 原因 |
| --- | --- | --- | --- |
| 多行 per program | “一行（或 BLOCK_T 行）” | 仅一行一 program | decode 场景 `T ≤ 64`，行数远小于 SM 数，一行一 program 已能提供足够 CTA；多行复用 weight 的收益在 `H=4096` 时被寄存器压力抵消，留作 GPU 调优项 |
| `H > 8192` 的分块两遍规约 | 要求说明阈值 | `H > 32768` 直接报错 | 单 program 持有 `BLOCK_H` 个 fp32 元素；`H=8192` 时约 32 KB/行，仍在预算内；超过 32768 需要两遍读，当前未实现，显式报错避免数值/性能陷阱 |
| `rsqrt` 近似 | 要求评估 | 使用 `1/sqrt` | 避免近似指令带来的额外误差，先保证与 fp64 参考的 1.5× 误差约束 |
