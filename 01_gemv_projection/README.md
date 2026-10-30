# 任务一：Decode 投影 GEMV 优化

## 1. 任务定义

Decode 阶段每个 step 只处理 1 个 token，投影退化为矩阵-向量乘：

```text
y = W x        W: [N, K]   x: [K]   y: [N]
```

| 项 | 内容 |
| --- | --- |
| 输入 | `W [N, K]`（K 连续，row-major）、`x [K]` |
| 输出 | `y [N]`，内部 fp32 累加，最终转回 dtype |
| Baseline | `torch.nn.functional.linear(x, W)`（cuBLAS GEMV） |
| 优化目标 | 有效带宽 ≥ 实测峰值 70%；相对 baseline 平均 ≥ 1.3×，至少一组 ≥ 1.5×；3 轮 CV ≤ 5% |

## 2. 文件

| 文件 | 内容 |
| --- | --- |
| `reference.py` | `make_inputs`、`torch_baseline`、float64 参考、`gemv_bytes` 流量公式 |
| `kernels.py` | Triton v0 naive / v1 tiled / v2 split-K + deterministic reduce |
| `test_correctness.py` | CPU 小形状（含非 2 的幂）+ GPU 完整形状；无 GPU 自动 skip Triton 路径 |
| `benchmark.py` | baseline vs 三种 kernel，输出 ms / GB/s / peak% / speedup / CV，`--repeat`、`--save` |

## 3. 瓶颈分析

**访存流量**（`gemv_bytes`）：

```text
bytes = (N·K + K)·itemsize + N·out_itemsize
```

**算术强度**：`FLOPs = 2NK`。

- fp16、`N=K=4096`：`bytes ≈ 33.6 MB`，`FLOPs ≈ 33.6 MFLOP`，`AI ≈ 1.0 FLOP/Byte`；
- A100 fp16 的机器脊点约 `312 TFLOPS / 2.0 TB/s ≈ 156`，**AI 比脊点低两个数量级**，属于纯带宽受限；
- 因此优化目标不是“少算”，而是“把每一份权重字节都用在一次有效计算上，并把请求打满显存”——即合并访存、足够多的在飞请求、足够的 CTA 数。

**并行度**：可并行的最小粒度是 `N` 行（每行输出互不依赖）。

- 网格 `ceil(N / BLOCK_N)` 个 CTA；`N=2048, BLOCK_N=8` 只有 256 个 CTA，A100 108 SM 时约 2.4 CTA/SM，短 kernel 下容易出现尾部拖尾；
- 每个 CTA 沿 K 串行循环，若 K 很大、BLOCK_N 很小，单 CTA 的指令级并行不足以掩盖 HBM 延迟；
- 权重是纯流式读（无复用），`x` 只有 `K·2B`（4–16 KB），可常驻 L2，被所有 CTA 反复命中。

结论：瓶颈是 **HBM 带宽 + CTA 数量/在飞请求不足**，不是算力，也不是单个 CTA 的指令效率。

## 4. 实现方案

### v0 naive（`gemv_naive`）

- grid `(N,)`，一个 program 处理 1 行；
- `acc` 是长度为 `BLOCK_K` 的 fp32 向量，沿 K 以 `BLOCK_K` 步长累加，最后一次 `tl.sum` 归约；
- 优点：实现最简单、CTA 数最多（= N）；缺点：每个 CTA 都重复读 `x` 的同一 tile（靠 L2 吸收），且每个 CTA 只有 1 个输出，权重的行内连续访问虽然合并，但跨 CTA 的调度开销占比高。

### v1 tiled（`gemv_tiled`）

- grid `(ceil(N/BLOCK_N),)`，一个 program 处理 `BLOCK_N` 行；
- 每个 K tile 只加载一次 `x`（`[BLOCK_K]`），与 `W` 的 `[BLOCK_N, BLOCK_K]` tile 做外积后按行规约；
- 收益：`x` 的加载次数从 `N` 次降到 `ceil(N/BLOCK_N)` 次；单 CTA 工作量增大，摊薄调度开销；
- 代价：寄存器/占用率随 `BLOCK_N × BLOCK_K` 增长。

### v2 split-K（`gemv_splitk` + `_splitk_reduce_kernel`）

- grid `(ceil(N/BLOCK_N), SPLIT_K)`；K 被切成 `SPLIT_K` 段，每段独立累加；
- 各 split 写 `partial [SPLIT_K, N]` fp32，再由 reduce kernel 一次读回并求和、转输出 dtype；
- 用确定性 reduce 而非 `atomic_add`，保证同样输入多次运行结果逐位一致（可复现）；
- 额外流量 `SPLIT_K × N × 4B`（写 + 读共 2 份），相对原始 `N·K·2B` 通常 < 1–2%，但多了一次 kernel 启动。

### 默认参数

| 参数 | 默认值 | 依据 |
| --- | --- | --- |
| `BLOCK_N` | 8 | `BLOCK_N=1` 等于 naive；`16` 时寄存器压力明显上升，`8` 在多数形状下平衡最好 |
| `BLOCK_K` | 256 | 256 个 fp16 = 512B/行，满足 128-bit 向量化，且循环次数少 |
| `SPLIT_K` | 4 | `N` 方向 CTA 数不足 SM 数 2–4 倍时补并行度；再大则 partial 流量与启动开销上升 |
| `num_warps` | 4 | 单 program 处理 8 行，4 warps 足以覆盖 `BLOCK_N × BLOCK_K` 的加载 |

`SPLIT_K` 的选择建议（benchmark 中可扫描）：

```text
SPLIT_K = clamp(ceil(target_ctas / ceil(N / BLOCK_N)), 1, 8)
target_ctas ≈ 2 × SM 数
```

## 5. 正确性

- 参考：float64 `W.double() @ x.double()`；
- 通过条件：`max_abs ≤ atol + rtol·max|ref|`，`atol = rtol = 1e-2`；
- 覆盖形状：
  - CPU：`(N=256,K=512)`、`(100,1000)`、`(513,777)`、`(2048,2048)`；
  - GPU：`(2048,2048)`、`(4096,4096)`、`(11008,4096)`、`(2048,8192)`。

本机 CPU 结果：全部 `PASS`，`max_abs` 在 `1e-7 ~ 1.3e-6` 量级，余弦相似度 1.000000。

## 6. 本机实测（CPU-only，fp32）

运行：`python 01_gemv_projection/benchmark.py --N 4096 --K 4096 --repeat 3 --save`

| impl | ms | GB/s | peak% | speedup | CV |
| --- | --- | --- | --- | --- | --- |
| torch_baseline | 3.7089 | 18.1 | 126.2% | 1.00x | 2.24% |

说明：

- 本机无 GPU/Triton，表内只有 baseline；三种 Triton kernel 在 GPU 机器上会自动加入同一张表；
- 实测“峰值带宽”用 D2D copy 得到约 14.3 GB/s，而 GEMV 是读主导且 `x` 常驻 cache，所以 `peak% > 100%` 是 CPU 口径的必然偏差。**CPU 数字只用于验证框架与 baseline，不作为验收依据**；
- `CV = 2.24% < 5%`，说明 CPU 计时框架稳定。

## 7. GPU 结果（待填）

一键执行：`bash 01_gemv_projection/run_gpu.sh`（主形状 + fp32 对照 + BLOCK_N/BLOCK_K/SPLIT_K 扫描，全部 `--repeat 3 --save`）。
完整表格模板见 [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md)。

在有 GPU + Triton 的机器上执行 `--repeat 3 --save` 后，把下表补齐（benchmark 已按同样列输出）：

| shape | impl | ms | GB/s | peak% | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- |
| N=4096,K=4096 | triton_naive | 待填 | 待填 | 待填 | 待填 | 待填 |
| N=4096,K=4096 | triton_tiled | 待填 | 待填 | 待填 | 待填 | 待填 |
| N=4096,K=4096 | triton_splitk | 待填 | 待填 | 待填 | 待填 | 待填 |
| N=11008,K=4096 | 最佳实现 | 待填 | 待填 | 待填 | 待填 | 待填 |
| N=2048,K=8192 | 最佳实现 | 待填 | 待填 | 待填 | 待填 | 待填 |

同时需在 GPU 上实测 `peak_bw_gbs`、SM 数、型号，并回答任务书的 4 个必答问题（见 `../TASK_SPEC.md` 3.6）。

## 8. 复现命令

```bash
python 01_gemv_projection/test_correctness.py
python 01_gemv_projection/benchmark.py --N 4096 --K 4096 --repeat 3 --save
python 01_gemv_projection/benchmark.py --N 11008 --K 4096 --block-n 8 --block-k 256 --split-k 4 --repeat 3 --save
python 01_gemv_projection/benchmark.py --N 2048 --K 8192 --split-k 8 --repeat 3 --save
```

## 9. 与规范的偏差

| 项 | 规范要求 | 实际实现 | 原因 |
| --- | --- | --- | --- |
| cache / eviction hint | 建议 `evict_first` / `.ca` | 未使用 | 不同 Triton 版本对 `eviction_policy`、`cache_modifier` 支持不一致；`x` 仅 4–16 KB 已可被 L2 吸收，收益需在目标 GPU 上实测后再加 |
| `tl.multiple_of` 对齐提示 | 建议 | 未使用 | K 维连续、`BLOCK_K` 为 2 的幂，编译器已能推断对齐；保留作为 GPU 调优项 |
| CPU 计时口径 | 报告 `peak_ratio` | 保留但不作为验收 | CPU 的 copy 峰值不含“读主导 + cache 命中”场景，`peak%` 会 > 100%，已在第 6 节说明 |
