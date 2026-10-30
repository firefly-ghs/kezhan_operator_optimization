# Decode 阶段算子优化任务书（4 项）

> 目标读者：算子 / 推理优化工程师
> 代码根目录：`kezhan_operator_optimization/`
> 技术栈：PyTorch 2.x + Triton 3.x（GPU kernel），NumPy/PyTorch float64（参考实现）
> 硬件基线：通用 NVIDIA GPU（以 A100/H100 级别为默认，运行时按 `device_properties` 自适应；双卡任务支持 NVLink 或 PCIe）

---

## 0. 总览

大模型 decode 阶段每个 step 只处理 1 个 token（或极少量 token），典型算子的算术强度极低，性能几乎完全由显存带宽、kernel 启动开销和小 M 下的 GPU 利用率决定。本任务书把 4 个独立的优化命题拆成可执行、可验收的工程任务：

| 编号 | 任务 | 主要瓶颈 | 优化手段 | 关键指标 |
| --- | --- | --- | --- | --- |
| T1 | Decode 投影 GEMV（`y = W x`） | 显存带宽 + 并行度不足 | 分块、向量化、split-K、工作划分 | 有效带宽 ≥ 70% 峰值，≥ 1.3× baseline |
| T2 | Residual Add + RMSNorm 融合 | 中间结果读写（3 次流式访存） | kernel 融合、寄存器内规约、fp32 累加 | norm-only ≥ 1.5×，带残差写回 ≥ 1.25× |
| T3 | Decode Attention（读 KV cache） | KV cache 带宽 + 长上下文并行度 | online softmax、flash-decoding split、paged KV | 长上下文 ≥ 3× eager，≥ 0.9× SDPA |
| T4 | MoE 专家计算 + 双 GPU 拆分投影 | 小 M GEMM 利用率、跨卡通信 | grouped GEMM、tile map、gate/up 融合、通信重叠/P2P | MoE ≥ 1.5×，双卡端到端 ≤ 0.6× 单卡 |

4 个任务各自独立成目录，互不依赖；公共的计时、误差统计、峰值带宽测量放在 `common/`。

---

## 1. 运行环境与目录结构

### 1.1 环境

| 项 | 要求 |
| --- | --- |
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.2（建议 2.9） |
| Triton | ≥ 3.0（仅 GPU 路径需要） |
| GPU | 任意支持 Triton 的 NVIDIA GPU；T4-B 需要 2 张同型号 GPU |
| CPU-only 机器 | 参考实现、正确性框架、baseline benchmark 可运行；所有 Triton 路径自动 skip |

安装：

```bash
pip install -r requirements.txt   # torch / numpy / triton
```

### 1.2 目录结构

```text
kezhan_operator_optimization/
├── TASK_SPEC.md                    # 本任务书
├── requirements.txt
├── common/
│   ├── __init__.py
│   └── utils.py                    # 设备探测、计时、误差统计、峰值带宽实测
├── 01_gemv_projection/
│   ├── README.md                   # 任务说明 + 优化记录 + 结果表
│   ├── reference.py                # torch baseline / float64 参考 / 输入构造
│   ├── kernels.py                  # Triton: naive / tiled / split-K
│   ├── test_correctness.py
│   └── benchmark.py
├── 02_fused_add_rmsnorm/
│   ├── README.md
│   ├── reference.py
│   ├── kernels.py                  # Triton: 单 kernel 融合（两种输出模式）
│   ├── test_correctness.py
│   └── benchmark.py
├── 03_decode_attention/
│   ├── README.md
│   ├── reference.py                # eager / SDPA / float64 参考 / KV cache 构造
│   ├── kernels.py                  # 连续 & paged KV，split 与非 split
│   ├── test_correctness.py
│   └── benchmark.py                # 多上下文长度扫描
└── 04_moe_tp2/
    ├── README.md
    ├── reference.py                # 路由/top-k/排序、per-expert baseline、float64 参考
    ├── kernels_moe.py              # grouped GEMM: gate/up、fused SwiGLU、down
    ├── kernels_tp2.py              # 双卡拆分投影的本地 GEMM 与通信原语
    ├── test_correctness.py
    ├── benchmark.py                # 单卡 MoE 扫描
    └── run_two_gpu.py              # 双进程（NCCL 或 gloo）端到端 benchmark
```

### 1.3 运行方式

在本机（CPU-only）：

```bash
python 01_gemv_projection/test_correctness.py
python 01_gemv_projection/benchmark.py --N 4096 --K 4096
python 02_fused_add_rmsnorm/test_correctness.py
python 02_fused_add_rmsnorm/benchmark.py
python 03_decode_attention/test_correctness.py
python 03_decode_attention/benchmark.py --contexts 256 1024
python 04_moe_tp2/test_correctness.py
python 04_moe_tp2/benchmark.py
python 04_moe_tp2/run_two_gpu.py --simulate   # 单进程模拟 2 rank，仅校验切分数值等价性
```

在 GPU 机器上：

```bash
python 01_gemv_projection/benchmark.py --N 4096 --K 4096 --repeat 3 --save
python 03_decode_attention/benchmark.py --contexts 512 2048 8192 32768 --repeat 3 --save
python 04_moe_tp2/benchmark.py --tokens 2048 --experts 64 --repeat 3 --save
torchrun --nproc_per_node=2 04_moe_tp2/run_two_gpu.py --backend nccl --modes all --repeat 3 --save
```

所有脚本必须满足：**在没有 GPU / 没有 Triton 的环境下不报错退出**，而是打印 skip 原因并继续执行 CPU 参考路径。

每个任务目录另提供：

- `run_gpu.sh`：在 GPU 机器上一键执行该任务的完整实验矩阵（含 `--repeat 3 --save`）；
- `RESULTS_TEMPLATE.md`：与 benchmark 输出列一致的性能对比表格模板，覆盖三类指标：
  - **显存占用**：静态分解（权重/激活/临时缓冲字节数）与实测单次调用峰值显存增量（`peak_extra_mem`）；
  - **吞吐量**：`tok/s`（T3 为 `B tok/step`，T4-A 为 `T tok/iter`，T4-B 为 `T tok/total_ms`）；
  - **延迟**：轮间中位数 + 首轮采样的 mean/min/max/std 分布（`latency_detail`），以及 CV；
  跑完后按 GPU 型号/实测峰值带宽逐项填入。

---

## 2. 通用评测协议

### 2.1 正确性

- 参考实现一律用 `float64`（NumPy 或 PyTorch double）计算，作为 ground truth。
- 待测实现按 fp16/bf16 存储、fp32 累加。
- 误差指标（`common.utils.error_stats`）：
  - `max_abs = max |a - ref|`
  - `max_rel = max |a - ref| / max(|ref|, 1e-6)`
  - `cosine = <a, ref> / (||a|| ||ref||)`
- 通过条件采用 **混合容差**：`max_abs ≤ atol + rtol * max|ref|`，其中每个任务在下方给出 `(atol, rtol)`。
- 边界用例必须覆盖：K/H 非 2 的幂、上下文长度非 block 整数倍、`S=0/1`、MoE 某专家 0 token、单卡/双卡结果一致性。
- 双卡任务：两个 rank 的最终结果与单卡 float64 参考比较，并额外校验 rank0/rank1 结果逐元素一致（all-reduce 后）。

### 2.2 性能

- 计时统一使用 `common.utils.bench_callable`：
  - GPU：`torch.cuda.Event`，测量前后各一次 `synchronize`；
  - CPU：`perf_counter`。
  - warmup ≥ 25 次，重复 ≥ 100 次，取 **中位数**；
  - 若单次耗时 < 1 ms，自动增大内层循环次数使每次采样 ≥ 1 ms，避免计时噪声。
- 所有输入/输出张量在计时循环外预分配；禁止在计时区间内做 dtype 转换、`.item()`、同步。
- 报告 `latency_ms`、`GB/s`、`speedup`、`peak_ratio`。

### 2.3 可复现性（repeatable improvement）

每个 benchmark 支持 `--repeat R`：完整跑 R 轮（每轮重新 warmup + 计时），输出每轮结果与：
- 中位数加速比；
- 轮间变异系数 `CV = std/mean`，要求 **CV ≤ 5%**；
- 若 CV > 5%，必须排查（时钟降频、后台占用、动态 shape 编译、L2 抖动）。

`--save` 时把结果写入 `results/<task>_<timestamp>.json`，字段包含：GPU 型号、SM 数、实测峰值带宽、各实现的中位数耗时与加速比。

### 2.4 带宽与 roofline 口径

- 峰值带宽不写死，运行时用 `common.utils.measure_peak_bandwidth()` 实测（大张量 device-to-device copy，字节数 = 2 × tensor bytes）。
- 有效带宽 `BW_eff = bytes_moved / latency`，其中 `bytes_moved` 按各任务给出的公式计算（只统计必须发生的 HBM 流量，不含被 cache 命中的重复读）。
- 理论上限 `latency_ideal = bytes_moved / BW_peak`；报告 `peak_ratio = latency_ideal / latency`。
- 算术强度 `AI = FLOPs / bytes_moved`，与本机 `ridge = peak_flops / BW_peak` 比较，明确“带宽受限 / 算力受限”。

---

## 3. 任务一：Decode 投影 GEMV 优化

### 3.1 背景与动机

Decode 每一步都要做投影 `y = W × x`：`W` 是 `[N, K]` 的权重（fp16），`x` 是当前 token 的隐状态 `[K]`，`y` 是 `[N]`。M=1 时：

- 计算量 `2NK` FLOPs，访存 `≈ 2NK` bytes（权重读取占绝对主导）；
- 算术强度 `AI ≈ 1 FLOP/Byte`，A100 的 ridge 约 156 TFLOPS(fp16) / 2.0 TB/s ≈ 78，**带宽受限 2 个数量级**；
- 但 `N` 行之间天然并行，若每个 CTA 只处理 1 行，`N=4096` 时只有 4096 个 CTA 中的一部分能同时驻留；若 `N` 较小或 CTA 太重，会出现 SM 利用率不足；
- cuBLAS 的 GEMV 路径对 M=1 有专门实现，但不同形状下未必吃满带宽，且无法与前后算子融合。

### 3.2 接口定义

```python
def gemv(W, x) -> y
# W: [N, K] float16, row-major (K 连续)
# x: [K]    float16
# y: [N]    float16（内部 fp32 累加）
```

Baseline：`torch.nn.functional.linear(x, W)`（即 cuBLAS GEMV）。

### 3.3 优化要求

必须实现并对比至少 3 个层级：

1. **v0 naive**：每个 program 处理 1 行，沿 K 以 `BLOCK_K` 循环，标量累加。
2. **v1 tiled**：每个 program 处理 `BLOCK_N` 行，复用 `x` 的 tile；提高单 CTA 工作量、减少调度开销。
3. **v2 split-K**：`N` 方向并行不足时，把 K 切成 `SPLIT_K` 份，二维 grid `(ceil(N/BLOCK_N), SPLIT_K)`，各 split 写 fp32 partial，再由 reduce kernel 求和并转 fp16（确定性，不用 atomic）。

要求覆盖的优化维度：

- **tiling**：`BLOCK_N ∈ {1, 2, 4, 8, 16}`、`BLOCK_K ∈ {128, 256, 512, 1024}`，给出选择依据；
- **memory access**：`x` 只在 program 内加载一次/每 tile 一次，避免每行重复读；权重按连续维（K）加载，保证合并访存；利用 L2 缓存 `x`（`K × 2B` 仅 4–16 KB）；
- **vectorization**：`BLOCK_K` 为 8 的倍数以保证 128-bit 向量化，必要时用 `tl.multiple_of` / `tl.max_contiguous` 提示对齐；
- **work distribution**：`SPLIT_K` 的选择（建议 `SPLIT_K = clamp(ceil(SM * target_ctas / (N/BLOCK_N)), 1, 8)`），并分析 split-K 的 extra traffic（partial 写读 `SPLIT_K × N × 4B`）与启动开销；
- **cache / eviction**：权重是流式读取，建议 `eviction_policy="evict_first"`；`x` 建议 `cache_modifier=".ca"`。

### 3.4 实验矩阵

| 维度 | 取值 |
| --- | --- |
| N | 2048, 4096, 11008 |
| K | 2048, 4096, 8192 |
| dtype | float16（主）；float32（对照） |
| BLOCK_N | 1, 2, 4, 8, 16 |
| BLOCK_K | 128, 256, 512, 1024 |
| SPLIT_K | 1, 2, 4, 8 |

至少报告 `(N=4096, K=4096)`、`(N=11008, K=4096)`、`(N=2048, K=8192)` 三组。

### 3.5 验收标准

| 项 | 标准 |
| --- | --- |
| 正确性 | `atol=1e-2, rtol=1e-2`（相对 `max|ref|`），三种 kernel 全部通过 |
| 性能 | 最佳 kernel 有效带宽 ≥ 实测峰值的 70% |
| 性能 | 相对 `F.linear` baseline：平均 ≥ 1.3×，至少一组 ≥ 1.5× |
| 可复现 | `--repeat 3` 加速比 CV ≤ 5% |
| 分析 | README 给出 roofline 数字、split-K 收益/代价曲线、最终参数选择理由 |

### 3.6 必答分析问题

1. 该 kernel 的瓶颈是带宽、延迟还是并行度？用 `AI` 与 `ridge` 定量说明。
2. `BLOCK_N` 增大后，`x` 的重复读取减少，但寄存器/占用率如何变化？给出 trade-off。
3. split-K 在什么形状下有效、什么形状下反而变慢？extra traffic 占原流量的百分比是多少？
4. 为什么不用 `tl.dot`（tensor core）？在 M=1 下 tensor core 的利用率是多少？

---

## 4. 任务二：融合 Residual Add + RMSNorm

### 4.1 背景与动机

Transformer 每个 block 的残差流更新为：

```text
h   = x + residual
out = h * rsqrt(mean(h²) + eps) * g
```

其中 `x` 是子层输出（attention / MLP），`residual` 是残差流，`g` 是 `[H]` 的 norm weight，`eps` 通常 1e-6。

未融合实现需要 2 个 kernel：
1. `h = x + residual`（读 2 个张量、写 1 个中间张量）；
2. `out = RMSNorm(h)`（读中间张量、读 weight、写输出）。

中间张量 `h` 的写+读是纯浪费。融合后每个元素只读 2 次、写 1 次（若还需要保留更新后的残差流则写 2 次）。

### 4.2 接口定义

```python
def fused_add_rmsnorm(x, residual, weight, eps=1e-6, write_residual=True) -> (out, residual_out | None)
# x, residual: [T, H] float16
# weight:      [H]    float16
# out:         [T, H] float16（归一化结果）
# residual_out:[T, H] float16（h = x + residual，可选）
```

Baseline：

```python
h = x + residual                      # kernel 1
out = F.rms_norm(h, (H,), weight, eps)  # kernel 2
```

### 4.3 访存分析（T × H 个元素，每元素 2B）

| 实现 | 读 | 写 | 总流量 | 理论加速比 |
| --- | --- | --- | --- | --- |
| baseline（2 kernel） | 2TH + TH + H | TH + TH | ≈ 5TH | 1.00× |
| fused，仅输出 norm | 2TH + H | TH | ≈ 3TH | 1.67× |
| fused，同时写回残差 | 2TH + H | 2TH | ≈ 4TH | 1.25× |

> 真实 LLM 需要保留 `h` 供下一层继续累加，因此 `write_residual=True` 是默认语义；`write_residual=False` 用于展示融合上限。

### 4.4 优化要求

1. 单 kernel：一次读入 `x`、`residual`，在寄存器内完成加法、平方和规约、归一化、缩放、写回。
2. 一个 program 处理一行（或 `BLOCK_T` 行），`BLOCK_H = next_power_of_2(H)`；当 `H` 超过寄存器预算时，采用“两遍读”或分块 + 二次规约，并在 README 说明阈值。
3. 数值精度：
   - 加法、平方、mean、rsqrt 全部 fp32；
   - `rsqrt` 用 fp32 近似指令时需评估误差，必要时用 `1/sqrt`；
   - `mean` 的除法用真实 `H`，不能用 `BLOCK_H`。
4. 访存：fp16 向量化加载（8 元素/线程），weight 只读一次，输出按 dtype 转回。
5. 可选进阶：多行 per program 复用 weight；对 `T` 很小（decode T=1）避免 CTA 数不足。

### 4.5 实验矩阵

| 维度 | 取值 |
| --- | --- |
| T | 1, 8, 64 |
| H | 2048, 4096, 8192 |
| dtype | float16（主）；float32（精度对照） |
| 模式 | write_residual=True / False |
| eps | 1e-6；额外测试 1e-5 |

### 4.6 验收标准

| 项 | 标准 |
| --- | --- |
| 正确性 | fp16：`atol=5e-3, rtol=1e-2`；fp32：`rtol=1e-5`；且与 baseline 的差异不超过 baseline 相对 fp64 误差的 1.5 倍 |
| 性能 | `write_residual=False` ≥ 1.5×；`write_residual=True` ≥ 1.25× |
| 性能 | 有效带宽 ≥ 实测峰值的 70% |
| 可复现 | 3 轮 CV ≤ 5% |
| 分析 | 说明为什么达不到理论上限（启动开销、写回残差、weight 读取、L2 写分配） |

### 4.7 必答分析问题

1. 融合后理论流量 3TH / 4TH，实测带宽是否达到 roofline？差在哪？
2. 为什么 `mean` 必须在 fp32 下做？用 fp16 累加会带来多大误差？
3. `BLOCK_H = next_pow2(H)` 在 `H=4096` 时的寄存器压力如何？`num_warps` 怎么选？
4. 如果残差流必须写回，还有什么进一步优化空间（例如与下一个 GEMM 融合 / 写入 L2 常驻）？

---

## 5. 任务三：Decode Attention 优化

### 5.1 背景与动机

Decode 时 `q` 只有 1 个 token，但需要读取全部历史 KV cache：

```text
scores = q · Kᵀ / sqrt(D)     # [HQ, S]
p      = softmax(scores)
out    = p · V                # [HQ, D]
```

- 计算量 `4 · B · HQ · S · D` FLOPs；
- 访存量 `2 · B · S · HKV · D · 2B`（K 和 V），与 `HQ` 无关（GQA 下 `HQ/HKV` 越大，KV 复用越多）；
- 在 `HQ/HKV=4, D=128` 时 `AI ≈ 2 FLOP/Byte`，**带宽受限**；
- 并行度只有 `B × HQ`（例如 B=1, HQ=32 → 32 个 CTA），长上下文时远不足以填满 GPU，因此需要 **split-K / flash-decoding**：按序列切分，各 split 产出 `(m_i, l_i, acc_i)`，再用 combine kernel 合并；
- 真实推理的 KV cache 通常是 **paged**（按 page 分配，`page_size=16/32`），需要 page table 间接寻址。

### 5.2 接口定义

```python
def decode_attention(q, k_cache, v_cache, page_table=None, page_size=16, num_splits=1) -> out
# q:        [B, HQ, D]      float16
# 连续布局: k_cache/v_cache: [B, S, HKV, D] float16
# 分页布局: k_cache/v_cache: [num_pages, page_size, HKV, D] float16
#          page_table: [B, max_pages] int32
# out:      [B, HQ, D]      float16
# GQA: HQ % HKV == 0，kv_head = q_head // (HQ // HKV)，scale = 1/sqrt(D)
```

Baseline（“当前实现”）：
- eager：`q @ k.transpose(-1,-2)` → `softmax` → `@ v`（3 个以上 kernel）；
- `F.scaled_dot_product_attention`（GPU 上走 flash/mem-efficient backend）。

### 5.3 优化要求

1. **v0 单 program per (b, h)**：`q` 常驻寄存器，沿 S 以 `BLOCK_S` 流式读取 K/V，online softmax（维护 `m, l, acc`），避免物化 `[S]` scores。
2. **v1 flash-decoding split**：grid `(B×HQ, num_splits)`，每个 split 处理 `ceil(S/num_splits)` 个 token（按 `BLOCK_S` 对齐），写 `partial_acc [BH, splits, D] fp32`、`partial_m/partial_l [BH, splits] fp32`；combine kernel 用 `exp(m_i - m_max)` 加权合并。`num_splits` 按 `SM 数 / (B×HQ)` 自适应。
3. **v2 paged KV**：通过 page table 读取，逻辑 token 位置映射到物理 page；split 边界必须与 page/block 对齐，保证不跨页错位。
4. GQA：同一 kv head 被 `HQ/HKV` 个 q head 共享；v0/v1 至少做到每个 program 只读自己 kv head 的 K/V（不重复读其他 head）。
5. 访存：K/V tile 连续加载（D 维合并），fp16 转 fp32 计算；`tl.exp` 用 fp32；对越界 token 置 `-inf`。
6. 可选进阶：同一 program 处理同组多个 q head，KV tile 只读一次（提升 GQA 复用）；或用 `tl.dot` 处理 `BLOCK_S` 维（M=1 时收益有限，需实测）。

### 5.4 实验矩阵

| 维度 | 取值 |
| --- | --- |
| 上下文 S | 512, 2048, 8192, 32768 |
| B | 1, 8 |
| HQ / HKV | 32 / 8（GQA 4:1）；额外 32 / 32（MHA 对照） |
| D | 128（主），64（对照） |
| 布局 | 连续 / paged (page_size=16) |
| num_splits | 1, 4, 8, 16, 32（按 SM 自适应） |

CPU 上只跑小尺寸（如 B=2, HQ=8, HKV=2, S≤1024），GPU 上跑完整矩阵。

### 5.5 验收标准

| 项 | 标准 |
| --- | --- |
| 正确性 | fp16：`atol=2e-2, rtol=2e-2`；paged 与连续结果一致（`max_abs ≤ 1e-3`） |
| 性能 | S ≥ 2048 时 ≥ 3× eager |
| 性能 | 与 SDPA 相比 ≥ 0.9×（长上下文目标 ≥ 1.0×） |
| 性能 | S=32768、B×HQ ≤ 64 时，split 相对非 split ≥ 1.4× |
| 性能 | 有效带宽 ≥ 实测峰值的 60%（GQA 4:1、D=128） |
| 可复现 | 3 轮 CV ≤ 5% |
| 分析 | 给出 latency–context 曲线与带宽曲线，解释 split 的收益拐点；paged 的额外开销 |

### 5.6 必答分析问题

1. decode attention 的并行度为什么只有 `B×HQ`？什么上下文长度开始需要 split？
2. GQA 如何改变带宽需求？`HQ/HKV=4` 时每个 KV 字节被复用几次？
3. paged KV 的间接寻址带来多少开销？page_size 如何影响？
4. SDPA 的 flash-decoding 已经很强，你的实现从哪里拿回性能（更少中间量、更适合的形状、更低启动开销）？

---

## 6. 任务四：MoE 专家计算 + 双 GPU 拆分投影

本任务包含两个子任务，均放在 `04_moe_tp2/`。

### 6.1 子任务 A：MoE gate/up、down 投影

#### 背景

MoE 层中每个 token 被路由到 top-k 个专家，每个专家只处理少量 token（如 T=2048、E=64、top_k=2 → 平均每专家 64 个 token）。若按专家逐个调用 cuBLAS：

- 每次 GEMM 的 M 很小（16–128），N=2I、K=H 很大，**M 方向严重欠填充**；
- kernel 启动次数 = E（×2：gate/up 与 down），启动与调度开销占比高；
- 权重按专家存放，每个专家的权重都要完整读一遍，读带宽是必须的，但 tile 形状差会导致 L2 复用差、SM 空闲。

#### 接口

```python
# 路由结果（host 侧准备）
# sorted_x:       [M, H]  float16，按 expert 连续排列的 token（M = T * top_k，可 padding）
# expert_offsets: [E + 1] int32，第 e 个专家的 token 区间 [offsets[e], offsets[e+1])
# W_gate, W_up:   [E, I, H] float16
# W_down:         [E, H, I] float16

def moe_gate_up(sorted_x, W_gate, W_up, expert_offsets) -> y  # [M, 2I] 或融合 SwiGLU -> [M, I]
def moe_down(sorted_y, W_down, expert_offsets) -> out         # [M, H]
```

Baseline：`for e in range(E): torch.mm(x[start:end], W[e].T)`。

#### 优化要求

1. **grouped GEMM**：所有专家一次 kernel 启动；host 预计算 tile map `(expert, m_tile, n_tile)`，1D grid 遍历，避免 `E × max_M` 的空转 tile。
2. **tiling**：`BLOCK_M ∈ {16, 32, 64}`（小 M 时 16）、`BLOCK_N ∈ {64, 128, 256}`、`BLOCK_K ∈ {32, 64, 128}`；说明在 `M≈64` 时为何 `BLOCK_M=16/32` 比 64 更优（padding 浪费 = `(BLOCK_M·ceil(M/BM) - M)/M`）。
3. **weight layout**：`[E, N, K]`（K 连续）；对比 `[E, K, N]` 的加载合并度，说明选择。门控/上投影权重按专家连续存放，避免跨专家 stride 破坏合并访存。
4. **融合 gate/up**：一个 program 同时算 `gate = x·Wg`、`up = x·Wup`，epilogue 直接 `silu(gate) * up`，`x` 只读一次，输出只写一份（`[M, I]` 而非 `[M, 2I]`）。同时提供非融合版本作为对照。
5. **work distribution**：可选 persistent kernel（grid = SM 数 × waves，内部 `for tile in range(pid, num_tiles, num_programs)`），并比较与静态 1D grid 的差异。
6. **down 投影**：同样 grouped，输入 `[M, I]`、权重 `[E, H, I]`；说明为何 down 的 N=H 较小、K=I 较大，tile 选择与 gate/up 不同。

#### 实验矩阵

| 维度 | 取值 |
| --- | --- |
| T | 512, 2048（GPU）；CPU 用小尺寸 |
| E | 8, 64 |
| top_k | 2 |
| H | 4096 |
| I | 14336（GPU）；CPU 用 512/1024 |
| dtype | float16 |
| BLOCK_M / N / K | 16/32/64 × 64/128/256 × 32/64/128 |
| 实现 | per-expert baseline / grouped / grouped+ fused SwiGLU / persistent |

#### 验收标准

| 项 | 标准 |
| --- | --- |
| 正确性 | `atol=2e-2, rtol=2e-2`（SwiGLU 输出同样比较） |
| 性能 | grouped 相对 per-expert baseline ≥ 1.5×（T=2048, E=64） |
| 利用率 | 报告 tile 数、padding 浪费、`num_tiles / SM` 波数；要求 ≥ 60% 的 SM 在主要 wave 中非空 |
| 分析 | 给出“每专家 token 数 < BLOCK_M”时 M 方向浪费的定量表；说明 split-K 是否值得（K 很大但 M 很小） |

#### 必答分析问题

1. 小 M GEMM 为什么利用率低？tensor core 的 tile 最小粒度是多少，padding 浪费多少？
2. tile map 调度相比 `grid=(E, max_m_tiles, n_tiles)` 省掉多少空 tile？
3. gate/up 融合省了多少 `x` 读取和输出写入？理论流量比是多少？
4. `[E, N, K]` 与 `[E, K, N]` 两种权重布局在合并访存上的差异？

### 6.2 子任务 B：双 GPU 拆分投影

#### 背景

把 down 投影 `y = W x`（`W [H, I]`、`x [T, I]`）按 K 维切分到 2 张 GPU：

```text
rank r: x_r = x[:, r*I/2 : (r+1)*I/2], W_r = W[:, r*I/2 : (r+1)*I/2]
        y_r = W_r x_r          # 局部部分和 [T, H]（fp32）
y = y_0 + y_1                  # all-reduce(SUM)
```

- 每张卡只读一半权重，理想情况本地计算时间减半；
- 代价是一次 all-reduce：ring 算法每卡收发 `2 × (world_size-1)/world_size × T·H·4B ≈ T·H·4B`（2 卡时各 1 份）；
- decode `T=1` 时通信量极小（`H=4096` → 16 KB），瓶颈在本地 kernel；`T` 增大后通信占比上升，需要 **通信与计算重叠**（按输出列/行分块，计算下一块时异步 reduce 上一块）；
- 计算 `T=1` 的本地 kernel 与任务一相同（GEMV，带宽受限）；`T` 较大时变为小 GEMM。

#### 接口与模式

```python
# run_two_gpu.py（torch.distributed，2 ranks）
# --mode matmul_allreduce : 本地 torch.mm + dist.all_reduce（baseline）
# --mode triton_allreduce : 本地 Triton GEMM + dist.all_reduce
# --mode chunked_overlap  : 输出按列分块，Triton 计算 + async all_reduce 重叠
# --mode p2p_reduce       : 两卡互发 partial，各自相加（batch_isend_irecv）
# --mode column_parallel  : 按 N 维切分，各自计算一半输出列（all-gather 对照）
```

要求：

1. 本地 kernel 支持 `M ∈ {1, 8, 64}` 的小 GEMM，fp32 累加；
2. 通信实现必须可选 NCCL（GPU）与 gloo（CPU 双进程模拟）；
3. 计时拆分为 `local_ms`、`comm_ms`、`total_ms`（预热后用 `barrier` 对齐）；
4. 与单卡全量 GEMM 对比：`speedup = T_single / T_2gpu`，并报告相对理想 2× 的并行效率；
5. 分析 NCCL all-reduce 与 P2P 的差异、chunk 大小对重叠效果的影响、PCIe 与 NVLink 的带宽差异。

#### 实验矩阵

| 维度 | 取值 |
| --- | --- |
| T | 1, 8, 64, 512 |
| H | 4096 |
| I | 14336 |
| dtype | float16（GPU/NCCL）；float32（CPU/gloo） |
| world_size | 2 |
| 模式 | 上述 5 种 |

#### 验收标准

| 项 | 标准 |
| --- | --- |
| 正确性 | 双卡结果 vs 单卡 float64：`atol=2e-2, rtol=2e-2`；两个 rank 输出一致 |
| 性能 | `T=1/8` 时端到端 ≥ 1.6× 单卡全量（本地带宽减半） |
| 性能 | `T=64/512` 时 chunked overlap 相对 non-overlap ≥ 1.2×，通信占比 ≤ 30% |
| 可复现 | 3 轮 CV ≤ 5% |
| 分析 | 通信量公式、扩展效率、模式选择的决策表（按 T 与互联类型） |

#### 必答分析问题

1. 为什么按 K 切分需要 all-reduce，而按 N 切分只需要 all-gather / 甚至不通信？两种切分各适合什么并行策略（TP 的 column/row parallel）？
2. 2 卡 ring all-reduce 的实际通信量是多少？与 `T·H` 的关系？
3. 为什么 `T=1` 时通信几乎免费，而 `T=512` 时必须重叠？给出通信/计算比的临界点估算。
4. chunked overlap 的 chunk 数如何选？过小/过大分别有什么问题？

---

## 7. 提交清单

每个任务目录必须包含：

- [ ] `README.md`：任务说明、实现思路、瓶颈分析、最优参数、结果表（含 GPU 型号与实测峰值带宽）、复现命令
- [ ] `reference.py`：baseline + float64 ground truth + 输入构造
- [ ] `kernels.py` / `kernels_moe.py` / `kernels_tp2.py`：Triton 实现（无 GPU 时可 import、不执行）
- [ ] `test_correctness.py`：CPU 也能跑的边界用例；GPU 路径自动 skip
- [ ] `benchmark.py`（T4-B 为 `run_two_gpu.py`）：`--repeat`、`--save`、结果 JSON
- [ ] 实测结果：至少一组形状的完整数据（latency / GB/s / speedup / peak_ratio / CV）

顶层的 `TASK_SPEC.md` 不随实现改动；若实现与规范有偏差，必须在对应 `README.md` 中说明原因。

---

## 附录 A：常用公式

```text
有效带宽      BW_eff   = bytes_moved / latency
理论上限延迟  lat_ideal = bytes_moved / BW_peak
带宽利用率    peak_ratio = lat_ideal / latency
算术强度      AI       = FLOPs / bytes_moved
机器脊点      ridge    = peak_flops / BW_peak
实测峰值带宽  BW_peak  ≈ 2 × tensor_bytes / copy_latency   （device-to-device copy）
加速比        speedup  = lat_baseline / lat_optimized
轮间稳定性    CV       = std(latency) / mean(latency)
```

各任务的 `bytes_moved`：

| 任务 | bytes_moved |
| --- | --- |
| T1 GEMV | `(N·K + K)·2 + N·2` |
| T2 fused norm（norm-only） | `2·T·H·2 + H·2 + T·H·2` |
| T2 fused norm（写残差） | `2·T·H·2 + H·2 + 2·T·H·2` |
| T3 attention | `B·S·HKV·D·2·2 + B·HQ·D·2·2`（GQA 下 K/V 只读一份） |
| T4 MoE gate/up（融合） | `M·H·2 + 2·E·I·H·2 + M·I·2`（所有被激活专家权重） |
| T4-B 双卡 local（每卡） | `T·(I/2)·2 + H·(I/2)·2 + T·H·4`（fp32 输出） |

## 附录 B：优化 checklist

- [ ] 是否用 `torch.cuda.Event` 计时并在计时前后 `synchronize`？
- [ ] 是否在计时循环外做 dtype 转换 / 分配 / 编译（Triton 首次 launch 会 JIT）？
- [ ] grid 是否足够大（≥ 2×SM），又不过大（调度开销）？
- [ ] 连续维是否用于最内层索引，保证 128-bit 合并访存？
- [ ] 规约是否在 fp32 下完成？
- [ ] 是否处理非 2 的幂 / 非 block 整数倍 / 空输入？
- [ ] 是否用 L2 复用（小张量常驻、权重流式 `evict_first`）？
- [ ] 是否报告了 roofline 与瓶颈结论，而不只是“变快了”？
