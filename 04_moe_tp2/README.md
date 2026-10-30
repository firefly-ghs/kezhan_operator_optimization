# 任务四：MoE 专家计算 + 双 GPU 拆分投影

本目录包含两个子任务：

- **T4-A**：MoE 中 gate/up、down 投影的 grouped GEMM 优化（单卡）；
- **T4-B**：把一个投影按 K 维切到 2 张 GPU，本地 kernel + 通信优化。

## 1. 文件

| 文件 | 内容 |
| --- | --- |
| `reference.py` | 路由（top-k / 排序 / offsets）、per-expert baseline、float64 参考、`moe_bytes` |
| `kernels_moe.py` | grouped GEMM、gate/up 融合 SwiGLU、down、`build_tile_map` |
| `kernels_tp2.py` | 双卡本地小 GEMM（支持切片 view 的显式 stride） |
| `test_correctness.py` | MoE 各实现 vs float64 / baseline；含 0-token 专家 |
| `benchmark.py` | T4-A：baseline vs grouped，tile 数、padding 浪费、波数 |
| `run_two_gpu.py` | T4-B：双进程 benchmark + `--simulate` 单进程逻辑校验 |

---

## 2. T4-A：MoE gate/up、down 投影

### 2.1 任务定义

```python
# 路由结果由 host 准备
# sorted_x:       [M, H]，按 expert 连续排列（M = T * top_k）
# expert_offsets: [E + 1] int32，专家 e 的区间 [offsets[e], offsets[e+1])
# W_gate, W_up:   [E, I, H]，W_down: [E, H, I]

moe_gate_up(sorted_x, W_gate, W_up, offsets) -> [M, 2I] | 融合 SwiGLU -> [M, I]
moe_down(sorted_y, W_down, offsets)          -> [M, H]
```

Baseline：`for e in range(E): torch.mm(x[start:end], W[e].T)`。

### 2.2 瓶颈分析

以 `T=2048, E=64, top_k=2` 为例：

- 平均每专家 `M = 64` 个 token，而 `N = 2I = 28672`、`K = 4096`；
- 每个专家的权重都必须完整读一遍（`2·I·H` 个元素），**权重读取是必须发生的流量**；
- 但 `M=64` 远小于 GEMM tile 的 M 方向粒度：`BLOCK_M=32` 时每专家需要 2–3 个 M tile，实测总 padding 浪费约 25%；
- per-expert 循环意味着 `E`（甚至 `2E`）次 kernel 启动，且每次 GEMM 的 M 都很小，SM 填充不足、tensor core 空闲；
- 结论：瓶颈是 **小 M 导致的 tile 利用率低 + 启动/调度次数多**，而不是算力或总带宽。

流量公式见 `moe_bytes`：gate/up 为 `M·H + E·2I·H + M·I`（SwiGLU 只写一份 `[M, I]`）。

### 2.3 实现方案

**grouped GEMM（`_grouped_gemm_kernel`）**

- host 侧 `build_tile_map` 生成 `(expert, m_tile, n_tile)` 列表，**跳过 0-token 专家**，也不为 `E × max_m_tiles` 生成空 tile；
- kernel 只用一个 1D grid，一次 launch 覆盖全部专家；每个 program 从 tile map 读自己的三元组；
- 用 `tl.dot(a, b, acc)` 走 tensor core，累加器 fp32；
- 专家偏移与权重指针用 `int64` 计算（`e64 = e.to(tl.int64)`），避免大权重（>2 GB）时 32 位偏移溢出；
- 权重布局 `[E, N, K]`，K 连续，保证合并访存；跨专家 stride 固定为 `N·K`。

**融合 gate/up（`_grouped_gemm_silu_kernel`）**

- 一个 program 同时算 `gate = x·Wg`、`up = x·Wup`：`x` tile 只读一次，两次 `tl.dot` 复用同一份 A；
- epilogue 直接 `silu(gate) * up = gate * sigmoid(gate) * up`，输出 `[M, I]`，不写 `[M, 2I]` 中间结果；
- 对照实现 `moe_gate_up_plain` 保留两次 launch + `[M, 2I]` 输出，用于量化融合收益。

**down 投影（`moe_down`）**

- 同样的 grouped 框架，输入 `[M, I]`、权重 `[E, H, I]`；
- down 的 `N = H = 4096` 较小、`K = I = 14336` 较大，因此 `BLOCK_N` 仍取 128，但每 tile 的 K 循环更长，tile 数约为 gate/up 的一半。

**参数**

| 参数 | 默认值 | 依据 |
| --- | --- | --- |
| `BLOCK_M` | 32 | 每专家约 64 token 时正好 2 个 tile；`BLOCK_M=16` 可把 padding 降到 ~12%，但 tile 数翻倍、启动开销上升；`tl.dot` 要求 ≥ 16 |
| `BLOCK_N` | 128 | 兼顾 tensor core tile 与寄存器；`256` 在 M 很小时会拉低占用率 |
| `BLOCK_K` | 64 | fp16 下 64 深度的 K tile 足够喂满 MMA，且不需要 split-K |
| `num_warps` | 4 | 单 tile `32×128` 的常规配置 |

### 2.4 正确性

- `test_correctness.py` 在 CPU 上覆盖 `(E=4,T=32,H=128,I=256)` 与 `(E=5,T=16,H=96,I=192,drop=1)`，后者包含 **0-token 专家**（`counts=[7,0,7,8,10]`）；
- 校验 baseline vs float64、Triton vs float64、Triton vs baseline（容差 `5e-3/1e-2`，GPU fp16 用 `2e-2/2e-2`）；
- 本机 CPU：全部 `PASS`，`max_abs ≤ 2.8e-9`（baseline）与 `≤ 2e-7`（float64 对比）。

### 2.5 本机实测（T4-A，CPU-only，fp32）

运行：

```bash
python 04_moe_tp2/benchmark.py --tokens 128 --experts 4 --hidden 512 --intermediate 1024 --repeat 3 --save
```

配置：`T=128, E=4, top_k=2, H=512, I=1024, M=256`；每专家 token 数 `min=59, max=68, avg=64.0`；`BLOCK_M=32` 时 padding 浪费 **25.0%**；tile 数 `gate_up_silu=80`、`gate_up_plain=160`、`down=40`。

| group | impl | MB/iter | ms | GB/s | peak% | speedup | CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up | baseline_silu | 18.4 | 3.4593 | 5.3 | 36.2% | 1.00x | 6.12% |
| gate_up | baseline_plain | 18.4 | 3.4741 | 5.3 | 36.1% | 1.00x | 2.43% |
| down | baseline_down | 10.0 | 1.5349 | 6.5 | 44.3% | 1.00x | 3.69% |

说明：本机无 GPU/Triton，grouped 实现不参与计时；该表用于固定 baseline、验证 padding/tile 统计口径。CPU 上 `peak%` 偏低是正常的（小张量 + 无向量化 GEMM），不作为验收依据。

### 2.6 GPU 结果（待填）

一键执行：`bash 04_moe_tp2/run_gpu.sh`（T4-A 主/小配置 + BLOCK_M/BLOCK_N 扫描，以及 T4-B 全部模式）。
完整表格模板见 [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md)。

| group | impl | ms | GB/s | peak% | speedup vs baseline | CV |
| --- | --- | --- | --- | --- | --- | --- |
| gate_up | grouped_silu | 待填 | 待填 | 待填 | 待填 | 待填 |
| gate_up | grouped_plain | 待填 | 待填 | 待填 | 待填 | 待填 |
| down | grouped_down | 待填 | 待填 | 待填 | 待填 | 待填 |

还需记录 `SM` 数、`waves`（benchmark 已打印 `tiles` 与 `waves`）、以及 `T=2048/E=64/H=4096/I=14336` 下的 padding 浪费表。

---

## 3. T4-B：双 GPU 拆分投影

### 3.1 切分方式

K 维切分（row parallel）：

```text
rank r: x_r = x[:, r·I/2 : (r+1)·I/2],  W_r = W[:, r·I/2 : (r+1)·I/2]
        y_r = W_r x_r                    # 局部部分和 [T, H]，fp32
y = y_0 + y_1                            # all_reduce(SUM)
```

对照：N 维切分（column parallel），每卡算 `[T, H/2]`，最后 `all_gather` 拼成 `[T, H]`。

- K 切分：每卡只读一半权重，本地计算量减半，但必须 all-reduce；
- N 切分：无需求和，只需 all-gather，通信量相同但语义不同（TP 的 column parallel）；
- `T=1` 时通信量仅 `H·4B = 16 KB`，几乎免费；`T` 增大后 all-reduce 占比上升，需要重叠。

### 3.2 模式（`run_two_gpu.py --modes`）

| mode | 本地计算 | 通信 | 用途 |
| --- | --- | --- | --- |
| `allreduce_torch` | `torch.mm` | `dist.all_reduce` | 通信 baseline |
| `allreduce_triton` | Triton 小 GEMM | `dist.all_reduce` | 本地 kernel 优化收益 |
| `chunked_overlap` | Triton 小 GEMM | 按行 chunk，`async_op=True` all_reduce，与下一 chunk 计算重叠 | 计算/通信重叠 |
| `p2p_reduce` | Triton 小 GEMM | `batch_isend_irecv` 互发 partial 后相加 | 避开 NCCL 的 ring 开销；仅 world_size=2 |
| `column_parallel` | `torch.mm`（半份 N） | `all_gather_into_tensor` | N 切分对照 |

计时拆分为 `local_ms`、`comm_ms`、`total_ms`，并测单卡全量 `single_gpu_ms`：

```text
speedup      = single_gpu_ms / total_ms
efficiency   = speedup / world_size
comm_ratio   = comm_ms / total_ms
```

### 3.3 实现要点

- 本地 GEMM（`kernels_tp2.local_projection`）支持**非连续切片 view**：host 传 `data_ptr` 偏移与显式 stride，kernel 不假设 contiguous；
- `tl.dot` 要求 `BLOCK_M ≥ 16`，`T=1/8` 时 M 方向会 padding，需在 `T=1` 场景对比 torch GEMV；
- `chunked_overlap` 按输出行切 `chunks` 份，逐块 `all_reduce(async_op=True)`，最后统一 `wait`；
- 正确性：两个 rank 的 partial 在 fp64 下求和作为参考，`atol=rtol=1e-4`（fp32）/ `2e-2`（fp16），并额外校验 all-reduce 后两 rank 结果一致。

### 3.4 本机验证（CPU 单进程模拟）

Windows 的 gloo 后端不可用（`makeDeviceForHostname(): unsupported gloo device`），因此提供 `--simulate`：单进程虚拟 2 个 rank，只验证 K 切分 + 求和与 N 切分 + 拼接的**数学等价性**，不做真实通信。

```bash
python 04_moe_tp2/run_two_gpu.py --simulate
```

```text
[k_split_allreduce]      PASS max_abs=1.431e-06 bound=6.235e-04
[column_split_allgather] PASS max_abs=1.863e-06 bound=6.235e-04
SIMULATION PASS
```

模拟中的 `sim_ms` 与 `single_ms` 不可用于性能结论：两个虚拟 rank 共享同一颗 CPU，且默认形状为 `T=8,H=4096,I=14336`，只能说明逻辑正确。

### 3.5 GPU 结果（待填）

模板见 [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md) 第 5/6 节；`run_gpu.sh` 会用 `torchrun --nproc_per_node=2` 依次跑 T=1/8/64/512 与 chunk 扫描。

在 2 卡 Linux 机器上：

```bash
torchrun --nproc_per_node=2 04_moe_tp2/run_two_gpu.py --backend nccl --modes all --repeat 3 --save
```

| T | mode | local_ms | comm_ms | total_ms | comm% | speedup vs 单卡 | eff | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8 | allreduce_triton | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 64 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | chunked_overlap | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | p2p_reduce | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | column_parallel | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

验收：`T=1/8` 端到端 ≥ 1.6× 单卡；`T=64/512` chunked overlap 相对 non-overlap ≥ 1.2×，通信占比 ≤ 30%；3 轮 CV ≤ 5%；同时记录 NVLink/PCIe 类型。

---

## 4. 复现命令

```bash
python 04_moe_tp2/test_correctness.py
python 04_moe_tp2/benchmark.py --tokens 128 --experts 4 --hidden 512 --intermediate 1024 --repeat 3 --save
python 04_moe_tp2/run_two_gpu.py --simulate

# GPU 单卡
python 04_moe_tp2/benchmark.py --tokens 2048 --experts 64 --hidden 4096 --intermediate 14336 --repeat 3 --save
python 04_moe_tp2/benchmark.py --tokens 2048 --experts 8 --block-m 16 --block-n 128 --block-k 64 --repeat 3

# GPU 双卡
torchrun --nproc_per_node=2 04_moe_tp2/run_two_gpu.py --backend nccl --modes all --tokens 8 --repeat 3 --save
torchrun --nproc_per_node=2 04_moe_tp2/run_two_gpu.py --backend nccl --modes chunked_overlap --tokens 512 --chunks 4 --repeat 3
```

## 5. 与规范的偏差

| 项 | 规范要求 | 实际实现 | 原因 |
| --- | --- | --- | --- |
| `[E, K, N]` 权重布局对照 | 要求对比 | 未实现 | 当前只支持 `[E, N, K]`（K 连续，合并访存最好）。`[E, K, N]` 需要转置拷贝，会引入额外一次性成本，属于 GPU 阶段实验项 |
| persistent kernel | 可选 | 未实现 | 使用静态 1D grid + tile map；tile 数足够（80–160/专家组）时静态调度已能填满，persistent 主要收益在 tile 数远大于 SM 时 |
| MoE split-K | 要求分析 | 未实现 | 每专家 M 很小、K 很大，split-K 会增加 partial 写读与 reduce 启动；在 `M≈64` 时收益需 GPU 实测，当前优先保证 grouped + 融合 |
| gloo 双进程 | 要求可切换 NCCL/gloo | Windows 上仅 `--simulate` | Windows 的 `gloo` 不支持 `makeDeviceForHostname`；Linux 上 `--backend gloo` 可直接跑真实双进程 |
| `p2p_reduce` | 未限制 | 仅 world_size=2 | P2P 互发语义只对 2 卡成立，超过 2 卡需改用 ring/tree |
