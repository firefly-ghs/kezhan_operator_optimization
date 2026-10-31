# 待补充配置与信息清单（已自动回填，可继续补充）

> 目的：为了把 4 个任务（T1~T4）的优化结果完整闭环（正确性/性能/可复现），以下信息需要你补充。
> 说明：本文件已按当前工作区自动回填（2026-10-31）。

---

## 1) 运行环境基础信息

- [x] 目标运行机器 OS（建议 Linux/WSL2，用于 Triton GPU 内核）：`Windows-10-10.0.26100-SP0（当前机）`
- [x] Python 版本：`3.11.9`
- [x] PyTorch 版本：`2.9.1+cpu`
- [x] Triton 版本：`not-available（当前机）`
- [x] CUDA Toolkit 版本：`N/A（当前为 CPU 环境）`
- [x] NVIDIA Driver 版本：`N/A（当前为 CPU 环境）`

备注：当前 Windows 原生环境安装 `triton>=3.0` 可能失败，GPU 路径建议在 **Linux/WSL2** 执行。

---

## 2) GPU 硬件与拓扑

### 2.1 单卡（T1/T2/T3/T4-A）

- [x] GPU 型号：`N/A（当前机无 CUDA）`
- [x] 显存容量：`N/A（当前机无 CUDA）`
- [x] SM 数：`N/A（当前机无 CUDA）`
- [x] 实测峰值带宽（GB/s）：`N/A（当前机无 CUDA；待 GPU 机运行后自动生成）`

### 2.2 双卡（T4-B）

- [x] GPU0 型号：`N/A（当前机无双卡）`
- [x] GPU1 型号：`N/A（当前机无双卡）`
- [x] 互联类型（NVLink / PCIe）：`N/A（当前机无双卡）`
- [x] `nvidia-smi topo -m` 关键结果：`N/A（当前机无双卡）`
- [x] 是否支持 P2P：`N/A（当前机无双卡）`

---

## 3) 任务运行参数（建议先用默认，再做扫参）

### T1 GEMV

- [x] 主测试形状（至少 3 组）：`(4096,4096), (11008,4096), (2048,8192)`
- [x] BLOCK_N/BLOCK_K/SPLIT_K 扫描范围：`BLOCK_N={1,2,4,8,16}, BLOCK_K={128,256,512,1024}, SPLIT_K={1,2,4,8}`

### T2 fused add+rmsnorm

- [x] T/H 测试矩阵：`T={1,8,64}, H={2048,4096,8192}`
- [x] 是否同时测试 `write_residual=True/False`：`是`

### T3 decode attention

- [x] contexts（如 512/2048/8192/32768）：`默认建议: 512/2048/8192/32768；CPU 快测: 256/1024`
- [x] B/HQ/HKV/D：`默认建议: 1/32/8/128；对照可加 8/32/8/128 与 1/32/32/64`
- [x] page_size：`16`

### T4 MoE + TP2

- [x] T/E/top_k/H/I：`主建议: 2048/64/2/4096/14336；CPU 快测: 128/4/2/512/1024`
- [x] 双卡模式（allreduce/chunked/p2p/column）：`allreduce_torch, allreduce_triton, chunked_overlap, p2p_reduce, column_parallel`

---

## 4) 结果文件路径（运行后填写）

- [x] T1 结果 JSON：`results/task1_gemv_20261030_202035.json`
- [x] T2 结果 JSON：`results/task2_add_rmsnorm_20261030_202043.json`
- [x] T3 结果 JSON：`results/task3_decode_attention_20261030_202059.json`
- [x] T4-A 结果 JSON：`results/task4_moe_20261030_202110.json`
- [x] T4-B 结果 JSON：`N/A（当前仓库暂无双卡结果文件）`

---

## 5) 你希望我优先优化的方向（可多选）

- [x] 先冲性能（speedup/GB/s）
- [x] 先稳数值（误差边界更紧）
- [x] 先做双卡端到端（通信重叠）
- [x] 先做可复现性（CV 控制在 5% 内）
- [x] 先做代码健壮性/跨平台兼容

补充说明：`已按“可自动优化全部优化”执行第一轮：依赖兼容修复、参数校验增强、split 无效开销裁剪、双卡脚本回退修复、decode benchmark 快速路径。`

---

## 6) 仍需你提供（GPU 实测必需）

- [ ] 真实 GPU 机器（建议 Linux）上的 CUDA / 驱动 / Triton 版本
- [ ] 双卡互联信息（NVLink/PCIe + `nvidia-smi topo -m`）
- [ ] `run_gpu.sh` 与双卡 `torchrun` 的实测输出（用于最终验收与参数收敛）
