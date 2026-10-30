# 任务三 GPU 实测结果（表格模板）

> 用法：GPU 机器上执行 `bash run_gpu.sh`，把终端中位数填入本表；完整数据在 `results/task3_decode_attention_*.json`。
> `triton_splitN` 的 N 由 benchmark 打印的 `num_splits=` 决定（自适应为 `min(32, next_pow2(2·SM/(B·HQ)))`），请把实际值写进列名。未测项保留 `待填`。
> 吞吐口径：decode 每一步生成 `B` 个 token，`tok/s = B / latency`。

## 三指标速查（B=1, HQ=32, HKV=8, D=128 · GQA 4:1）

| impl | 指标 | S=512 | S=2048 | S=8192 | S=32768 |
| --- | --- | --- | --- | --- | --- |
| eager | 显存占用 peak MB | 待填 | 待填 | 待填 | 待填 |
| eager | 吞吐 tok/s | 待填 | 待填 | 待填 | 待填 |
| eager | 延迟 median ms | 待填 | 待填 | 待填 | 待填 |
| sdpa | 显存占用 peak MB | 待填 | 待填 | 待填 | 待填 |
| sdpa | 吞吐 tok/s | 待填 | 待填 | 待填 | 待填 |
| sdpa | 延迟 median ms | 待填 | 待填 | 待填 | 待填 |
| triton_split1 | 显存占用 peak MB | 待填 | 待填 | 待填 | 待填 |
| triton_split1 | 吞吐 tok/s | 待填 | 待填 | 待填 | 待填 |
| triton_split1 | 延迟 median ms | 待填 | 待填 | 待填 | 待填 |
| triton_splitN | 显存占用 peak MB | 待填 | 待填 | 待填 | 待填 |
| triton_splitN | 吞吐 tok/s | 待填 | 待填 | 待填 | 待填 |
| triton_splitN | 延迟 median ms | 待填 | 待填 | 待填 | 待填 |
| paged_splitN | 显存占用 peak MB | 待填 | 待填 | 待填 | 待填 |
| paged_splitN | 吞吐 tok/s | 待填 | 待填 | 待填 | 待填 |
| paged_splitN | 延迟 median ms | 待填 | 待填 | 待填 | 待填 |

> 数据来源：`peak_extra_mem`（实测峰值显存增量）│ `tok/s = B / median_ms` │ 3 轮中位数（延迟分布 mean/min/max 见 §2）。

## 0. 环境信息

| 项 | 值 |
| --- | --- |
| GPU 型号 / 显存 | 待填 |
| SM 数 | 待填 |
| 实测峰值带宽 BW_peak (GB/s) | 待填 |
| CUDA / Driver | 待填 |
| PyTorch / Triton | 待填 |
| 实际 num_splits（B=1, HQ=32） | 待填 |
| page_size | 16 |

## 1. 显存占用

### 1.1 静态分解（B=1, HQ=32, HKV=8, D=128, fp16）

| S | K cache | V cache | q/out | KV 合计 | partial（split=N） |
| --- | --- | --- | --- | --- | --- |
| 512 | 1 MB | 1 MB | 8 KB | 2 MB | `BH·N·(BLOCK_D+2)·4B`，待填 |
| 2048 | 4 MB | 4 MB | 8 KB | 8 MB | 待填 |
| 8192 | 16 MB | 16 MB | 8 KB | 32 MB | 待填 |
| 32768 | 64 MB | 64 MB | 8 KB | 128 MB | 待填 |

> paged 布局额外分配一整块 `[num_pages, page_size, HKV, D]`（末页可能有 padding：`((cdiv(S,16)·16 − S)·HKV·D·2B)`）。

### 1.2 实测峰值显存增量（单次调用，MB，终端 `peak_extra_mem` 行；S=32768）

| impl | S=2048 | S=8192 | S=32768 |
| --- | --- | --- | --- |
| eager | 待填 | 待填 | 待填 |
| sdpa | 待填 | 待填 | 待填 |
| triton_split1 | 待填 | 待填 | 待填 |
| triton_splitN | 待填 | 待填 | 待填 |
| paged_splitN | 待填 | 待填 | 待填 |

## 2. 延迟分布（首轮 100 次采样，ms；S=8192）

| impl | mean | min | max | std |
| --- | --- | --- | --- | --- |
| eager | 待填 | 待填 | 待填 | 待填 |
| sdpa | 待填 | 待填 | 待填 | 待填 |
| triton_split1 | 待填 | 待填 | 待填 | 待填 |
| triton_splitN | 待填 | 待填 | 待填 | 待填 |
| paged_splitN | 待填 | 待填 | 待填 | 待填 |

## 3. 主配置 GQA 4:1（B=1, HQ=32, HKV=8, D=128）

| S | impl | ms | GB/s | peak% | tok/s | memMB | vs_eager | vs_sdpa | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | eager | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 | 待填 |
| 512 | sdpa | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 512 | triton_split1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | triton_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 512 | paged_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 2048 | eager | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 | 待填 |
| 2048 | sdpa | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 2048 | triton_split1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 2048 | triton_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 2048 | paged_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | eager | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 | 待填 |
| 8192 | sdpa | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 8192 | triton_split1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | triton_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | paged_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32768 | eager | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 | 待填 |
| 32768 | sdpa | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 32768 | triton_split1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32768 | triton_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32768 | paged_splitN | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 4. 吞吐量汇总（tok/s，B=1）

| impl | S=512 | S=2048 | S=8192 | S=32768 |
| --- | --- | --- | --- | --- |
| eager | 待填 | 待填 | 待填 | 待填 |
| sdpa | 待填 | 待填 | 待填 | 待填 |
| triton_split1 | 待填 | 待填 | 待填 | 待填 |
| triton_splitN | 待填 | 待填 | 待填 | 待填 |
| paged_splitN | 待填 | 待填 | 待填 | 待填 |

## 5. 多 batch（B=8, 32/8, D=128）

| S | eager ms | sdpa ms | triton_split1 ms | triton_splitN ms | paged_splitN ms | 最佳 tok/s | 最佳 vs_eager | 最佳 CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 6. MHA 对照（B=1, HQ=HKV=32, D=128）

| S | eager ms | sdpa ms | triton_split1 ms | triton_splitN ms | 最佳 tok/s | 最佳 vs_eager | 最佳 CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 7. D=64 对照（B=1, 32/8）

| S | eager ms | sdpa ms | triton_split1 ms | triton_splitN ms | 最佳 tok/s | 最佳 vs_eager | 最佳 CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 8. num_splits 扫描（B=1, 32/8, D=128）

| S | splits | ms | GB/s | peak% | tok/s | partial 显存 MB | vs split=1 | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 1 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 8192 | 4 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | 8 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | 16 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 8192 | 32 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32768 | 1 | 待填 | 待填 | 待填 | 待填 | 待填 | 1.00x | 待填 |
| 32768 | 4 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32768 | 8 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32768 | 16 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 32768 | 32 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

> partial 显存 = `BH·splits·(BLOCK_D+2)·4B`，`BLOCK_D = next_pow2(D)`。

## 9. paged KV 额外开销

| S | triton_splitN ms | paged_splitN ms | 开销 % | paged 末页 padding bytes |
| --- | --- | --- | --- | --- |
| 2048 | 待填 | 待填 | 待填 | 0（2048 % 16 == 0） |
| 8192 | 待填 | 待填 | 待填 | 0 |
| 32768 | 待填 | 待填 | 待填 | 0 |

## 10. 验收检查

| 项 | 目标 | 实测 | 是否达标 |
| --- | --- | --- | --- |
| S ≥ 2048 相对 eager | ≥ 3× | 待填 | 待填 |
| 相对 SDPA | ≥ 0.9×（长上下文 ≥ 1.0×） | 待填 | 待填 |
| S=32768 split vs 非 split | ≥ 1.4× | 待填 | 待填 |
| 有效带宽（GQA 4:1, D=128） | ≥ 60% BW_peak | 待填 | 待填 |
| paged 额外开销 | ≤ 10% | 待填 | 待填 |
| 3 轮 CV | ≤ 5% | 待填 | 待填 |

## 11. 结论与必答问题

### Q1 为什么并行度只有 B×HQ？从多长上下文开始需要 split？

答：待填

### Q2 GQA 如何改变带宽需求？HQ/HKV=4 时每个 KV 字节被复用几次？

答：待填

### Q3 paged 间接寻址带来多少开销？page_size 如何影响？

答：待填

### Q4 相对 SDPA，你的实现从哪里拿回性能？

答：待填