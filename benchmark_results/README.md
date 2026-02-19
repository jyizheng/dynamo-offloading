# LongBench2 Benchmark Results

## Setup
- **Model**: Qwen2.5-72B-Instruct (BF16, 135.4GB weights, 80 layers, 8 KV heads)
- **GPU**: NVIDIA GB200 (189GB VRAM)
- **Framework**: NVIDIA Dynamo + TRT-LLM PyTorch backend v0.9.0
- **Benchmark**: LongBench2 (multiple-choice, long-context understanding)
- **Max tokens**: 30,000 (truncated from up to 284K)
- **Engine config**: max_batch_size=8, max_num_tokens=4096, chunked_prefill=true, free_gpu_memory_fraction=0.80
- **KV cache**: 10.07 GiB GPU, 32,992 tokens max, tokens_per_block=32
- **KV block size**: ~10 MB/block (80 layers x 2 KV x 32 tokens x 1024 dim x 2B BF16)
- **Date**: 2026-02-17

## Full Benchmark (503 samples, KVBM)

| Metric | Value |
|---|---|
| **Accuracy** | **198/503 (39.4%)** |
| Avg latency/question | 5.8s |
| Total time | 2938s (49m) |
| 5k-15k token bucket | 15 qs, avg 3.4s, avg_tokens=12,225 |
| 15k-30k token bucket | 88 qs, avg 5.4s, avg_tokens=21,656 |

## Key Finding: Disk Offload Filter

KVBM 默认开启 `FrequencyFilter`，要求一个 sequence 被访问 >=2 次才会从 CPU offload 到 Disk。
这导致大部分 KV block 在 CPU 层被 LRU 直接丢弃，disk cache 几乎没有被利用。

设置 `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true` 后，所有 block 都会被 offload 到 disk，
第二次访问时可以从 disk 直接 onboard 回 GPU，跳过 prefill 计算，**延迟降低 40%**。

### A/B: Filter OFF vs Filter ON (50 samples)

| Metric | Baseline (无KVBM) | Filter ON Cold | Filter ON Warm | **Filter OFF Cold** | **Filter OFF Warm** |
|---|---|---|---|---|---|
| Accuracy | 18/50 (36%) | 19/50 (38%) | 19/50 (38%) | 20/50 (40%) | 20/50 (40%) |
| **Avg latency** | 8.8s | 10.8s | 9.2s | 6.5s | **5.3s** |
| Total time | 441s | 540s | 459s | 324s | **263s** |
| vs Baseline | — | +23% | +5% | **-26%** | **-40%** |

### Cache Metrics: Filter OFF (after 2 runs)

| Metric | Filter ON | Filter OFF |
|---|---|---|
| GPU→CPU offload (d2h) | 85,110 blocks | 59,351 blocks |
| **CPU→Disk offload (h2d)** | **936 blocks** | **43,928 blocks (47x)** |
| **Disk→GPU onboard (d2d)** | **468 blocks** | **29,394 blocks (63x)** |
| Matched tokens | 29,952 | 940,608 |
| **Disk cache hit rate (暖跑)** | **2.1%** | **67%** |

---

## KVBM Offload 架构详解

### Offload Chain (数据流)

```
┌──────────┐    offload     ┌──────────┐    offload     ┌──────────┐
│ G1: GPU  │ ──────────────>│ G2: CPU  │ ──────────────>│ G3: Disk │
│ (Device) │    async D2H   │ (Host)   │    async H2D   │ (Local)  │
│ 10 GiB   │                │ 40 GB    │                │ 500 GB   │
└──────────┘                └──────────┘                └──────────┘
      ^                                                       │
      │                      onboard                          │
      └───────────────────── Disk→GPU (D2D) ──────────────────┘
```

### Block 生命周期

1. **推理产生 KV block** → 存入 GPU KV cache
2. **GPU cache 满** → 触发 offload，block 异步拷贝到 CPU (pinned memory)
3. **Block 注册到 CPU pool** → 立即 enqueue H2D offload 请求
4. **H2D worker 处理** → 经过 FrequencyFilter 检查：
   - Filter ON (默认): 访问次数 < 2 → **跳过，不写 disk**
   - Filter OFF: 直接写入 disk
5. **CPU pool 满** → LRU 淘汰旧 block → **直接释放，不触发 offload**
6. **下次相同 prompt** → token hash 匹配 → 从 disk onboard 回 GPU → 跳过 prefill

### 关键问题：淘汰 ≠ Offload

CPU pool 的 LRU 淘汰和 H2D offload 是**完全解耦的**两条路径：
- Offload: block 注册时触发，异步写 disk（受 filter 门控）
- 淘汰: pool 满时 LRU 释放，**不经过 offload 路径**

这意味着：
- CPU 40GB ≈ 4,000 blocks，50 个 sample 需要 ~44,000 blocks
- 如果 Filter ON: 大部分 block 在 CPU 被 LRU 淘汰时直接丢弃
- 如果 Filter OFF: block 注册时就开始异步写 disk，淘汰前已持久化

### Offload 性能特点

- **完全异步**: 推理线程只做 `enqueue_offload()` (unbounded channel)，不阻塞
- **并发限制**: `MAX_CONCURRENT_TRANSFERS = 4`
- **Onboard 路径**: Disk→GPU 直接传输 (D2D)，不经过 CPU

### 为什么 Disk Cache 命中率只有 67% 而不是 100%？

第二次（暖）运行的 disk cache 命中率 = 29,394 onboard blocks / ~44,000 total blocks ≈ **67%**。
（Disk 500GB 足够容纳全部 block，命中率未达 100% 是因为部分 block 未能及时写入 disk。）

#### 为什么不是 100%？

根本原因：**CPU→Disk offload 速度跟不上 block 产生速度**。

| 阶段 | Blocks | 说明 |
|---|---|---|
| GPU→CPU offload | 59,351 | 全部成功 |
| CPU→Disk offload | 43,928 | 只有 74% 写入 disk |
| **丢失（未写入 disk）** | **15,423 (26%)** | **LRU 淘汰先于 offload 完成** |

```
推理产生 block ──→ GPU offload ──→ CPU pool (4,000 blocks 容量)
                                      │
                         ┌─────────────┼─────────────┐
                         ↓             ↓             ↓
                    异步写 disk    LRU 淘汰        新 block 挤入
                   (4 并发 × 10MB)  (直接丢弃!)
```

- CPU pool 只能容纳 ~4,000 blocks（40GB / 10MB per block）
- 50 个 sample 总共产生 ~44,000 blocks → CPU pool 要翻转 **11 次**
- 异步 offload 只有 4 并发（`MAX_CONCURRENT_TRANSFERS = 4`）
- 每个 block ~10MB，写入速度受限于磁盘 I/O 带宽
- 结果：15,423 个 block 在异步写入完成**之前**就被 LRU 淘汰丢弃了

#### 可能的改进方向

1. **增大 CPU cache** → block 在 CPU 停留更久，给 offload 更多时间完成写入
2. **Bypass CPU 模式** (`CPU_CACHE_GB=0`) → GPU 直接写 Disk，省掉 CPU 中间层和 LRU 竞争
3. **增加 offload 并发数** → 修改源码中 `MAX_CONCURRENT_TRANSFERS`（当前硬编码为 4）
4. **使用更快的存储** → NVMe SSD 或 3FS 替代本地磁盘，提高写入带宽

---

## KVBM 环境变量参考

### 缓存容量

| 环境变量 | 说明 | 本次设置 |
|---|---|---|
| `DYN_KVBM_CPU_CACHE_GB` | CPU pinned memory 缓存大小 | 40 |
| `DYN_KVBM_DISK_CACHE_GB` | 本地磁盘缓存大小 | 500 |
| `DYN_KVBM_DISK_CACHE_DIR` | 磁盘缓存目录 | /tmp/kvbm_disk_cache |
| `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` | 按 block 数设置 CPU 缓存 (覆盖 GB) | — |
| `DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS` | 按 block 数设置 Disk 缓存 (覆盖 GB) | — |

### Offload 控制

| 环境变量 | 说明 | 本次设置 |
|---|---|---|
| `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` | **禁用 FrequencyFilter (关键!)** | **true** |
| `DYN_KVBM_DISK_DISABLE_O_DIRECT` | 禁用 O_DIRECT (网络文件系统需要) | true |
| `DYN_KVBM_DISK_ZEROFILL_FALLBACK` | 禁用 fallocate，用 zero-fill | true |
| `DYN_KVBM_METRICS` | 启用 KVBM metrics (端口 6880) | true |

### Bypass CPU 模式

当 `DYN_KVBM_CPU_CACHE_GB=0` 且 `DYN_KVBM_DISK_CACHE_GB>0` 时，
自动启用 GPU→Disk 直接 offload，跳过 CPU 层。

### FrequencyFilter 硬编码参数 (不可配置)

| 参数 | 值 | 说明 |
|---|---|---|
| `min_offload_frequency` | 2 | sequence 访问 >=2 次才 offload 到 disk |
| `flush_interval` | 600s | 每 10 分钟衰减频率计数 |
| `max_num_entries` | 1,000,000 | 最多跟踪 100 万个 sequence |

代码位置: `lib/bindings/kvbm/src/block_manager.rs:67-73`
带有 TODO: "These values seem plausible for most use cases, but we need to figure out a better way to configure them."

---

## Files
- `kvbm_full_503.jsonl` — Full 503-question KVBM results (filter ON, disk 20GB)
- `baseline_50.jsonl` — Baseline results (no KVBM), 50 questions
- `kvbm_50.jsonl` — KVBM cold cache results (filter ON), 50 questions
- `kvbm_warm_50.jsonl` — KVBM warm cache results (filter ON), 50 questions
- `kvbm_nofilter_warm_50.jsonl` — KVBM warm cache results (**filter OFF, disk 500GB**), 50 questions
- `run_longbench.py` — Benchmark script with timing, resume, token stats
- Each JSONL line has fields: `_id`, `question`, `answer`, `pred`, `judge`, `response`, `_tokens`, `_orig_tokens`, `_time_s`
