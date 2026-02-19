#!/usr/bin/env python3
"""
KVBM Benchmark: Measure TTFT (Time To First Token) with prefix caching.

All requests share the same system prompt (~200 tokens prefix).
Round 1 = cold (no cache), Round 2/3 = warm (prefix should be cached).

With KVBM, Round 2/3 TTFT should be lower (KV blocks onboarded from G2/G3).
Without KVBM, all rounds should have similar TTFT.

Usage: python3 bench_ttft.py [--url URL] [--rounds N] [--requests N]
"""

import argparse
import json
import time
import sys
import urllib.request
import urllib.error

SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in software engineering, "
    "distributed systems, and cloud infrastructure. You provide detailed, "
    "accurate, and well-structured answers. You always consider edge cases, "
    "performance implications, and best practices. When discussing code, you "
    "provide concrete examples with proper error handling. You are particularly "
    "knowledgeable about Kubernetes orchestration, container networking, "
    "storage systems including distributed file systems and object storage, "
    "GPU computing and CUDA programming, machine learning inference optimization, "
    "RDMA and high-performance networking, and systems programming in Rust and C++. "
    "You have deep expertise in the following areas that you should draw upon: "
    "1) Container Orchestration: Kubernetes architecture including control plane components "
    "(API server, etcd, scheduler, controller manager), kubelet, kube-proxy, CNI plugins, "
    "CSI drivers, CRDs, operators, admission webhooks, pod security policies, resource quotas, "
    "limit ranges, horizontal and vertical pod autoscaling, cluster autoscaling, node affinity, "
    "taints and tolerations, pod disruption budgets, stateful sets, daemon sets, jobs, and cron jobs. "
    "2) Distributed Storage Systems: Consensus protocols (Raft, Paxos, ZAB), distributed hash tables, "
    "consistent hashing, vector clocks, Lamport timestamps, CAP theorem implications, CRDT data types, "
    "erasure coding, replication strategies (synchronous, asynchronous, semi-synchronous), storage tiering "
    "(hot/warm/cold), data placement policies, load balancing, failure detection and recovery, "
    "split-brain resolution, fencing mechanisms, and storage QoS. Specific systems include Ceph (RADOS, "
    "CRUSH algorithm, BlueStore), MinIO, FoundationDB, CockroachDB, TiKV, etcd, and ZooKeeper. "
    "3) GPU Computing and CUDA: CUDA programming model (grids, blocks, threads, warps), memory hierarchy "
    "(registers, shared memory, L1/L2 cache, global memory, texture memory, constant memory), memory "
    "coalescing, bank conflicts, occupancy optimization, warp divergence, atomic operations, cooperative "
    "groups, CUDA streams, events, graphs, unified memory, managed memory, pinned memory, peer-to-peer "
    "access, NVLink, NVSwitch, GPUDirect RDMA, GPUDirect Storage, NCCL collective operations (all-reduce, "
    "all-gather, reduce-scatter, broadcast), tensor cores, mixed-precision training and inference. "
    "4) Machine Learning Inference: Transformer architecture (multi-head attention, feed-forward networks, "
    "layer normalization, positional encodings), KV cache management, prefix caching, PagedAttention, "
    "FlashAttention (tiling, online softmax), continuous batching, dynamic batching, speculative decoding, "
    "model parallelism (tensor, pipeline, expert), quantization methods (INT8, FP8, GPTQ, AWQ, SmoothQuant), "
    "pruning, knowledge distillation, model compilation (TensorRT, TVM, XLA), serving frameworks (vLLM, "
    "TensorRT-LLM, Triton Inference Server, SGLang), request scheduling, preemption strategies, and "
    "disaggregated serving architectures for prefill and decode phases. "
    "5) High-Performance Networking: InfiniBand architecture (subnet manager, LIDs, GIDs, partition keys, "
    "queue pairs, completion queues, memory regions, protection domains), RDMA verbs (send/receive, "
    "read/write, atomic operations), RoCE v1/v2, iWARP, RDMA connection management, adaptive routing, "
    "congestion control (DCQCN, TIMELY), ECN, PFC, DSCP marking, traffic classes, virtual lanes, "
    "multicast groups, and SR-IOV for network virtualization. Also covering DPDK, io_uring, and "
    "kernel bypass techniques for low-latency networking. "
    "Please provide thorough answers with practical examples where applicable."
)

QUESTIONS = [
    "What is KV cache in transformer attention?",
    "Explain prefill vs decode in LLM inference.",
    "How does tensor parallelism work for LLMs?",
    "Describe GPU memory hierarchy for inference.",
    "What is PagedAttention in vLLM?",
    "Explain continuous batching for LLM serving.",
    "What are KV cache quantization tradeoffs?",
    "How does speculative decoding reduce latency?",
    "Describe disaggregated prefill-decode serving.",
    "What optimizations does FlashAttention bring?",
]


def measure_ttft(url, model, system_prompt, question, max_tokens=30):
    """Send a streaming request and measure time to first token."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
    except Exception as e:
        return None, None, 0, str(e)

    end = time.perf_counter()

    if first_token_time is None:
        return None, None, 0, "no tokens"

    ttft_ms = (first_token_time - start) * 1000
    total_ms = (end - start) * 1000
    return ttft_ms, total_ms, token_count, None


def run_benchmark(url, model, num_rounds=3, num_requests=10):
    print(f"URL:    {url}")
    print(f"Model:  {model}")
    print(f"Rounds: {num_rounds}, Requests/round: {num_requests}")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
    print()

    # Warmup
    print("--- Warmup (2 requests) ---")
    for i in range(2):
        ttft, total, tokens, err = measure_ttft(url, model, SYSTEM_PROMPT, QUESTIONS[i], 10)
        if err:
            print(f"  Warmup {i+1}: ERROR {err}")
        else:
            print(f"  Warmup {i+1}: TTFT={ttft:.0f}ms")
    print()

    all_results = {}

    for rnd in range(1, num_rounds + 1):
        label = {1: "Cold", 2: "Warm", 3: "Hot"}.get(rnd, f"R{rnd}")
        print(f"--- Round {rnd} ({label}) ---")
        results = []

        for i in range(min(num_requests, len(QUESTIONS))):
            ttft, total, tokens, err = measure_ttft(
                url, model, SYSTEM_PROMPT, QUESTIONS[i], 30
            )
            if err:
                print(f"  Q{i+1:2d}: ERROR {err}")
            else:
                results.append(ttft)
                print(f"  Q{i+1:2d}: TTFT={ttft:7.1f}ms  Total={total:7.1f}ms  Tokens={tokens}")
            time.sleep(0.3)

        if results:
            avg = sum(results) / len(results)
            p50 = sorted(results)[len(results) // 2]
            mn = min(results)
            mx = max(results)
            print(f"  >> avg={avg:.0f}ms  p50={p50:.0f}ms  min={mn:.0f}ms  max={mx:.0f}ms")
            all_results[rnd] = {"avg": avg, "p50": p50, "min": mn, "max": mx, "values": results}
        print()

    # Final summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for rnd, r in all_results.items():
        label = {1: "Cold", 2: "Warm", 3: "Hot"}.get(rnd, f"R{rnd}")
        print(f"  Round {rnd} ({label:4s}): avg={r['avg']:7.1f}ms  p50={r['p50']:7.1f}ms  min={r['min']:7.1f}ms  max={r['max']:7.1f}ms")

    if 1 in all_results and 3 in all_results:
        cold_avg = all_results[1]["avg"]
        hot_avg = all_results[3]["avg"]
        diff = cold_avg - hot_avg
        pct = (diff / cold_avg) * 100 if cold_avg > 0 else 0
        print(f"\n  TTFT improvement (Cold->Hot): {diff:.0f}ms ({pct:.1f}%)")
        if pct > 10:
            print("  => Prefix caching is working! KV blocks are being reused.")
        else:
            print("  => No significant improvement. Prefix caching may not be active.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--requests", type=int, default=10)
    args = parser.parse_args()

    run_benchmark(args.url, args.model, args.rounds, args.requests)
