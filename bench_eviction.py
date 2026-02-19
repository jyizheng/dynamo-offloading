#!/usr/bin/env python3
"""
KVBM Eviction-Recovery Benchmark

Tests KVBM benefit by:
1. Phase 1 (Target): Send requests with shared prefix → measures cold TTFT
2. Phase 2 (Flood):  Send many diverse long requests to fill GPU KV cache → causes evictions
3. Phase 3 (Recall): Re-send Phase 1 requests → measures TTFT after eviction

Without KVBM: Phase 3 TTFT ≈ Phase 1 TTFT (must re-prefill, KV cache lost)
With KVBM:    Phase 3 TTFT < Phase 1 TTFT (KV blocks onboarded from G2/G3)

Requires limited GPU KV cache (set free_gpu_memory_fraction low, e.g. 0.10)
"""

import argparse
import json
import time
import sys
import urllib.request
import concurrent.futures

# ~2000 token system prompt (shared prefix across all target requests)
SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in software engineering, "
    "distributed systems, and cloud infrastructure. You provide detailed, "
    "accurate, and well-structured answers. You always consider edge cases, "
    "performance implications, and best practices. When discussing code, you "
    "provide concrete examples with proper error handling. You are particularly "
    "knowledgeable about Kubernetes orchestration including control plane "
    "components like the API server, etcd, scheduler, and controller manager, "
    "as well as kubelet, kube-proxy, CNI plugins, CSI drivers, CRDs, operators, "
    "admission webhooks, pod security policies, resource quotas, limit ranges, "
    "horizontal and vertical pod autoscaling, cluster autoscaling, node affinity, "
    "taints and tolerations, pod disruption budgets, stateful sets, daemon sets. "
    "You are also expert in distributed storage systems including consensus "
    "protocols like Raft, Paxos and ZAB, distributed hash tables, consistent "
    "hashing, vector clocks, Lamport timestamps, CAP theorem, CRDT data types, "
    "erasure coding, replication strategies both synchronous and asynchronous, "
    "storage tiering for hot warm and cold data, failure detection and recovery, "
    "split brain resolution and fencing mechanisms. You know Ceph RADOS CRUSH "
    "algorithm BlueStore, MinIO, FoundationDB, CockroachDB, TiKV, etcd and "
    "ZooKeeper in depth. For GPU computing you cover CUDA programming model "
    "with grids blocks threads and warps, memory hierarchy including registers "
    "shared memory L1 L2 cache global memory, memory coalescing, bank conflicts, "
    "occupancy optimization, warp divergence, CUDA streams events and graphs, "
    "unified memory, pinned memory, NVLink NVSwitch, GPUDirect RDMA and Storage, "
    "NCCL collective operations, tensor cores and mixed precision computing. "
    "For ML inference you cover transformer architecture with multi head attention "
    "feed forward networks layer normalization positional encodings, KV cache "
    "management, prefix caching, PagedAttention, FlashAttention, continuous and "
    "dynamic batching, speculative decoding, model parallelism including tensor "
    "pipeline and expert parallelism, quantization methods like INT8 FP8 GPTQ AWQ "
    "SmoothQuant, serving frameworks vLLM TensorRT-LLM Triton SGLang, request "
    "scheduling preemption and disaggregated serving. For networking you cover "
    "InfiniBand architecture subnet manager LIDs GIDs partition keys queue pairs "
    "completion queues memory regions, RDMA verbs, RoCE, adaptive routing, "
    "congestion control DCQCN TIMELY ECN PFC, SR-IOV, DPDK and io_uring. "
    "Always provide thorough answers with practical examples."
)

TARGET_QUESTIONS = [
    "What is KV cache in transformer attention?",
    "Explain prefill vs decode in LLM inference.",
    "How does tensor parallelism work for LLMs?",
    "Describe GPU memory hierarchy for inference.",
    "What is PagedAttention in vLLM?",
]

# Different prompts to flood the KV cache (each is unique, no shared prefix)
FLOOD_PROMPTS = [
    f"Write a detailed technical essay about topic number {i}. "
    f"Cover the history, current state, and future directions. "
    f"Include specific examples, comparisons with alternatives, "
    f"and practical recommendations for implementation. "
    f"Topic: {'distributed consensus algorithms' if i%5==0 else 'GPU memory optimization' if i%5==1 else 'container networking' if i%5==2 else 'inference serving' if i%5==3 else 'storage systems'} "
    f"variant {i}. Provide extensive detail and code examples."
    for i in range(200)
]


def send_request(url, model, messages, max_tokens=30, stream=True):
    """Send request, return (ttft_ms, total_ms, token_count, error)"""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            if stream:
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
            else:
                data = json.loads(resp.read())
                first_token_time = time.perf_counter()
                token_count = data.get("usage", {}).get("completion_tokens", 1)
    except Exception as e:
        return None, None, 0, str(e)

    end = time.perf_counter()
    if first_token_time is None:
        return None, None, 0, "no tokens"

    ttft_ms = (first_token_time - start) * 1000
    total_ms = (end - start) * 1000
    return ttft_ms, total_ms, token_count, None


def run_target_phase(url, model, label):
    """Send target requests (shared system prompt) and return TTFT list"""
    print(f"\n--- {label} ---")
    results = []
    for i, q in enumerate(TARGET_QUESTIONS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        ttft, total, tokens, err = send_request(url, model, messages, max_tokens=30)
        if err:
            print(f"  Q{i+1}: ERROR {err}")
        else:
            results.append(ttft)
            print(f"  Q{i+1}: TTFT={ttft:7.1f}ms  Total={total:7.1f}ms  Tokens={tokens}")
        time.sleep(0.2)

    if results:
        avg = sum(results) / len(results)
        p50 = sorted(results)[len(results) // 2]
        print(f"  >> avg={avg:.1f}ms  p50={p50:.1f}ms  min={min(results):.1f}ms  max={max(results):.1f}ms")
    return results


def run_flood_phase(url, model, num_requests=50, max_tokens=200, concurrency=4):
    """Send diverse requests to fill up KV cache"""
    print(f"\n--- Flood Phase ({num_requests} requests, concurrency={concurrency}) ---")
    completed = 0
    errors = 0

    def do_request(idx):
        messages = [{"role": "user", "content": FLOOD_PROMPTS[idx % len(FLOOD_PROMPTS)]}]
        _, _, _, err = send_request(url, model, messages, max_tokens=max_tokens, stream=False)
        return err

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(do_request, i): i for i in range(num_requests)}
        for f in concurrent.futures.as_completed(futures):
            err = f.result()
            if err:
                errors += 1
            else:
                completed += 1
            if (completed + errors) % 10 == 0:
                print(f"  Progress: {completed + errors}/{num_requests} (errors={errors})")

    print(f"  Flood complete: {completed} ok, {errors} errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--flood-requests", type=int, default=50,
                        help="Number of flood requests to fill KV cache")
    parser.add_argument("--flood-tokens", type=int, default=200,
                        help="Max tokens per flood request")
    parser.add_argument("--flood-concurrency", type=int, default=4)
    args = parser.parse_args()

    print(f"URL:    {args.url}")
    print(f"Model:  {args.model}")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"Flood: {args.flood_requests} requests × {args.flood_tokens} tokens")

    # Warmup
    print("\n--- Warmup ---")
    msgs = [{"role": "user", "content": "Hello, how are you?"}]
    ttft, _, _, _ = send_request(args.url, args.model, msgs, max_tokens=5)
    print(f"  TTFT={ttft:.0f}ms")

    # Phase 1: Target requests (cold - first time seeing this prefix)
    phase1 = run_target_phase(args.url, args.model, "Phase 1: Target (Cold)")

    # Phase 1b: Target again (should be cached in GPU already)
    phase1b = run_target_phase(args.url, args.model, "Phase 1b: Target (GPU-cached)")

    # Phase 2: Flood to evict KV cache
    run_flood_phase(args.url, args.model, args.flood_requests,
                    args.flood_tokens, args.flood_concurrency)

    # Phase 3: Re-send target requests (after eviction)
    phase3 = run_target_phase(args.url, args.model, "Phase 3: Target (After Eviction)")

    # Phase 3b: One more time
    phase3b = run_target_phase(args.url, args.model, "Phase 3b: Target (After Eviction, 2nd)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    def fmt(label, vals):
        if vals:
            avg = sum(vals) / len(vals)
            p50 = sorted(vals)[len(vals) // 2]
            print(f"  {label:30s}: avg={avg:7.1f}ms  p50={p50:7.1f}ms")
            return avg
        return 0

    a1 = fmt("Phase 1 (Cold)", phase1)
    a1b = fmt("Phase 1b (GPU-cached)", phase1b)
    a3 = fmt("Phase 3 (After Eviction)", phase3)
    a3b = fmt("Phase 3b (After Eviction 2nd)", phase3b)

    print()
    if a1 > 0 and a3 > 0:
        diff = a3 - a1
        if abs(diff) > 2:
            if diff > 0:
                print(f"  Phase 3 is {diff:.1f}ms SLOWER than Phase 1")
                print(f"  => GPU cache was evicted and needs re-prefill")
            else:
                print(f"  Phase 3 is {-diff:.1f}ms FASTER than Phase 1")
                print(f"  => KVBM successfully onboarded from G2/G3!")
        else:
            print(f"  Phase 3 ≈ Phase 1 (diff={diff:.1f}ms)")
            print(f"  => Either GPU cache not fully evicted, or TRT-LLM internal prefix cache still active")

    if a1b > 0 and a3b > 0:
        diff = a3b - a1b
        print(f"\n  GPU-cached vs After-Eviction-2nd: diff={diff:.1f}ms")


if __name__ == "__main__":
    main()
