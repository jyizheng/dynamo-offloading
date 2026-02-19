#!/usr/bin/env python3
"""
KVBM Long Prefix Benchmark

Uses a very long system prompt (~5000 tokens) to make prefill time significant.
On H100 with Qwen3-8B, 5000 tokens prefill takes ~30ms.
With KVBM prefix caching, this can be reduced to ~5ms (onboard from G2/G3).
"""

import argparse
import json
import time
import urllib.request
import concurrent.futures

# Generate a very long system prompt (~5000 tokens)
# Each paragraph is unique to prevent tokenizer compression
def generate_long_prompt():
    sections = []
    sections.append(
        "You are a highly specialized AI research assistant with deep expertise across "
        "multiple technical domains. Below is your complete knowledge base that you must "
        "reference when answering questions. Study all sections carefully."
    )

    topics = [
        ("Kubernetes Architecture",
         "The Kubernetes control plane consists of several key components. The API Server "
         "(kube-apiserver) serves as the central management entity, handling all REST operations "
         "and serving as the frontend to the cluster's shared state. It validates and configures "
         "data for API objects including pods, services, replication controllers, and others. "
         "etcd is a consistent and highly-available key-value store used as Kubernetes' backing "
         "store for all cluster data. It stores the entire cluster state and configuration. "
         "The Scheduler (kube-scheduler) watches for newly created pods with no assigned node "
         "and selects a node for them to run on based on individual and collective resource "
         "requirements, hardware/software/policy constraints, affinity and anti-affinity "
         "specifications, data locality, inter-workload interference, and deadlines. "
         "The Controller Manager runs controller processes including the node controller, "
         "job controller, endpoint slice controller, and service account controller. "
         "Each controller is a separate process but they are compiled into a single binary. "
         "The kubelet is an agent that runs on each node ensuring containers are running in a pod. "
         "kube-proxy maintains network rules on nodes allowing network communication to pods. "
         "Container runtimes like containerd or CRI-O manage the execution and lifecycle of containers."),

        ("Distributed Consensus Protocols",
         "Raft consensus algorithm divides time into terms of arbitrary length numbered with "
         "consecutive integers. Each term begins with an election. Raft ensures the following: "
         "Election Safety - at most one leader can be elected in a given term. Leader Append-Only - "
         "a leader never overwrites or deletes entries in its log. Log Matching - if two logs "
         "contain an entry with the same index and term, then the logs are identical in all "
         "entries up through the given index. Leader Completeness - if a log entry is committed "
         "in a given term, that entry will be present in the logs of the leaders for all higher "
         "numbered terms. State Machine Safety - if a server has applied a log entry at a given "
         "index to its state machine, no other server will ever apply a different log entry for "
         "the same index. Paxos requires three phases: prepare, promise, and accept. Multi-Paxos "
         "optimizes this by electing a stable leader. ZAB (ZooKeeper Atomic Broadcast) provides "
         "total order broadcast with leader election, discovery, synchronization, and broadcast "
         "phases. Byzantine fault tolerance protocols like PBFT require 3f+1 nodes to tolerate "
         "f Byzantine failures and proceed in three phases: pre-prepare, prepare, and commit."),

        ("GPU Computing and CUDA Programming",
         "Modern GPU architecture features Streaming Multiprocessors (SMs), each containing "
         "multiple CUDA cores, tensor cores, load/store units, and special function units. "
         "The CUDA programming model organizes threads into a hierarchy: grids contain blocks, "
         "blocks contain warps, and warps contain 32 threads that execute in SIMT fashion. "
         "Memory hierarchy includes per-thread registers (fastest, ~1 cycle), shared memory "
         "per block (~5 cycles, programmer-managed), L1 cache per SM (~20 cycles), L2 cache "
         "shared across SMs (~200 cycles), and global memory (~400-800 cycles, highest latency). "
         "Memory coalescing occurs when threads in a warp access contiguous memory addresses, "
         "allowing the hardware to combine individual access requests into fewer transactions. "
         "Bank conflicts happen when multiple threads access the same shared memory bank. "
         "Occupancy is the ratio of active warps to maximum warps supported by an SM. "
         "Higher occupancy can hide memory latency through warp scheduling. "
         "CUDA streams enable concurrent kernel execution and overlap computation with data "
         "transfers. CUDA graphs capture a sequence of operations for efficient replay. "
         "Unified Memory provides a single memory space accessible from any processor. "
         "NVLink provides high-bandwidth GPU-to-GPU interconnect up to 900 GB/s per link."),

        ("Transformer Architecture and Attention",
         "The transformer model architecture uses self-attention mechanisms to process sequences. "
         "Multi-head attention projects input into query, key, and value matrices using learned "
         "weight matrices WQ, WK, WV. Attention is computed as softmax(QK^T/sqrt(d_k))V where "
         "d_k is the dimension of key vectors. Multi-head attention runs h parallel attention "
         "functions (heads), concatenates outputs, and projects through WO. This allows the "
         "model to jointly attend to information from different representation subspaces. "
         "Feed-forward networks in each transformer block consist of two linear transformations "
         "with a GELU or SiLU activation: FFN(x) = W2 * activation(W1 * x + b1) + b2. "
         "Modern architectures use RMSNorm instead of LayerNorm for faster normalization. "
         "Rotary Position Embeddings (RoPE) encode position information by rotating query "
         "and key vectors, enabling length extrapolation. Grouped Query Attention (GQA) "
         "shares key-value heads across multiple query heads, reducing KV cache size. "
         "Multi-Query Attention (MQA) uses a single key-value head for all query heads. "
         "FlashAttention tiles the attention computation to fit in SRAM, using online "
         "softmax to avoid materializing the full attention matrix in HBM."),

        ("KV Cache Management",
         "In autoregressive decoding, the KV cache stores key and value tensors from previous "
         "tokens to avoid recomputation. For a model with L layers, H heads, and dimension D, "
         "each token adds 2 * L * H * D bytes (for both K and V in the chosen precision). "
         "For Llama-70B with 80 layers, 8 KV heads, 128 dim in FP16: each token uses "
         "80 * 2 * 8 * 128 * 2 = 327,680 bytes. A 4096-token sequence needs ~1.28 GB. "
         "PagedAttention manages KV cache in fixed-size blocks (typically 16-32 tokens), "
         "similar to OS virtual memory paging. This eliminates memory fragmentation and "
         "enables memory sharing between requests with common prefixes. "
         "Prefix caching identifies common prefixes across requests (e.g., system prompts) "
         "and reuses their KV cache blocks. This avoids redundant prefill computation. "
         "KVBM (KV Block Manager) extends this with multi-tier caching: "
         "G1 (GPU VRAM) for active inference, G2 (CPU pinned memory) for fast offloading, "
         "G3 (local NVMe disk) for larger capacity, G4 (remote storage) for near-infinite "
         "capacity. Blocks are offloaded down the hierarchy when GPU memory is pressured "
         "and onboarded back when needed. The offload filter uses frequency-based policies "
         "to prioritize which blocks to offload."),

        ("InfiniBand and RDMA Networking",
         "InfiniBand is a high-bandwidth, low-latency networking technology. The subnet "
         "manager assigns LIDs (Local Identifiers) to each port and computes routing tables. "
         "Each port has a GUID and can have multiple GIDs for RoCE compatibility. "
         "Queue Pairs (QPs) are the fundamental communication abstraction: each QP has a "
         "Send Queue and Receive Queue. Completion Queues (CQs) receive work completion "
         "notifications. Memory Regions (MRs) register memory with the HCA for RDMA access. "
         "Protection Domains (PDs) provide access control for QPs and MRs. "
         "RDMA operations include: Send/Receive (two-sided, both sides post work requests), "
         "Read/Write (one-sided, only initiator posts WR, target CPU not involved), and "
         "Atomic operations (compare-and-swap, fetch-and-add). "
         "RoCE v2 encapsulates IB transport in UDP/IP packets for Ethernet networks. "
         "Adaptive routing allows switches to dynamically select output ports. "
         "DCQCN (Data Center QCN) provides congestion control for RoCE using ECN marking. "
         "PFC (Priority Flow Control) prevents packet loss by pausing traffic on specific "
         "priority levels. GPUDirect RDMA enables direct data transfer between GPU memory "
         "and network adapters, bypassing CPU and system memory entirely."),

        ("Storage Systems and Data Placement",
         "Ceph uses CRUSH (Controlled Replication Under Scalable Hashing) algorithm for "
         "deterministic data placement without central lookup. CRUSH maps objects to placement "
         "groups (PGs), then PGs to OSDs using a hierarchical cluster map. The algorithm "
         "considers failure domains (hosts, racks, rooms) to ensure replica separation. "
         "BlueStore is Ceph's default storage backend that manages raw block devices directly, "
         "bypassing the filesystem layer. It uses RocksDB for metadata and allocates space "
         "using a bitmap-based allocator. Erasure coding provides space-efficient redundancy: "
         "k data chunks + m coding chunks can tolerate m failures while using only "
         "(k+m)/k times the original data size, compared to m+1 times for m-way replication. "
         "FoundationDB uses a shared-nothing architecture with role-based servers: "
         "coordinators (Paxos-based), cluster controller, master/proxy, resolvers, "
         "storage servers, and log servers. It provides serializable ACID transactions "
         "across arbitrary key ranges with optimistic concurrency control. "
         "TiKV uses a multi-Raft architecture where data is split into regions, each "
         "managed by an independent Raft group. PD (Placement Driver) manages metadata "
         "and provides timestamp oracle for distributed transactions."),

        ("ML Inference Optimization",
         "Quantization reduces model precision to decrease memory usage and increase throughput. "
         "INT8 quantization uses calibration data to determine scale factors for each tensor. "
         "Symmetric quantization: q = round(x/s) where s = max(|x|)/127. "
         "Asymmetric quantization: q = round(x/s) + z where z is zero-point offset. "
         "GPTQ uses second-order information (Hessian) for optimal weight quantization. "
         "AWQ (Activation-aware Weight Quantization) identifies salient weight channels "
         "using activation magnitude and applies per-channel scaling before quantization. "
         "SmoothQuant smooths outlier activations by migrating quantization difficulty "
         "from activations to weights using a mathematically equivalent transformation. "
         "FP8 quantization (E4M3/E5M2) provides better dynamic range than INT8. "
         "Speculative decoding uses a small draft model to generate candidate tokens "
         "that the large model verifies in parallel, achieving up to 2-3x speedup "
         "for latency-bound single-request scenarios. "
         "Continuous batching dynamically adds new requests to running batches as slots "
         "become available, maximizing GPU utilization compared to static batching."),
    ]

    for title, content in topics:
        sections.append(f"\n## {title}\n{content}")

    return "\n".join(sections)


SYSTEM_PROMPT = generate_long_prompt()

TARGET_QUESTIONS = [
    "Explain how Raft leader election works step by step.",
    "What is memory coalescing in CUDA and why does it matter?",
    "How does PagedAttention reduce KV cache fragmentation?",
    "Describe the CRUSH algorithm in Ceph.",
    "What is GPUDirect RDMA and when should it be used?",
]

FLOOD_PROMPTS = [
    f"Question {i}: Tell me about {'databases' if i%4==0 else 'algorithms' if i%4==1 else 'networks' if i%4==2 else 'operating systems'} topic variant {i}. Be detailed."
    for i in range(500)
]


def send_request(url, model, messages, max_tokens=30, stream=True):
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


def run_target(url, model, label):
    print(f"\n--- {label} ---")
    results = []
    for i, q in enumerate(TARGET_QUESTIONS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        ttft, total, tokens, err = send_request(url, model, messages)
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


def run_flood(url, model, n=50, max_tokens=200, concurrency=4):
    print(f"\n--- Flood ({n} requests, concurrency={concurrency}) ---")
    done = [0]

    def do(idx):
        msgs = [{"role": "user", "content": FLOOD_PROMPTS[idx % len(FLOOD_PROMPTS)]}]
        _, _, _, err = send_request(url, model, msgs, max_tokens=max_tokens, stream=False)
        done[0] += 1
        if done[0] % 10 == 0:
            print(f"  Progress: {done[0]}/{n}")
        return err

    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(do, i) for i in range(n)]
        for f in concurrent.futures.as_completed(futs):
            if f.result():
                errors += 1
    print(f"  Done: {n - errors} ok, {errors} errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--flood-requests", type=int, default=100)
    parser.add_argument("--flood-tokens", type=int, default=200)
    parser.add_argument("--flood-concurrency", type=int, default=4)
    args = parser.parse_args()

    prompt_words = len(SYSTEM_PROMPT.split())
    prompt_est_tokens = int(prompt_words * 1.3)
    print(f"URL:    {args.url}")
    print(f"Model:  {args.model}")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars, ~{prompt_words} words, ~{prompt_est_tokens} est tokens")
    print(f"Flood: {args.flood_requests} × {args.flood_tokens} tokens")

    # Warmup
    print("\n--- Warmup ---")
    ttft, _, _, _ = send_request(args.url, args.model,
                                  [{"role": "user", "content": "Hi"}], max_tokens=5)
    print(f"  TTFT={ttft:.0f}ms")

    p1 = run_target(args.url, args.model, "Phase 1: Target (Cold - first prefill)")
    p1b = run_target(args.url, args.model, "Phase 1b: Target (GPU-cached)")
    run_flood(args.url, args.model, args.flood_requests,
              args.flood_tokens, args.flood_concurrency)
    p3 = run_target(args.url, args.model, "Phase 3: Target (After Eviction)")
    p3b = run_target(args.url, args.model, "Phase 3b: Target (2nd after eviction)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    def fmt(label, vals):
        if vals:
            avg = sum(vals) / len(vals)
            p50 = sorted(vals)[len(vals) // 2]
            print(f"  {label:35s}: avg={avg:7.1f}ms  p50={p50:7.1f}ms")
            return avg
        return 0

    a1 = fmt("Phase 1  (Cold prefill)", p1)
    a1b = fmt("Phase 1b (GPU-cached)", p1b)
    a3 = fmt("Phase 3  (After eviction)", p3)
    a3b = fmt("Phase 3b (After eviction 2nd)", p3b)

    print(f"\n  Prefill savings from GPU cache: {a1 - a1b:.1f}ms ({(a1-a1b)/a1*100:.1f}%)" if a1 > 0 else "")
    print(f"  Eviction penalty (Phase3 vs 1b): {a3 - a1b:.1f}ms")
    if a3 > a1b + 5:
        print(f"  => KV cache was evicted! Phase 3 needs re-prefill.")
        print(f"  => With KVBM, Phase 3 would onboard from G2/G3 instead of re-prefilling.")
    else:
        print(f"  => KV cache NOT fully evicted. Need more flood requests or smaller GPU cache.")


if __name__ == "__main__":
    main()
