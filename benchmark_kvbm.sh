#!/bin/bash
# KVBM Benchmark: Compare Baseline vs KVBM with prefix caching
#
# Usage:
#   ./benchmark_kvbm.sh baseline   # Test without KVBM
#   ./benchmark_kvbm.sh kvbm       # Test with KVBM
#   ./benchmark_kvbm.sh all        # Run both sequentially
#
# Prerequisites:
#   - Frontend pod running (dynamo-frontend)
#   - Worker pod running (dynamo-worker)
#   - curl available

set -e

FRONTEND_IP="${FRONTEND_IP:-172.29.68.78}"
BASE_URL="http://${FRONTEND_IP}:8000"
MODEL="Qwen/Qwen3-8B"
RESULTS_DIR="/root/3fs_deploy/dynamo_deploy/benchmark_results"
mkdir -p "$RESULTS_DIR"

# Shared system prompt (~500 tokens) - same across all requests to test prefix caching
SYSTEM_PROMPT="You are an expert AI assistant specializing in software engineering, distributed systems, and cloud infrastructure. You provide detailed, accurate, and well-structured answers. You always consider edge cases, performance implications, and best practices. When discussing code, you provide concrete examples with proper error handling. You are particularly knowledgeable about: Kubernetes orchestration, container networking, storage systems (distributed file systems, object storage, block storage), GPU computing and CUDA programming, machine learning inference optimization, RDMA and high-performance networking, and systems programming in Rust and C++. Please provide thorough answers with practical examples where applicable. Always structure your response with clear headings and bullet points for readability."

# Different user questions - each shares the same system prompt (prefix)
QUESTIONS=(
  "Explain how KV cache works in transformer attention mechanisms and why it speeds up autoregressive generation."
  "What are the key differences between prefill and decode phases in LLM inference?"
  "How does tensor parallelism work for serving large language models across multiple GPUs?"
  "Describe the memory hierarchy in modern GPU systems and how it affects LLM inference performance."
  "What is PagedAttention and how does it improve KV cache memory efficiency in vLLM?"
  "Explain the concept of continuous batching and how it improves throughput in LLM serving."
  "What are the tradeoffs between different KV cache quantization approaches (FP16, FP8, INT8)?"
  "How does speculative decoding work and when is it beneficial for inference latency?"
  "Describe the architecture of a disaggregated prefill-decode serving system for LLMs."
  "What optimizations does FlashAttention bring to transformer inference and how does it reduce memory usage?"
)

# Send a single request and measure TTFT
send_request() {
  local question="$1"
  local max_tokens="${2:-50}"

  local payload=$(python3 -c "
import json
data = {
    'model': '$MODEL',
    'messages': [
        {'role': 'system', 'content': '''$SYSTEM_PROMPT'''},
        {'role': 'user', 'content': '''$question'''}
    ],
    'max_tokens': $max_tokens,
    'stream': True
}
print(json.dumps(data))
")

  # Measure time to first token using streaming
  local start_time=$(date +%s%N)
  local first_token_time=""
  local total_tokens=0

  while IFS= read -r line; do
    if [[ "$line" == data:* ]] && [[ "$line" != "data: [DONE]" ]]; then
      if [ -z "$first_token_time" ]; then
        first_token_time=$(date +%s%N)
      fi
      total_tokens=$((total_tokens + 1))
    fi
  done < <(curl -s -N "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null)

  local end_time=$(date +%s%N)

  if [ -z "$first_token_time" ]; then
    echo "ERROR: No tokens received"
    return 1
  fi

  local ttft_ms=$(( (first_token_time - start_time) / 1000000 ))
  local total_ms=$(( (end_time - start_time) / 1000000 ))
  local itl_ms=0
  if [ "$total_tokens" -gt 1 ]; then
    itl_ms=$(( (end_time - first_token_time) / 1000000 / (total_tokens - 1) ))
  fi

  echo "$ttft_ms $total_ms $total_tokens $itl_ms"
}

# Run benchmark
run_benchmark() {
  local mode="$1"
  local output_file="$RESULTS_DIR/${mode}_$(date +%Y%m%d_%H%M%S).csv"

  echo "=== Running $mode benchmark ==="
  echo "Frontend: $BASE_URL"
  echo "Output: $output_file"
  echo ""

  # Check connectivity
  echo "Checking server health..."
  if ! curl -s --connect-timeout 5 "$BASE_URL/v1/models" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach $BASE_URL/v1/models"
    echo "Make sure frontend and worker are running."
    exit 1
  fi
  echo "Server is healthy."
  echo ""

  # Header
  echo "round,question_idx,ttft_ms,total_ms,tokens,itl_ms" > "$output_file"

  # Warmup: send 2 requests to warm up the model
  echo "--- Warmup (2 requests) ---"
  for i in 0 1; do
    echo -n "  Warmup $((i+1))... "
    result=$(send_request "${QUESTIONS[$i]}" 20)
    echo "TTFT=$(echo $result | awk '{print $1}')ms"
  done
  echo ""

  # Round 1: Cold requests (no prefix cache hits expected)
  echo "--- Round 1: Cold (first time seeing each prefix) ---"
  for i in "${!QUESTIONS[@]}"; do
    echo -n "  Q$((i+1)): "
    result=$(send_request "${QUESTIONS[$i]}" 50)
    ttft=$(echo $result | awk '{print $1}')
    total=$(echo $result | awk '{print $2}')
    tokens=$(echo $result | awk '{print $3}')
    itl=$(echo $result | awk '{print $4}')
    echo "TTFT=${ttft}ms, Total=${total}ms, Tokens=${tokens}, ITL=${itl}ms"
    echo "1,$((i+1)),$ttft,$total,$tokens,$itl" >> "$output_file"
    sleep 0.5
  done

  echo ""

  # Round 2: Warm requests (prefix cache should hit if KVBM is enabled)
  echo "--- Round 2: Warm (same prefix, should see cache hits with KVBM) ---"
  for i in "${!QUESTIONS[@]}"; do
    echo -n "  Q$((i+1)): "
    result=$(send_request "${QUESTIONS[$i]}" 50)
    ttft=$(echo $result | awk '{print $1}')
    total=$(echo $result | awk '{print $2}')
    tokens=$(echo $result | awk '{print $3}')
    itl=$(echo $result | awk '{print $4}')
    echo "TTFT=${ttft}ms, Total=${total}ms, Tokens=${tokens}, ITL=${itl}ms"
    echo "2,$((i+1)),$ttft,$total,$tokens,$itl" >> "$output_file"
    sleep 0.5
  done

  echo ""

  # Round 3: Repeat again (more cache hits expected)
  echo "--- Round 3: Hot (prefix fully cached in KVBM) ---"
  for i in "${!QUESTIONS[@]}"; do
    echo -n "  Q$((i+1)): "
    result=$(send_request "${QUESTIONS[$i]}" 50)
    ttft=$(echo $result | awk '{print $1}')
    total=$(echo $result | awk '{print $2}')
    tokens=$(echo $result | awk '{print $3}')
    itl=$(echo $result | awk '{print $4}')
    echo "TTFT=${ttft}ms, Total=${total}ms, Tokens=${tokens}, ITL=${itl}ms"
    echo "3,$((i+1)),$ttft,$total,$tokens,$itl" >> "$output_file"
    sleep 0.5
  done

  echo ""

  # Summary
  echo "=== Summary ($mode) ==="
  echo "Results saved to: $output_file"
  python3 -c "
import csv
rounds = {}
with open('$output_file') as f:
    reader = csv.DictReader(f)
    for row in reader:
        r = int(row['round'])
        if r not in rounds:
            rounds[r] = []
        rounds[r].append(int(row['ttft_ms']))

labels = {1: 'Cold', 2: 'Warm', 3: 'Hot'}
for r in sorted(rounds.keys()):
    vals = rounds[r]
    avg = sum(vals) / len(vals)
    mn = min(vals)
    mx = max(vals)
    p50 = sorted(vals)[len(vals)//2]
    print(f'  Round {r} ({labels[r]:4s}): avg={avg:.0f}ms, p50={p50}ms, min={mn}ms, max={mx}ms')
" 2>/dev/null || true

  echo ""
}

# Collect KVBM metrics
collect_metrics() {
  echo "=== KVBM Metrics ==="
  curl -s "http://10.2.10.55:8081/metrics" 2>/dev/null | grep kvbm || echo "  (metrics not available)"
  echo ""
}

# Main
case "${1:-all}" in
  baseline)
    run_benchmark "baseline"
    ;;
  kvbm)
    run_benchmark "kvbm"
    collect_metrics
    ;;
  all)
    echo "########################################"
    echo "# Step 1: Deploy BASELINE worker"
    echo "########################################"
    echo "Run:"
    echo "  kubectl delete pod -n dynamo-qwen dynamo-worker --ignore-not-found"
    echo "  kubectl apply -f 03-worker-baseline.yaml"
    echo "  # Wait for worker to be ready, then:"
    echo "  ./benchmark_kvbm.sh baseline"
    echo ""
    echo "########################################"
    echo "# Step 2: Deploy KVBM worker"
    echo "########################################"
    echo "Run:"
    echo "  kubectl delete pod -n dynamo-qwen dynamo-worker --ignore-not-found"
    echo "  kubectl apply -f 04-worker-kvbm-local.yaml"
    echo "  # Wait for worker to be ready (~1-2 min for disk cache init), then:"
    echo "  ./benchmark_kvbm.sh kvbm"
    echo ""
    echo "########################################"
    echo "# Step 3: Compare results"
    echo "########################################"
    echo "Compare TTFT between baseline Round 2/3 and kvbm Round 2/3."
    echo "With KVBM, Round 2/3 TTFT should be significantly lower"
    echo "due to prefix cache hits (KV blocks onboarded from G2/G3)."
    ;;
  *)
    echo "Usage: $0 {baseline|kvbm|all}"
    exit 1
    ;;
esac
