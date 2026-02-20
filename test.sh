# Send same requests again to test disk cache onboard
for i in $(seq 1 5); do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain topic $i: distributed systems, consensus, Paxos, Raft, and their implementations in detail with examples and code.\"}],\"max_tokens\":200}" > /dev/null 2>&1 &
done
wait
echo "Second batch done"

sleep 2
curl -s http://localhost:6880/metrics | grep -E "kvbm_(offload|onboard|matched|cache_hit)"

