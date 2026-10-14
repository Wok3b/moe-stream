#!/bin/bash
# Cold start-up of the moe-stream llama-server: seconds to /health, then one short request (TTFT, decode).
#   startup.sh TAG [server args...]
B=$(cd "$(dirname "$0")" && pwd); O=$B/startup; tag=$1; shift
GGUF=$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
python3 "$HOME/moe-stream/bench/evict-file.py" "$GGUF"
t0=$(date +%s.%N)
MOE_CACHE_SLOTS=auto MOE_IO_THREADS=4 "${BIN:-$HOME/moe-stream/llama.cpp/build/bin/llama-server}" -m "$GGUF" \
    -c 16384 -np 1 -lzm on -rea off --host 127.0.0.1 --port 8080 "$@" > "$O/$tag.log" 2>&1 &
pid=$!
until curl -sf http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"'; do
    kill -0 $pid 2>/dev/null || { echo "$tag: server died"; exit 1; }; sleep 0.1; done
t1=$(date +%s.%N)
r=$(curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a 400-word story about a lighthouse keeper who finds a message in a bottle."}],"max_tokens":32,"temperature":0}')
t2=$(date +%s.%N)
kill $pid; wait $pid 2>/dev/null
echo "$r" > "$O/$tag.json"
echo "$r" | python3 -c "
import json,sys; t=json.load(sys.stdin)['timings']
print(f'$tag: ready {$t1-$t0:.1f}s, first request {$t2-$t1:.1f}s (prompt {t[\"prompt_ms\"]/1000:.1f}s, decode {t[\"predicted_per_second\"]:.2f} tok/s)')" | tee -a "$O/summary.txt"
