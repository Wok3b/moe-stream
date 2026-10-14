#!/bin/bash
# One measured round: model files dropped from the page cache, the server started under a
# memory budget (no swap), the workload sent through Colibri's own HTTP benchmark, then stopped.
#   round.sh moe|coli BUDGET ROUND [extra server args...]     BUDGET: 8G, 10G, ... or infinity
# Env for coli: COLI_ENV="DIRECT=1 PILOT=1" (engine env), extra args e.g. --cap 40
set -u
E=$1 BUD=$2 R=$3; shift 3
B=$(cd "$(dirname "$0")" && pwd)
O=$B/results; mkdir -p "$O"
tag=$E-$BUD-r$R${TAG:+-$TAG}
unit=cmp-$tag
GGUF=$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
CDIR=$HOME/models/qwen36-colibri-i4-gs64

case $E in
moe)
    python3 "$HOME/moe-stream/bench/evict-file.py" "$GGUF"
    port=8080 model=qwen36
    cmd=(env MOE_CACHE_SLOTS=auto MOE_IO_THREADS=4
         "$HOME/moe-stream/llama.cpp/build/bin/llama-server" -m "$GGUF" -c 16384 -np 1 -lzm on -rea off
         --host 127.0.0.1 --port $port "$@") ;;
coli)
    python3 "$HOME/moe-stream/bench/evict-file.py" "$CDIR"/*.safetensors
    port=8000 model=qwen36
    cmd=(env ${COLI_ENV:-} "${COLI_BIN:-$HOME/colibri/c/coli}" serve --model "$CDIR" --ctx 16384 --no-think --temp 0
         --host 127.0.0.1 --port $port --model-id $model "$@") ;;
*) echo "engine: moe or coli" >&2; exit 2 ;;
esac

avail=$(awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)
t0=$(date +%s.%N)
systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax="$BUD" -p MemorySwapMax=0 -- \
    "${cmd[@]}" > "$O/$tag.server.log" 2>&1 &
cg=
for _ in $(seq 1 1200); do
    [ -z "$cg" ] && cg=$(systemctl --user show -p ControlGroup --value "$unit.scope" 2>/dev/null)
    if curl -sf "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"'; then break; fi
    if ! systemctl --user is-active --quiet "$unit.scope" && [ -n "$cg" ]; then echo "$tag: server died" >&2; break; fi
    sleep 0.5
done
t1=$(date +%s.%N)

python3 "$HOME/colibri/c/tools/benchmark_http_serving.py" --base-url "http://127.0.0.1:$port/v1" \
    --model $model --workload "$B/workload.jsonl" --concurrency 1 --repeats ${REPEATS:-2} \
    --max-tokens 128 --temperature 0 --timeout 900 --output "$O/$tag.json" > "$O/$tag.bench.log" 2>&1
rc=$?
[ "${CAPTURE:-0}" = 1 ] && python3 "$B/capture.py" "http://127.0.0.1:$port/v1" $model "$B/workload.jsonl" "$O/$tag.text.json"

peak=$(cat "/sys/fs/cgroup$cg/memory.peak" 2>/dev/null || echo 0)
events=$(grep -E '^(oom|oom_kill) ' "/sys/fs/cgroup$cg/memory.events" 2>/dev/null | tr '\n' ' ')
systemctl --user stop "$unit.scope" 2>/dev/null
sleep 2
python3 - "$O/$tag.json" "$tag" "$rc" "$t0" "$t1" "$peak" "$events" "$avail" <<'EOF' | tee -a "$O/summary.txt"
import json, sys
path, tag, rc, t0, t1, peak, events, avail = sys.argv[1:]
try:
    rep = json.load(open(path))
except Exception as e:
    print(f"{tag}: no report ({e}), bench exit {rc}, {events}")
    sys.exit()
cells = []
for r in rep.get("requests", []):
    n, d, f = r.get("completion_tokens"), r.get("duration_seconds"), r.get("first_output_seconds")
    if r.get("status") == "ok" or r.get("success"):
        dec = (n - 1) / (d - f) if n and f is not None and d > f else float("nan")
        cells.append(f"ttft {f:.1f}s n {n} dec {dec:.2f}")
    else:
        cells.append(f"FAIL {r.get('failure_category') or r.get('error')}")
print(f"{tag}: avail {avail} GiB ready {float(t1) - float(t0):.0f}s peak {int(peak) / 2**30:.2f} GiB bench exit {rc} {events}| " + " | ".join(cells))
EOF
