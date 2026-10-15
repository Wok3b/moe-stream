#!/bin/bash
# Results 20: llama-server on the 122B MTP GGUF under a budget, cold: story + code prompts, 128 greedy tokens each.
#   mtp.sh TAG BUDGET [server args...]     prints decode tok/s, draft acceptance; saves the answers
O=$HOME/moe-stream/bench/results/linux/q122/mtp; mkdir -p "$O"
tag=$1 bud=$2; shift 2
M=$HOME/models/Qwen3.5-122B-A10B-MTP-UD-IQ3_XXS.gguf
python3 "$HOME/moe-stream/bench/evict-file.py" "$M"
MOE_CACHE_SLOTS=auto MOE_IO_THREADS=4 systemd-run --user --scope --quiet --unit="mtp-$tag" -p MemoryMax="$bud" -p MemorySwapMax=0 -- \
    "$HOME/moe-stream/llama.cpp/build/bin/llama-server" -m "$M" -c 16384 -np 1 -lzm on -rea off \
    --host 127.0.0.1 --port 8094 "$@" > "$O/$tag.log" 2>&1 &
for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8094/health | grep -q ok && break; sleep 0.5; done
python3 - "$O/$tag" <<'PY'
import json, sys, urllib.request
out = sys.argv[1]
qs = {"story": "Write a 400-word story about a lighthouse keeper who finds a message in a bottle.",
      "code": "Write a Python function that reads a CSV file and returns the mean, median and standard deviation of every numeric column. Then explain how it handles missing values and non-numeric columns."}
res = {}
for k, q in qs.items():
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": 128, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8094/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
    t = r["timings"]
    res[k] = {"text": r["choices"][0]["message"]["content"], "timings": t}
    acc = f", drafts {t.get('draft_n_accepted')}/{t.get('draft_n')} accepted" if t.get("draft_n") else ""
    print(f"  {k}: prompt {t['prompt_per_second']:.1f} tok/s, decode {t['predicted_per_second']:.2f} tok/s{acc}")
json.dump(res, open(out + ".json", "w"), ensure_ascii=False, indent=1)
PY
systemctl --user stop "mtp-$tag.scope" 2>/dev/null; sleep 2
# run as: mtp.sh off 12G; mtp.sh n1 12G --spec-type draft-mtp --spec-draft-n-max 1; same for n2, n3
