#!/bin/sh
# Qwen3-Coder-30B: what the KV cache type costs inside 8 GB.
# 1. context memory: f16 / q8_0 / q4_0 at 16k / 32k / 64k, read off the slots auto leaves (16 tokens out)
# 2. speed and text: 32k context, story prompt, 128 tokens, each type
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf; O=bench/results/linux/qwen3-coder-30b
run() { # tag ctx type n [args]
    tag=$1 c=$2 t=$3 n=$4; shift 4
    python3 bench/evict-file.py "$M"
    timeout 1800 systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -n $n -t 6 -c $c -fa on -ctk $t -ctv $t -lzm on \
        --json $O/$tag.json --cache-slots auto --io-threads 4 --stream-prefill "$@" > $O/$tag.log 2>&1
    echo "$tag exit $? | $(grep -h 'expert_cache: auto' $O/$tag.log | sed 's/.*auto: //') | $(grep -h -E 'main: (prefill|decode)' $O/$tag.log | sed 's/.*main: //; s/ page faults.*//' | tr '\n' ' ') | $(grep -h 'tokens hash' $O/$tag.log | sed 's/.*hash \([0-9a-f]*\).*/\1/')" | tee -a $O/kv.txt
}
for t in f16 q8_0 q4_0; do for c in 16384 32768 65536; do
    run kv-$t-$c $c $t 16 -f bench/prompts/qc30-story.txt
done; done
for t in f16 q8_0 q4_0; do
    run kvspeed-$t 32768 $t 128 -f bench/prompts/qc30-story.txt
done
