#!/bin/sh
# Qwen3-Coder-30B inside 8 GB at 32k context: K/V cache types on the code prompt (128 tokens) and on a long prompt
# (the first 30 KB of llama.cpp's src/llama-context.cpp at 2f3fd0252, summarised; 64 tokens). Types are K/V.
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf; O=bench/results/linux/qwen3-coder-30b
run() { # tag K V n prompt
    tag=$1 k=$2 v=$3 n=$4 p=$5
    python3 bench/evict-file.py "$M"
    timeout 1800 systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -f $p -n $n -t 6 -c 32768 -b 16384 -fa on -ctk $k -ctv $v -lzm on \
        --json $O/$tag.json --cache-slots auto --io-threads 4 --stream-prefill > $O/$tag.log 2>&1
    echo "$tag exit $? | $(grep -h 'expert_cache: auto' $O/$tag.log | sed 's/.*= //') | $(grep -h -E 'main: (prefill|decode)' $O/$tag.log | sed 's/.*main: //; s/ page faults.*//' | tr '\n' ' ') | $(grep -h 'tokens hash' $O/$tag.log | sed 's/.*hash \([0-9a-f]*\).*/\1/')" | tee -a $O/kv2.txt
}
run kvspeed-q8q4 q8_0 q4_0 128 bench/prompts/qc30-story.txt
for kv in "f16 f16" "q8_0 q8_0" "q8_0 q4_0"; do set -- $kv
    run kvcode-$1-$2 $1 $2 128 bench/prompts/qc30-code.txt
    run kvlong-$1-$2 $1 $2 64 bench/prompts/qc30-long.txt
done
