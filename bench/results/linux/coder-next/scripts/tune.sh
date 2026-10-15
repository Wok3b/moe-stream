#!/bin/sh
# Qwen3-Coder-Next at 12 GB, code prompt: one cache/prefetch setting changed at a time, cold model file, 128 tokens.
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/Qwen3-Coder-Next-Q4_K_S.gguf; P=${PROMPT:-bench/prompts/coder-next-code.txt}; O=bench/results/linux/coder-next
mkdir -p $O
run() { name=$1; shift
    tag=tune-$name
    python3 bench/evict-file.py "$M"
    timeout 900 systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -f "$P" -n 128 -t 6 -c 16384 -lzm on --json $O/$tag.json \
        --cache-slots auto "$@" > $O/$tag.log 2>&1
    echo "$name exit $? | $(grep -h 'main: decode' $O/$tag.log | sed 's/.*main: //') | $(grep -h 'tokens hash' $O/$tag.log | sed 's/.*hash \([0-9a-f]*\).*/\1/') | $(grep -h 'expert_cache: auto' $O/$tag.log | sed 's/.*= //') | $(grep -h 'main: cache:' $O/$tag.log | sed 's/.*per token: //; s/, evictions.*wait/ ... wait/; s/, I\/O.*//')" | tee -a $O/tune.txt
}
run base       --io-threads 4 --stream-prefill
run nostream   --io-threads 4
run depth2     --io-threads 4 --stream-prefill --prefetch-depth 2
run width14    --io-threads 4 --stream-prefill --prefetch-width 14
run width20    --io-threads 4 --stream-prefill --prefetch-width 20
run io6        --io-threads 6 --stream-prefill
run io8        --io-threads 8 --stream-prefill
run t5         --io-threads 4 --stream-prefill -t 5
run base2      --io-threads 4 --stream-prefill
