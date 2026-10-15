#!/bin/sh
# Results 20: 122B at 12 GB, one cache/prefetch setting changed at a time, cold model file, 128 tokens.
# width4/width6 are clamped to the 8 experts the model picks, so those two runs repeat base.
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf; P=bench/prompts/qwen36-story.txt; O=bench/results/linux/q122
run() { name=$1; shift
    tag=tune-$name
    python3 bench/evict-file.py "$M"
    timeout 900 systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -f "$P" -n 128 -t 6 -c 16384 -lzm on --json $O/$tag.json \
        --cache-slots auto --stream-prefill "$@" > $O/$tag.log 2>&1
    echo "$name exit $? | $(grep -h 'main: decode' $O/$tag.log | sed 's/.*main: //') | $(grep -h 'tokens hash' $O/$tag.log | sed 's/.*hash \([0-9a-f]*\).*/\1/') | $(grep -h 'main: cache:' $O/$tag.log | sed 's/.*per token: //; s/, evictions.*wait/ ... wait/; s/, I\/O.*//')" | tee -a $O/tune.txt
}
run base       --io-threads 4
run depth2     --io-threads 4 --prefetch-depth 2
run width12    --io-threads 4 --prefetch-width 12
run width16    --io-threads 4 --prefetch-width 16
run io8        --io-threads 8
run lfu32      --io-threads 4 --cache-lfu 32
run lru        --io-threads 4 --cache-lfu 0
run base2      --io-threads 4
run width4     --io-threads 4 --prefetch-width 4
run width6     --io-threads 4 --prefetch-width 6
run depth0     --io-threads 4 --prefetch-depth 0
