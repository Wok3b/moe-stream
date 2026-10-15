#!/bin/sh
# Results 20: the MTP GGUF (a different quant) at 12 GB, cold model file, 128 tokens.
# width below 8 and --prefetch-evict come from an experimental patch that was not kept.
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/Qwen3.5-122B-A10B-MTP-UD-IQ3_XXS.gguf; P=bench/prompts/qwen36-story.txt; O=bench/results/linux/q122
run() { name=$1; shift
    tag=mtpfile-tune-$name
    python3 bench/evict-file.py "$M"
    timeout 900 systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -f "$P" -n 128 -t 6 -c 16384 -lzm on --json $O/$tag.json \
        --cache-slots auto --stream-prefill "$@" > $O/$tag.log 2>&1
    echo "$name exit $? | $(grep -h 'main: decode' $O/$tag.log | sed 's/.*main: //') | $(grep -h 'tokens hash' $O/$tag.log | sed 's/.*hash \([0-9a-f]*\).*/\1/') | $(grep -h 'main: cache:' $O/$tag.log | sed 's/.*per token: //; s/, evictions.*wait/ ... wait/; s/, I\/O.*//')" | tee -a $O/mtpfile-tune.txt
}
run base       --io-threads 4
run width4     --io-threads 4 --prefetch-width 4
run width6     --io-threads 4 --prefetch-width 6
run evict0.5   --io-threads 4 --prefetch-evict 0.5
run evict1     --io-threads 4 --prefetch-evict 1
run evict2     --io-threads 4 --prefetch-evict 2
run base2      --io-threads 4
