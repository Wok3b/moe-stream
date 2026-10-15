#!/bin/sh
# North-Mini-Code: prefetch two layers ahead (depth 2) against the default, no limit and 8 GB, both prompts.
# Its router input is the previous layer's output, so depth 1 is exact but has only one attention to hide the read.
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/North-Mini-Code-1.0-UD-Q4_K_M.gguf; O=bench/results/linux/north-mini-code
for b in infinity 8G; do for p in story code; do for d in 2 3; do
    tag=tune-$b-$p-depth$d
    python3 bench/evict-file.py "$M"
    timeout 900 systemd-run --user --scope --quiet -p MemoryMax=$b -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -f bench/prompts/north-$p.txt -n 128 -t 6 -c 16384 -lzm on --json $O/$tag.json \
        --cache-slots auto --io-threads 4 --stream-prefill --prefetch-depth $d > $O/$tag.log 2>&1
    echo "$tag exit $? | $(grep -h 'main: decode' $O/$tag.log | sed 's/.*main: //; s/ page faults.*//') | $(grep -h 'tokens hash' $O/$tag.log | sed 's/.*hash \([0-9a-f]*\).*/\1/') | $(grep -h 'main: cache:' $O/$tag.log | sed 's/.*per token: //; s/, evictions.*wait/ ... wait/; s/, I\/O.*//')" | tee -a $O/tune.txt
done; done; done
