#!/bin/sh
# North-Mini-Code inside 8 GB: what each context length costs, read off the slots auto leaves.
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/North-Mini-Code-1.0-UD-Q4_K_M.gguf; O=bench/results/linux/north-mini-code
for c in 16384 65536 131072 262144; do
    timeout 600 systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -f bench/prompts/north-story.txt -n 4 -t 6 -c $c -lzm on \
        --cache-slots auto --io-threads 4 --stream-prefill > $O/ctx-$c.log 2>&1
    echo "ctx $c exit $? | $(grep -h 'expert_cache: auto' $O/ctx-$c.log | sed 's/.*auto: //')" | tee -a $O/context.txt
done
