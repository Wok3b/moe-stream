#!/bin/sh
# Kimi-Linear inside 8 GB: what each context length costs, read off the KV/state sizes and the slots auto leaves.
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf; O=bench/results/linux/kimi-linear
for c in 16384 65536 131072 262144 1048576; do
    timeout 600 systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -f bench/prompts/kimi-linear-story.txt -n 8 -t 6 -c $c -lzm on \
        --cache-slots auto --io-threads 4 --stream-prefill > $O/ctx-$c.log 2>&1
    echo "ctx $c exit $? | $(grep -h -E 'llama_kv_cache: size|llama_memory_recurrent: size|memory_recurrent.*size' $O/ctx-$c.log | sed 's/.*: size = *//' | tr '\n' ';') | $(grep -h 'expert_cache: auto' $O/ctx-$c.log | sed 's/.*auto: //')" | tee -a $O/context.txt
done
