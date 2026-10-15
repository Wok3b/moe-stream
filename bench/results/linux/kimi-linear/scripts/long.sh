#!/bin/sh
# Kimi-Linear: decode after a long prompt (the first 30 KB of llama.cpp's src/llama-context.cpp at 2f3fd0252,
# summarised), no limit and 8 GB, cold model file, 64 tokens out. -b 16384: moe-run sends the prompt in one call.
cd "$HOME/moe-stream" || exit 1
M=$HOME/models/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf; O=bench/results/linux/kimi-linear
for b in infinity 8G; do
    python3 bench/evict-file.py "$M"
    timeout 1800 systemd-run --user --scope --quiet -p MemoryMax=$b -p MemorySwapMax=0 -- \
        llama.cpp/build/bin/llama-moe-run -m "$M" -f bench/prompts/kimi-linear-long.txt -n 64 -t 6 -c 16384 -b 16384 -lzm on \
        --json $O/long-$b.json --cache-slots auto --io-threads 4 --stream-prefill > $O/long-$b.log 2>&1
    echo "long $b exit $?" | tee -a $O/long.txt
    grep -E "main: (prefill|decode|cache:|stream:|tokens hash)" $O/long-$b.log | sed 's/^.*main: /    /' | tee -a $O/long.txt
done
