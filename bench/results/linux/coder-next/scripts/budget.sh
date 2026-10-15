#!/bin/sh
# Qwen3-Coder-Next Q4_K_S under memory budgets, cold model file every run.
# mem-budget.sh writes to bench/results/linux/<TAG>-*; the files are moved into coder-next/ afterwards.
cd "$HOME/moe-stream" || exit 1
export MODEL=$HOME/models/Qwen3-Coder-Next-Q4_K_S.gguf
# prompts rendered with the model's own template
for s in 110 120; do for p in story code; do
    [ -s bench/prompts/coder-next-$p.txt ] && continue
    msg=$(sed -n '2p' bench/prompts/qwen36-$p.txt | sed 's/<|im_end|>$//')
    python3 bench/chat-prompt.py "$MODEL" "$msg" --out bench/prompts/coder-next-$p.txt || exit 1
done
# hashes at 32 tokens, mmap included
PROMPT=bench/prompts/coder-next-story.txt N_PREDICT=32 SLOTS=auto TAG=cn-hash bench/mem-budget.sh infinity 8G
# 128 tokens, story and code
for s in 110 120; do for p in story code; do
    PROMPT=bench/prompts/coder-next-$p.txt N_PREDICT=128 SLOTS=auto MMAP=0 TAG=cn-$p bench/mem-budget.sh 6G 8G 10G 12G infinity
done
echo BUDGET DONE
# no limit again with Firefox closed (~14.0 GB available instead of ~12.3), two rounds
for r in 1 2; do for p in story code; do
    PROMPT=bench/prompts/coder-next-$p.txt N_PREDICT=128 SLOTS=auto MMAP=0 TAG=cn-nofx$r-$p bench/mem-budget.sh infinity
done; done
# no limit, 110 and 120 slots by hand (auto keeps 15% of RAM free and takes 91)
for s in 110 120; do for p in story code; do
    python3 bench/evict-file.py "$MODEL"
    llama.cpp/build/bin/llama-moe-run -m "$MODEL" -f bench/prompts/coder-next-$p.txt -n 128 -t 6 -c 16384 -lzm on \
        --json bench/results/linux/coder-next/s$s-$p.json --cache-slots $s --io-threads 4 --stream-prefill \
        > bench/results/linux/coder-next/s$s-$p.log 2>&1
done; done
