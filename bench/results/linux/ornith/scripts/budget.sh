#!/bin/sh
# Ornith-1.5-35B-A3B Q4_K_M under memory budgets, cold model file every run: one mmap check for the hash,
# then the cache only. mem-budget.sh writes to bench/results/linux/<TAG>-*; the files are moved into ornith/ afterwards.
cd "$HOME/moe-stream" || exit 1
export MODEL=$HOME/models/Ornith-1.5-35B-Q4_K_M.gguf
for p in story code; do
    [ -s bench/prompts/ornith-$p.txt ] && continue
    msg=$(sed -n '2p' bench/prompts/qwen36-$p.txt | sed 's/<|im_end|>$//')
    python3 bench/chat-prompt.py "$MODEL" "$msg" --out bench/prompts/ornith-$p.txt || exit 1
done
PROMPT=bench/prompts/ornith-story.txt N_PREDICT=32 SLOTS=auto TAG=or-hash bench/mem-budget.sh infinity
for p in story code; do
    PROMPT=bench/prompts/ornith-$p.txt N_PREDICT=128 SLOTS=auto MMAP=0 TAG=or-$p bench/mem-budget.sh 6G 8G 10G 12G infinity
done
cd bench/results/linux && for f in or-*; do mv "$f" "ornith/${f#or-}"; done
echo BUDGET DONE
