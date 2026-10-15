#!/bin/sh
# North-Mini-Code-1.0 UD-Q4_K_M under memory budgets, cold model file every run.
# mem-budget.sh writes to bench/results/linux/<TAG>-*; the files are moved into north-mini-code/ afterwards.
cd "$HOME/moe-stream" || exit 1
export MODEL=$HOME/models/North-Mini-Code-1.0-UD-Q4_K_M.gguf
# prompts rendered with the model's own template
for p in story code; do
    [ -s bench/prompts/north-$p.txt ] && continue
    msg=$(sed -n '2p' bench/prompts/qwen36-$p.txt | sed 's/<|im_end|>$//')
    python3 bench/chat-prompt.py "$MODEL" "$msg" --out bench/prompts/north-$p.txt || exit 1
done
# hashes at 32 tokens, mmap included
PROMPT=bench/prompts/north-story.txt N_PREDICT=32 SLOTS=auto TAG=nm-hash bench/mem-budget.sh infinity 8G 6G
# 128 tokens, cache only (the mmap baseline above is enough), story and code
for p in story code; do
    PROMPT=bench/prompts/north-$p.txt N_PREDICT=128 SLOTS=auto MMAP=0 TAG=nm-$p bench/mem-budget.sh 6G 8G 10G 12G infinity
done
echo BUDGET DONE
