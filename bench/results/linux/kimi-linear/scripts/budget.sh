#!/bin/sh
# Kimi-Linear-48B-A3B-Instruct Q4_K_M under memory budgets, cold model file every run.
# mem-budget.sh writes to bench/results/linux/<TAG>-*; the files are moved into kimi-linear/ afterwards.
cd "$HOME/moe-stream" || exit 1
export MODEL=$HOME/models/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf
# prompts rendered with the model's own template
for p in story code; do
    [ -s bench/prompts/kimi-linear-$p.txt ] && continue
    msg=$(sed -n '2p' bench/prompts/qwen36-$p.txt | sed 's/<|im_end|>$//')
    python3 bench/chat-prompt.py "$MODEL" "$msg" --out bench/prompts/kimi-linear-$p.txt || exit 1
done
# hashes at 32 tokens, mmap included
PROMPT=bench/prompts/kimi-linear-story.txt N_PREDICT=32 SLOTS=auto TAG=kl-hash bench/mem-budget.sh infinity 8G
# 128 tokens, story and code
for p in story code; do
    PROMPT=bench/prompts/kimi-linear-$p.txt N_PREDICT=128 SLOTS=auto MMAP=0 TAG=kl-$p bench/mem-budget.sh 6G 8G 10G 12G infinity
done
echo BUDGET DONE
