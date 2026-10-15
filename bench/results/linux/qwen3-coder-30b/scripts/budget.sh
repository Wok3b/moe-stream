#!/bin/sh
# Qwen3-Coder-30B-A3B-Instruct Q4_K_M under memory budgets, cold model file every run.
# mem-budget.sh writes to bench/results/linux/<TAG>-*; the files are moved into qwen3-coder-30b/ afterwards.
cd "$HOME/moe-stream" || exit 1
export MODEL=$HOME/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
# prompts rendered with the model's own template
for p in story code; do
    [ -s bench/prompts/qc30-$p.txt ] && continue
    msg=$(sed -n '2p' bench/prompts/qwen36-$p.txt | sed 's/<|im_end|>$//')
    python3 bench/chat-prompt.py "$MODEL" "$msg" --out bench/prompts/qc30-$p.txt || exit 1
done
# hashes at 32 tokens, mmap included
PROMPT=bench/prompts/qc30-story.txt N_PREDICT=32 SLOTS=auto TAG=qc-hash bench/mem-budget.sh infinity 8G 6G
# 128 tokens: story with the mmap baseline at every budget (the model is only 1.2x RAM), code without
PROMPT=bench/prompts/qc30-story.txt N_PREDICT=128 SLOTS=auto TAG=qc-story bench/mem-budget.sh 6G 8G 10G 12G infinity
PROMPT=bench/prompts/qc30-code.txt N_PREDICT=128 SLOTS=auto MMAP=0 TAG=qc-code bench/mem-budget.sh 6G 8G 10G 12G infinity
echo BUDGET DONE
