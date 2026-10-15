#!/bin/sh
# Results 20: Qwen3.5-122B-A10B UD-IQ3_XXS under memory budgets, cold model file every run.
# mem-budget.sh writes to bench/results/linux/<TAG>-*; the files were moved into q122/ without the "q122-" prefix.
cd "$HOME/moe-stream" || exit 1
export MODEL=$HOME/models/Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf PROMPT=bench/prompts/qwen36-story.txt
# hashes at 32 tokens, mmap included (mmap at 6-12 GB takes 4+ min per run)
N_PREDICT=32 SLOTS=auto TAG=q122-hash bench/mem-budget.sh infinity 8G
# 128 tokens, two rounds (8 GB with stream prefill does not start: the buffers leave too few slots)
N_PREDICT=128 SLOTS=auto MMAP=0 TAG=q122 bench/mem-budget.sh 8G 10G 12G infinity
N_PREDICT=128 SLOTS=auto MMAP=0 TAG=q122-r2 bench/mem-budget.sh infinity 10G 12G
# 8 GB without stream prefill
O=bench/results/linux; tag=q122-8G-sauto-nostream
python3 bench/evict-file.py "$MODEL"
timeout 900 systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- \
    llama.cpp/build/bin/llama-moe-run -m "$MODEL" -f "$PROMPT" -n 128 -t 6 -c 16384 -lzm on --json $O/$tag.json \
    --cache-slots auto --io-threads 4 > $O/$tag.log 2>&1
echo "8G sauto-nostream exit $?" | tee -a $O/q122-budget.txt
grep -E "main: (prefill|decode|cache:|stream:|tokens hash)" $O/$tag.log | sed 's/^.*main: /    /' | tee -a $O/q122-budget.txt
