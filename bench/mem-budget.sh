#!/bin/sh
# Results 11: run under a cgroup memory budget (no swap), cold page cache for the model each time.
# Usage: bench/mem-budget.sh [budgets...]   e.g. bench/mem-budget.sh 6G 8G 10G 12G infinity
# Needs a user systemd with the memory controller delegated (the Arch default); no root.
cd "$(dirname "$0")/.." || exit 1
B=llama.cpp/build/bin/llama-moe-run
M=${MODEL:-$HOME/models/gemma-4-26B-A4B-it-UD-IQ3_S.gguf}
N=${N_PREDICT:-128}
O=bench/results/linux
mkdir -p $O
[ $# -gt 0 ] || set -- 6G 8G 10G 12G infinity

run() { # budget name args...
    budget=$1; name=$2; shift 2
    tag=mem-$budget-$name
    python3 bench/evict-file.py "$M"
    t0=$(date +%s.%N)
    timeout 900 systemd-run --user --scope --quiet -p MemoryMax=$budget -p MemorySwapMax=0 -- \
        $B -m "$M" -f bench/prompts/gemma4-story.txt -n $N -t 6 -c 16384 -lzm on --json $O/$tag.json "$@" \
        > $O/$tag.log 2>&1
    rc=$?
    t1=$(date +%s.%N)
    awk -v b="$budget" -v n="$name" -v rc=$rc -v t0=$t0 -v t1=$t1 'BEGIN { printf "%s %s exit %d wall %.1fs\n", b, n, rc, t1 - t0 }' | tee -a $O/mem-budget.txt
    grep -E "main: (prefill|decode|cache:|stream:|tokens hash)" $O/$tag.log | sed 's/^.*main: /    /' | tee -a $O/mem-budget.txt
}

for b in "$@"; do
    run $b mmap
    run $b s32-stream --cache-slots 32 --io-threads 4 --stream-prefill
    run $b s64-stream --cache-slots 64 --io-threads 4 --stream-prefill
done
