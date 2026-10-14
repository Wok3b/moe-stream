#!/bin/bash
# The rest of round 2 (12G, infinity; Colibri first as in round 2), then the Colibri Vulkan row at no limit.
B=$(cd "$(dirname "$0")" && pwd); S=$(dirname "$B")
export REPEATS=1
"$B/round.sh" coli 12G 2 --cap 109
"$B/round.sh" moe 12G 2
"$B/round.sh" coli infinity 2 --cap 0
"$B/round.sh" moe infinity 2
echo "CPU ROUNDS DONE"
# Vulkan build in a copy, so the CPU engine above stays the one measured
rsync -a --delete --exclude .git "$HOME/colibri/" "$S/colibri-vk/"
make -C "$S/colibri-vk/c" qwen36 VK=1 > "$B/results/build-vk.log" 2>&1 || { echo "VK BUILD FAILED"; exit 1; }
# the starting expert history coli setup copies in for this model (read only while the tier is on)
[ -f "$HOME/models/qwen36-colibri-i4-gs64/.coli_usage" ] || cp "$S/colibri-vk/c/profiles/qwen36-35b.coli_usage" "$HOME/models/qwen36-colibri-i4-gs64/.coli_usage"
# control: the same cap on the CPU; then the iGPU twice (fresh history, then trained by the first run)
TAG=cap64 "$B/round.sh" coli infinity 1 --cap 64
TAG=cap64-vk COLI_BIN="$S/colibri-vk/c/coli" COLI_ENV="COLI_VULKAN=1" "$B/round.sh" coli infinity 1 --cap 64
TAG=cap64-vk COLI_BIN="$S/colibri-vk/c/coli" COLI_ENV="COLI_VULKAN=1" "$B/round.sh" coli infinity 2 --cap 64
echo "ALL DONE"
