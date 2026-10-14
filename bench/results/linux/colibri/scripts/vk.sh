#!/bin/bash
# Colibri Vulkan row (copy of the source, so the measured CPU engine stays as it was), then moe-stream start-up A/B.
B=$(cd "$(dirname "$0")" && pwd); S=$(dirname "$B"); V=$S/colibri-vk
rm -rf "$V"; mkdir -p "$V"
git -C "$HOME/colibri" archive HEAD | tar -x -C "$V"
make -C "$V/c" qwen36 VK=1 > "$B/results/build-vk.log" 2>&1 || { echo "VK BUILD FAILED"; exit 1; }
echo "VK BUILD OK"
[ -f "$HOME/models/qwen36-colibri-i4-gs64/.coli_usage" ] || cp "$V/c/profiles/qwen36-35b.coli_usage" "$HOME/models/qwen36-colibri-i4-gs64/.coli_usage"
export REPEATS=1
TAG=cap64 "$B/round.sh" coli infinity 1 --cap 64
TAG=cap64-vk COLI_BIN="$V/c/coli" COLI_ENV="COLI_VULKAN=1" "$B/round.sh" coli infinity 1 --cap 64
TAG=cap64-vk COLI_BIN="$V/c/coli" COLI_ENV="COLI_VULKAN=1" "$B/round.sh" coli infinity 2 --cap 64
echo "VK DONE"
for r in 1 2 3; do
    "$B/startup.sh" base-r$r
    "$B/startup.sh" nowarmup-r$r --no-warmup
done
echo "ALL DONE"
