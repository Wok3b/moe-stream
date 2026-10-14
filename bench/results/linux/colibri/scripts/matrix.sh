#!/bin/bash
# Main comparison: 5 budgets x 2 rounds, engines alternating first (ABBA), one pass of the workload each.
B=$(cd "$(dirname "$0")" && pwd)
cap() { case $1 in 6G) echo 20;; 8G) echo 50;; 10G) echo 80;; 12G) echo 109;; infinity) echo 0;; esac; }
export REPEATS=1
for r in 1 2; do
    for b in 6G 8G 10G 12G infinity; do
        if [ $((r % 2)) = 1 ]; then order="moe coli"; else order="coli moe"; fi
        for e in $order; do
            if [ $e = moe ]; then "$B/round.sh" moe $b $r
            else "$B/round.sh" coli $b $r --cap "$(cap $b)"; fi
        done
    done
done
echo MATRIX DONE
