# moe-stream

Making medium/large Mixture-of-Experts models usable on ordinary consumer
machines — no discrete GPU, 8–16 GB RAM, an NVMe SSD — by streaming expert
weights from disk instead of holding the whole model in memory.

This is the x86/Windows counterpart to what [Edge0](https://github.com/Edge0-AI/Edge0)
does on Apple Silicon. See [docs/NOTES.md](docs/NOTES.md) for the thesis,
prior art, and plan.

## Getting started

**[SETUP.md](SETUP.md)** — rebuilds a working tree from scratch: clone llama.cpp
at the pinned commit, apply `patches/`, build, fetch a model, smoke test.
Windows is fully supported; the Linux section lists exactly what still needs
porting.

## Reference machine

| | |
|---|---|
| Dell Vostro 15 3530 | Windows 11 Pro 24H2 |
| Intel Core i5-1334U | 2 P-cores + 8 E-cores, 12 threads, AVX2/AVX-VNNI, no AVX-512 |
| 16 GB DDR4-3200 dual channel | ~7–8 GB actually available to a model |
| Intel Iris Xe (shared memory) | no dGPU, no NPU |
| Samsung BM9C1 512 GB NVMe | |

## Layout

```
llama.cpp/      upstream clone (gitignored) — commit pinned in docs/NOTES.md
patches/        our changes to llama.cpp, as patch files
bench/          benchmark harness (PowerShell + Python)
bench/results/  raw llama-bench JSON + summaries
docs/           notes, measurements, design
C:\models\      GGUF weights (outside the repo)
```

## Status (2026-09-18)

1. **Baseline** — done. tok/s vs. resident fraction, disk bytes, working set;
   the NVMe does 1.2-1.5 GB/s for expert-sized reads but demand paging gets
   ~150-250 MB/s out of it. `docs/NOTES.md` Results 1, 3, 4.
2. **Spike** — done. A zero-training "stale router" predicts the next layers'
   experts with 85-98% recall (Results 2), but steering the page cache with
   `PrefetchVirtualMemory` yields 0-25%: the fault path and the OS cache
   policy are the bottleneck, not prediction (Results 5).
3. **Explicit expert cache** — done. `patches/0002-expert-cache-hook.patch`
   + `tools/moe-run/expert_cache.*`: experts live in locked slots filled by
   unbuffered reads on an I/O pool, addressed through a rewritten
   `ffn_moe_slot_ids` tensor. Bit-exact output; page faults gone; 1.4-1.56x
   over the page cache at equal RAM budget, now purely bandwidth-bound
   (Results 6). Next levers are bytes per token (K, quant) and the drive.

## Tools

```
bench/run-bench.ps1        llama-bench + disk/fault/working-set counters
bench/sweep-pressure.ps1   tok/s vs. available RAM (uses ram-eater.py)
bench/ab-prefetch.ps1      A/B llama-moe-run configs under pressure
bench/disk-bench.py        unbuffered NVMe read throughput by pattern
bench/analyze-trace.py     expert popularity / reuse / stale-router accuracy
bench/purge-standby.ps1    drop the file cache (UAC) for cold runs
tools/moe-trace/           dump router decisions per layer and token
tools/moe-run/             decode with predictive prefetch; --cache-slots enables the
                           explicit expert cache (expert_cache.cpp)
patches/                   changes to llama.cpp; install-tools.ps1 copies
                           tools/ into llama.cpp/examples and registers them
```
