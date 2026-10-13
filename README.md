# moe-stream

Running Mixture-of-Experts models whose weights do not fit in RAM, on an
ordinary laptop: no discrete GPU, 16 GB of memory, an NVMe SSD. The experts stay
on disk and are streamed into an explicit, locked cache instead of being
demand-paged by the operating system.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/dark/fig0-headline.png">
  <img alt="Three measured comparisons: 10.1x more read throughput, 1.6x decode throughput from the explicit expert cache, 1.5x more from the bundled layout" src="docs/img/fig0-headline.png">
</picture>

**Read the write-up: [Why a 16 GB laptop can't run a big MoE model from SSD, and
what I built to fix it](docs/streaming-moe-on-a-laptop.md).** It has the whole
story with charts; this page is the short version.

Everything below was measured on one laptop (Core i5-1334U, 16 GB DDR4-3200,
Samsung BM9C1 NVMe, CPU only) with OLMoE-1B-7B-0924-Instruct Q4_K_M: 16 layers,
64 experts, top-8, 3.92 GiB on disk.

## The problem: the page cache reads at a tenth of the drive's speed

Memory-map the model and let the OS handle it, and the experts arrive one 4 KB
fault at a time, 25 000 to 50 000 faults per generated token. That path moved
**148 MB/s** on a drive that does **1496 MB/s** for expert-sized random reads.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/dark/fig3-disk-patterns.png">
  <img alt="Read throughput by request size, with the rate demand paging actually achieved marked" src="docs/img/fig3-disk-patterns.png">
</picture>

The drive is not slow. The requests are small, and nothing tells the kernel what
to read next, so it can only react to a fault that already happened. Predicting
the experts and calling `PrefetchVirtualMemory` does not fix it either: that
bought between 0 and 25%, because the prefetch call blocks the compute thread,
cannot be parallelised, and the pages get evicted before use.

## What works: owning the I/O path

Experts live in locked slots in our own memory, filled by unbuffered reads on an
I/O thread pool, addressed through a rewritten `ffn_moe_slot_ids` tensor so
`mul_mat_id` indexes the cache instead of the model. The page cache is out of
the picture entirely.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/dark/fig6-cache-ab.png">
  <img alt="Decode throughput and page faults per token, page cache against explicit cache" src="docs/img/fig6-cache-ab.png">
</picture>

**1.42x to 1.56x at an equal RAM budget**, and page faults per token fall from
25 000 - 50 000 to around 850.

## Then give each expert one contiguous home on disk

A GGUF stores the experts as three stacked tensors, so fetching one expert is
three reads from three places in the file. `llama-moe-bundle` writes a sidecar
in which each expert's matrices sit back to back, and the cache reads whole
experts from it. The GGUF is untouched and still loads normally.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/dark/fig8-bundled-layout.png">
  <img alt="Decode throughput, read requests per token and per-thread read throughput, split against bundled" src="docs/img/fig8-bundled-layout.png">
</picture>

**1.52x to 1.61x more**, with bytes per token identical on both sides. Requests
per token drop by exactly 3x and the drive gives 1040 MB/s per I/O thread
instead of 649. Output is bit-exact: the generated token hash is identical on
the mmap path and on every cache and layout configuration.

## Three more things the measurements settled

| | |
|---|---|
| **Prediction is free** | Applying layer L+1's router to the residual entering layer L's FFN picks the exact top-8 **84%** of the time, and covers **97%** of the real choice if you fetch 16 experts. No training, no extra parameters. |
| **There is no hot set** | Covering 80% of expert selections needs **38 of the 64** experts, so cache policy is a small lever on a load-balanced model. |
| **K=4 is not free** | Halving the active experts halves the bytes but costs **+41%** perplexity on wikitext-2 and **+87%** on technical prose. K=6 costs 9 to 12% for 25% fewer bytes and is a real knob. |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/dark/fig4-stale-router.png">
    <img src="docs/img/fig4-stale-router.png" width="48%" alt="Stale-router accuracy against lookahead distance">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/dark/fig7-ppl-vs-k.png">
    <img src="docs/img/fig7-ppl-vs-k.png" width="48%" alt="Perplexity against the number of active experts">
  </picture>
</p>

## How to reproduce

[SETUP.md](SETUP.md) has the details. On Windows with Visual Studio and Python:

```powershell
git clone --recursive https://github.com/Wok3b/moe-stream; cd moe-stream
.\build.ps1
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/OLMoE-1B-7B-0924-Instruct-GGUF', filename='OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf', local_dir=r'C:\models')"
```

On Linux (C++ compiler, CMake, git): `git clone --recursive https://github.com/Wok3b/moe-stream && cd moe-stream && ./build.sh`.

Build the bundled sidecar and run the comparison:

```powershell
.\llama.cpp\build\bin\Release\llama-moe-bundle.exe -m C:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf --verify
.\bench\ab-layout.ps1 -Model C:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf -Slots 32 -Rounds 3
```

The rest of the measurements:

```powershell
python bench\disk-bench.py C:\models\<model>.gguf --json bench\results\disk-bench.json
.\bench\run-bench.ps1      -Model C:\models\<model>.gguf -Threads "4,6,8,10,12"
.\bench\sweep-pressure.ps1 -Model C:\models\<model>.gguf -AvailableGB 6,3,2,1.5 -Threads 10
.\bench\ab-prefetch.ps1    -Model C:\models\<model>.gguf -AvailableGB 2.5 -Configs "0 8","2 8"
.\bench\ppl-k-sweep.ps1    -Model C:\models\<model>.gguf -Arch olmoe -Text C:\models\eval\wikitext-2-raw\wiki.test.raw -K 8,6,4
python bench\make-charts.py
```

`sweep-pressure` and `ab-prefetch` pin RAM with `bench/ram-eater.py` to simulate
a smaller machine, so the box feels sluggish while they run. Every figure above
is regenerated by `bench/make-charts.py` from the files in `bench/results/`, so
the charts and the raw data cannot drift apart.

**The correctness check that matters:** the `tokens hash` printed by
`llama-moe-run` must be identical with and without `--cache-slots` and with and
without `--experts-file`, for the same prompt and greedy sampling. If it is not,
the cache is serving the wrong weights.

## Layout

```
build.sh         build everything (build.ps1 on Windows)
patches/         our llama.cpp commits as patch files, for reading
tools/moe-run/   decode with the explicit expert cache and the predictive prefetcher,
                 plus llama-moe-bundle, the expert-bundled layout converter
tools/moe-trace/ dump router decisions per layer and token
bench/           measurement harness (PowerShell + Python)
bench/results/   raw llama-bench / moe-run JSON and the CSV summaries
docs/            the write-up, and NOTES.md: every result in the order measured
llama.cpp/       submodule: Wok3b/llama.cpp, branch moe-stream (upstream 972d231 + our commits)
C:\models\       GGUF weights and the .experts sidecars (outside the repo)
```

## Status

Done and measured: baseline curves, the routing trace and the stale-router
predictor, the failed page-cache prefetch spike, the explicit expert cache, the
K sweep, and the expert-bundled layout. Results 1 to 8 in
[docs/NOTES.md](docs/NOTES.md) have the numbers in the order they were taken,
including what did not work and why.

**Gemma 4 26B-A4B and llama-server** (Results 9 and 10, 2026-10-13, a second
laptop): the cache runs bit-exact on Gemma 4 (patch 0003, per-expert scales),
inside `llama-server` (0004), and prompt processing now streams whole layers
through two buffers instead of paging every expert in (0005). Gemma 4 does have
a hot set: 64 of 128 slots serve 96% of decode lookups. In a real grading
workload the moe-stream server ties mmap on wall time while leaving ~0.9 GB of
RAM free instead of tens of MB. The stale-router prefetcher is a net loss on
this model (66-70% prediction accuracy).

Open, in the order it matters:

1. **Done: a model that does not fit.** Qwen3.6-35B-A3B (20.6 GiB) on a 16 GB
   laptop decodes at 6.4 tok/s in an 8 GB cgroup and 7.3 unlimited, against
   mmap's 0.8-3.5 (docs/NOTES.md, Results 12-17).
2. **Asynchronous I/O.** The reader is synchronous per thread, so queue depth
   equals thread count; the drive wants 2 to 4 from a single issuer.
3. **Done: Linux.** Ported (O_DIRECT, mlock, madvise); cgroup memory limits
   replace pinned pages for the measurements (Results 11).
4. **Upstream.** The expert-cache hook is small and architecture-agnostic, and
   upstream llama.cpp is already moving this way with `--load-mode dio` and lazy
   tensors.

## Prior art

This is the x86/Windows counterpart to what [Edge0](https://github.com/Edge0-AI/Edge0)
does on Apple Silicon with MLX. The ideas it builds on are credited in
[docs/NOTES.md](docs/NOTES.md): Pre-gated MoE, LLM in a flash, PowerInfer-2,
MoE-Infinity, Fiddler, ExpertFlow, HOBBIT, and llama.cpp itself, which is the
baseline every number here is measured against.
