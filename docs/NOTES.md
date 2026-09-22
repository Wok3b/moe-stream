# Notes

## What Edge0 actually is

Three techniques, published Apache-2.0, MLX/Apple-Silicon only
(https://github.com/Edge0-AI/Edge0, docs/architecture.md):

1. **SSD expert offload** — MoE expert weights are mmapped and read on demand;
   attention/norms/embeddings/router stay resident. Peak RAM ~2.9 GB for the
   35B model, ~1 GB for the 8B.
2. **Prerouter** — a small trained network that predicts which experts the
   *next* layer will select, so their weights can be fetched before the real
   router runs. (Idea: "Pre-gated MoE", Hwang et al. 2023.)
3. **Recover-LoRA** — aggressive int4 quantisation, then LoRA adapters
   distilled to recover the lost quality. Adapters stay unmerged.

Models: edge0-35b (Qwen3.6-35B-A3B, 40 layers, 256 experts, K=4, ~23 GB int4)
and edge0-8b (Ling 3.0 tiny hybrid, 24 layers, 128 experts, K=8, ~4.5 GB).

(1) is systems code. (2) and (3) are *trained artifacts* — they need a GPU to
produce for any new model. The runtime must work without them (fallback
heuristics, standard quants) and use them when available.

## Prior art on x86

- **llama.cpp** mmaps GGUF; for MoE only touched experts page in. The OS page
  cache is an LRU that can only react to misses. Windows/CPU/Vulkan/SYCL/CUDA.
  *This is the baseline to beat.*
- **mistral.rs**, **candle** — Rust inference; MoE, GGUF, Windows.
- **ktransformers** — CPU/GPU hybrid expert offload for DeepSeek-class MoE.
- **PowerInfer-2** (SJTU 2024) — neuron-level sparsity streamed from phone flash.
- **"LLM in a flash"** (Apple 2023) — windowing + row-column bundling for
  flash-resident weights.
- **Pre-gated MoE** (2023), **MoE-Infinity**, **Fiddler**, **ExpertFlow**,
  **HOBBIT** — expert prefetch / caching / offload papers.

## Thesis

The page cache cannot prefetch layer L+1's experts while layer L computes,
because nothing has told it which experts will be selected. A predictor can.
If prediction is good enough and NVMe bandwidth is the bottleneck, explicit
prefetch + an explicit expert cache should beat plain mmap under memory
pressure. It might not beat it by much — measure first.

Things a new runtime could own that llama.cpp does not:
- **Expert-bundled layout.** GGUF stores `ffn_up_exps`, `ffn_gate_exps`,
  `ffn_down_exps` as three stacked tensors; one expert = three scattered reads.
  Bundling an expert's three matrices contiguously = one ~2–3 MB read at near
  sequential NVMe speed.
- **Explicit expert cache** with a RAM budget, LRU + pinned hot experts
  (expert popularity is heavily skewed), instead of trusting the page cache.
- **Prefetch scheduler** driven by a prerouter, with a cheap fallback
  (recent-history / previous-layer experts) so it works on any MoE.

## Back-of-envelope for the reference machine

Qwen3-30B-A3B-class at ~4.5 bpw: 48 layers x 8 active experts x ~2.6 MB
= ~1 GB of expert bytes per token if *everything* misses. At ~3 GB/s NVMe that
is a ~3 tok/s floor with zero caching. With a 6–8 GB expert cache and realistic
hit rates the I/O ceiling is 10–15 tok/s; CPU compute on 3B active params on
an i5-1334U is ~8–12 tok/s. Target: **~6–10 tok/s on a 30B-class MoE with 16 GB
and no GPU.**

## Measurement plan (phase 1)

Subjects:
- `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` (4.2 GB; 16 layers, 64 experts,
  top-8). Fits in RAM — control subject. Memory pressure is simulated with
  `bench/ram-eater.py` so we can sweep "fraction of model resident".
- `Qwen3.6-35B-A3B` (Edge0's base; ~22 GB Q4_K_M) — the real target, later.

Curves to produce with plain llama.cpp + mmap:
- tok/s (tg128) vs. threads — pick the best thread count for this hybrid CPU.
- tok/s vs. resident fraction (100%, 75%, 50%, 25%) via ram-eater.
- disk bytes read per generated token at each pressure level.

Cold vs warm: `bench/purge-standby.ps1` empties the standby list (needs UAC).

## What upstream llama.cpp already has (commit 972d231, 2026-09-17)

- `--load-mode auto|none|mmap|mlock|mmap+mlock|dio` (`-lm`). `dio` = direct I/O.
- `--lazy-mode auto|on|off` (`-lzm`) + a `TENSOR_READ_LAZY` flag in
  `src/llama-model-loader.h`. A lazy tensor stays in the mmap, is excluded from
  the load-time `PrefetchVirtualMemory` / `MADV_WILLNEED`, gets `MADV_RANDOM`
  on POSIX, is forced onto the host CPU buffer regardless of offload settings,
  and its rows fault in on demand. Today only the giant per-layer-embedding
  tables use it (`src/models/gemma4.cpp`, `src/models/qwen4exp.cpp`); the MoE
  expert tensors `ffn_{up,gate,down}_exps` do **not**. Still page-cache driven:
  no explicit cache, no prediction.
- `cb_eval` (`ggml_backend_sched_eval_callback`) — a per-node callback during
  graph execution (used by `llama-eval-callback`, imatrix). Node names carry the
  layer index (e.g. `ffn_moe_topk-12`). This is our hook point.
- Each expert's matrix is a contiguous slice of its stacked tensor
  (`[n_embd, n_ff, n_expert]`, expert outermost) -> one expert = three
  contiguous byte ranges (up, gate, down). Prefetchable by range.

## Spike plan (refined)

0. **Mark experts lazy.** Apply `TENSOR_READ_LAZY` to `ffn_*_exps` (a small
   patch). Stops the loader from trying to prefetch 22 GB into 8 GB at start.
1. **Instrumentation only, no I/O changes.** Via `cb_eval`, log per token and
   per layer: the router's top-k expert ids, and the hidden state entering the
   FFN. Offline (Python) compute:
   - expert popularity skew per layer (how concentrated is the working set?)
   - temporal reuse: P(expert used at layer L for token t | used at t-1)
   - **stale-router prediction accuracy**: apply layer L+1's router weights to
     the hidden state *entering* layer L's FFN (the residual stream changes
     little across one FFN) and measure top-k overlap with the real choice at
     L+1. Also L+2, L+3. This is a zero-training prerouter. If overlap is high,
     prefetch has headroom; if not, we need Edge0-style trained predictors.
2. **Baseline curves** with plain mmap: tok/s and disk MB/token vs. available
   RAM (`bench/sweep-pressure.ps1`), lazy on/off, load-mode mmap vs dio.
3. **Prefetch.** In the `cb_eval` hook after the FFN input of layer L is ready:
   run the stale router for L+1..L+k on it, `PrefetchVirtualMemory` the predicted
   experts' three byte ranges. Measure against (2).

Bound on the upside: prefetch hides latency, not bandwidth. Per token speedup
<= (compute + io) / max(compute, io). At an 80% cache hit rate on a 30B-A3B
(io ~66 ms, compute ~100 ms on this CPU) that is at most ~1.7x; at 0% hits it
is ~1.3x. The cache hit rate (RAM budget x expert-popularity skew x policy)
dominates. Prefetch is the second-order win; cache policy and layout are the
first-order ones.

## Results 1 — OLMoE-1B-7B Q4_K_M, warm, CPU only (2026-09-18)

`llama-bench`, pp512 / tg128, 3 reps, model fully resident:

| threads | pp512 tok/s | tg128 tok/s |
|---|---|---|
| 4 | 39.4 | 14.9 |
| 6 | 44.8 | 20.1 |
| 8 | 49.4 | 20.1 |
| 10 | **55.3** | **22.4** |
| 12 | 56.9 | 20.4 |

Generation saturates at ~20-22 tok/s from 6 threads: memory-bandwidth bound
(~1.3B active params at Q4 ~ 0.8 GB/token -> ~18 GB/s effective on DDR4-3200).
Prompt processing scales with threads. => during decode ~4 cores are idle:
that is free CPU for a prefetch scheduler.

## Results 2 — routing trace, OLMoE, 256 sampled decode tokens (temp 0.7)

`llama-moe-trace` + `bench/analyze-trace.py` (`bench/results/trace-olmoe-256.*`).

**Expert popularity is nearly flat.** Covering 80% of selections needs 38 of
64 experts on average; 95% needs 52; all 64 are used within 256 tokens. OLMoE
is trained with load balancing and it shows. Consequence: a cache holding a
fraction f of the experts hits ~f of the time. No small "hot set" to pin.

**Temporal reuse is low.** Consecutive tokens share only 31% of their experts
at the same layer (range 23-40%). "Same as last token" is a weak predictor.

**Stale-router prediction is strong.** Apply layer L+d's norm + router to the
residual entering layer L's FFN, compare with what L+d really picked:

| lookahead d | exact top-8 hit | recall if 16 prefetched | recall if 32 prefetched |
|---|---|---|---|
| 1 | 0.84 | 0.97 | 0.99 |
| 2 | 0.79 | 0.94 | 0.99 |
| 3 | 0.74 | 0.91 | 0.98 |
| 4 | 0.69 | 0.87 | 0.97 |

A zero-training predictor gives a 2-3 layer lookahead window with 90%+ recall.
Edge0's trained prerouter is not required to get most of the benefit.
Fetching 2k instead of k buys recall with 2x bytes — right call when latency
bound, wrong call when bandwidth bound; make it adaptive (fetch top-k plus any
expert within a score margin of the k-th).

**What this means.** With flat popularity, hit rate ~ resident fraction, so a
model 3x bigger than RAM reads ~2/3 of its per-token expert bytes from NVMe
no matter how clever the cache. The levers, in order:
1. bytes per expert (quantisation; Edge0's Recover-LoRA exists to make lower
   bits acceptable),
2. NVMe bandwidth actually achieved (layout: one contiguous read per expert),
3. hiding latency under compute (predictive prefetch, now shown feasible),
4. cache policy (small effect when popularity is flat).

## Results 3 — memory-pressure curve, OLMoE Q4_K_M (3.9 GB), t=10, mmap

`bench/sweep-pressure.ps1`, pp256 / tg64, 2 reps (`bench/results/summary-v1.csv`):

| available RAM target | pp256 tok/s | tg64 tok/s | disk read during run | run time |
|---|---|---|---|---|
| 6 GB (model fits) | 50.9 | 18.6 | 1.1 GB | 30 s |
| 3 GB | 49.5 | 19.1 | 6.9 GB | 62 s |
| 2 GB | 21.5 | 8.5 | 16.2 GB | 111 s |
| 1.5 GB | 15.4 | 3.65 | 23.4 GB | 163 s |

Caveats: Windows answers pressure by paging *other* processes to the pagefile
first (available rose 3.0 -> 5.2 GB during the 3 GB run), so "available" is a
loose proxy for how much of the model was resident. The wrapper now samples
the benchmark's working set, which measures it directly.

Effective disk throughput of the page-fault path: each pp256 pass at 1.5 GB
must touch >= 1.9 GB of non-resident experts and took 11.6 s longer than warm
-> **~150 MB/s**, on a drive that does 1.2-1.5 GB/s for expert-sized reads.

## Results 4 — what the NVMe can do (`bench/disk-bench.py`, unbuffered)

Samsung BM9C1 512 GB (DRAM-less PCIe 4 OEM drive), 1 GB per test:

| pattern | MB/s | ms/request |
|---|---|---|
| sequential 8 MB QD1 | 1561 | 5.4 |
| random 2 MB QD1 | 1194 | 1.8 |
| random 2 MB QD4 | 1461 | 5.7 |
| random 2 MB QD8 | 1039 | 16.1 |
| random 256 KB QD1 / QD8 | 363 / 770 | 0.7 / 2.7 |
| random 64 KB QD1 / QD8 | 241 / 441 | 0.3 / 1.2 |
| random 4 KB QD1 / QD8 | 24 / 150 | 0.17 / 0.22 |

**The headline.** Expert-sized random reads run at ~80-95% of sequential
bandwidth, QD 2-4 is enough. Demand paging of the mmap runs at ~150 MB/s
(consistent with 4-64 KB clustered faults at low queue depth). That is a
**~10x gap** on this laptop, and the ratio is worse on faster desktop NVMe
drives, whose small-random-read weakness is the same. This is a much larger
lever than latency hiding: it means the I/O floor for a model 3x bigger than
RAM is not "bytes / 150 MB/s" but "bytes / ~1.3 GB/s".

Revised per-token estimate for Qwen3.6-35B-A3B (8 of 256 experts, 1.8 MB each
at Q4, 40 layers => 570 MB/token all-miss; ~36% resident => ~365 MB/token):
- page cache as-is: 365 MB / 0.15 GB/s = 2.4 s/token  (**~0.4 tok/s**)
- expert-sized prefetch: 365 MB / 1.3 GB/s = 0.28 s/token, overlapped with
  ~0.1 s compute  (**~3.5 tok/s**), ~9x
- plus K=4 routing like Edge0: ~180 MB/token -> ~0.14 s  (**~6-7 tok/s**)
- plus 3-bit experts: another ~25%.
So "usable" is plausible on this laptop but it takes all three levers; the
first one (big reads) is the one nobody gets for free from the OS.

Lowest-effort path to capture it inside llama.cpp: predict experts via the
stale router (Results 2), then `PrefetchVirtualMemory` the predicted experts'
three byte ranges; the kernel issues large efficient reads for the ranges.
No custom I/O path needed for the spike.

## Results 5 — the spike: predictive prefetch inside llama.cpp (2026-09-18)

Built: `patches/0001-lazy-experts.patch` (with `-lzm on`, `*_exps.weight`
tensors are lazy: no load-time prefetch) and `tools/moe-run` (stale-router
prediction in a `cb_eval` hook at `ffn_inp-L`, `PrefetchVirtualMemory` of the
predicted experts' three ranges for layers L+1..L+depth, on a worker thread,
skipping experts whose pages are already in the working set via
`QueryWorkingSetEx`). Driver: `bench/ab-prefetch.ps1`; rows in
`bench/results/ab.csv`. OLMoE Q4_K_M, t=10, 48 greedy tokens, same prompt.

In-process prediction matches the offline trace exactly (0.847 exact top-8 at
depth 1; 0.878 union of depths 1-2; 0.98 at width 16).

What happened, in order:

1. Synchronous prefetch (v1): faults/token fell 23k -> 12k at depth 3, but
   `PrefetchVirtualMemory` blocked the compute thread for 170-330 ms/token —
   it walks every page of every range, ~0.16 ms/MB even when resident. Net:
   2.4 -> 3.0 tok/s at best.
2. Worker thread + residency filter (v2/v3): 2.4-2.6 -> 4.5-4.8 tok/s at
   "2 GB available"... but sequential runs drift: Windows keeps paging other
   processes out, so each later run has more effective cache.
3. More threads / touching pages to pre-map them (v3): both *worse*
   (3.4 and 2.7 tok/s; soft faults 39-47k/token). The compute threads' faults,
   the prefetch call and the toucher all contend for the process's memory-
   manager locks, and growing the working set makes the OS trim harder.
4. Interleaved A/B/A/B (v4), the honest numbers:

| available | base | prefetch d2 | base | prefetch d2 |
|---|---|---|---|---|
| 2 GB | 2.01 | 2.98 | 3.06 | 3.12 |
| 1.5 GB | 1.37 | 1.28 | 1.14 | 1.41 |

**~0-25% at moderate pressure, nothing at severe pressure.** Width 16 is a
clear loss (2x bytes in an I/O-bound regime). Depth beyond 2 does not help.

Why (all measured):
- `PrefetchVirtualMemory` moves ~0.6 GB/s, single-threaded, and cannot be
  parallelised (lock contention with the faulting compute threads).
- Under pressure the compute threads take ~48k hard faults/token = one per
  4 KB page, no clustering, each a synchronous I/O wait. That is the ~150 MB/s
  path from Results 3/4, and prefetch does not change how those pages arrive.
- Prefetched pages are evicted before use when memory is scarce: the OS cache
  policy is not ours to set.

**Conclusion of the spike.** Prediction works (88-98% recall, zero training).
The page cache is the wrong engine: it cannot be steered into large parallel
reads, it charges a fault per page, and it evicts what we prefetch. The
measured 5-8x gap between demand paging (150-250 MB/s) and explicit
expert-sized reads (1.2-1.5 GB/s, Results 4) can only be collected by owning
the I/O path: an expert cache in *our* memory (not file-backed), filled by
unbuffered overlapped `ReadFile` at 2 MB granularity, QD 2-4, and a
`mul_mat_id` that gathers experts through a slot table instead of a fixed
stride. That is the runtime design, and it is now justified by measurement.

Expected on this laptop once the I/O is owned (OLMoE, 50% resident):
~220 MB/token / 1.2 GB/s = 180 ms, overlapped with ~50 ms compute -> ~5 tok/s
vs 1.3 today (4x); at 80% resident ~10 tok/s vs 3 (3x). For Qwen3.6-35B-A3B
at 36% resident: 365 MB/token -> ~0.3 s -> ~3 tok/s at K=8, ~6 tok/s at K=4,
vs well under 1 tok/s with the page cache.

## Next step options

A. **Explicit expert cache inside llama.cpp** (C++, ~1-2 days): per-layer
   cache tensor `[n_embd, n_ff, C]` in anonymous memory + an I32 slot table;
   `build_moe_ffn` gains `slot_ids = get_rows(slot_table, selected_experts)`
   and runs `mul_mat_id` on the cache tensor; the `ffn_moe_topk-L` hook
   resolves misses (wait for in-flight prefetch, else synchronous 2 MB reads)
   and updates the slot table before the graph continues; prefetch as today
   but into slots via overlapped `ReadFile`. Gives the definitive number and
   is the reference implementation for the Rust runtime.
B. Skip to the Rust runtime with the design above.
C. First download Qwen3.6-35B-A3B (22 GB) and reproduce the baseline curve at
   real scale (K=8 vs K=4 quality check with llama-perplexity).

## Results 6 — explicit expert cache inside llama.cpp (2026-09-18)

Built (option A):
- `patches/0002-expert-cache-hook.patch` — `llama_expert_cache_register(orig, cache, max_tokens)`
  + a registry in `llama-graph.cpp`. `build_moe_ffn` looks the layer's expert
  tensors up; when cache tensors are registered and the batch is small enough,
  it emits `ffn_moe_slot_ids-<il>` (a `ggml_dup` of the selected expert ids)
  and runs the three `mul_mat_id`s on the cache tensors addressed by it.
  Biases/LoRA keep the real ids; prefill (n_tokens > max_tokens) uses the mmap.
- `tools/moe-run/expert_cache.{h,cpp}` — per-layer slot cache in anonymous,
  `VirtualLock`ed memory (a ggml CPU buffer of `[ne0, ne1, C]` tensors of the
  original quant type), LRU eviction with the current token's slots pinned,
  an I/O thread pool doing unbuffered `ReadFile` of each expert's three chunks
  (offsets from the GGUF metadata via `gguf.h`), misses queued ahead of
  prefetches. The `ffn_moe_slot_ids` hook resolves ids -> slots (loading
  misses, waiting for in-flight prefetches) and rewrites the tensor in place.
  Prefetch = same stale-router prediction as before, now landing in slots.
- `--cache-slots C --io-threads N [--cache-max-tokens T] [--no-lock]`,
  `--io-bench N` (in-process raw read test).

**Correctness:** greedy token hash identical on the mmap path and on every
cache configuration (`2247ad9cb26432b9` for the reference prompt).

**Bugs found on the way:** the per-layer prediction set was only cleared by
the stats hook, so without `--stats` nothing was prefetched after token 1
(now keyed by token); misses were queued behind prefetches (now front).
Threads/touching in the page-cache path made things worse (MM lock
contention); the `max` macro from windows.h.

**The drive throttles.** The Samsung BM9C1 (DRAM-less, laptop) sustains
1.2-1.4 GB/s for 2 MB reads when cool and drops to 600-750 MB/s after a few
minutes of sustained random reads (measured repeatedly in both Python and
C++: same code, same file, 2x swings within minutes). Sustained ~650 MB/s is
the realistic ceiling here; every A/B below is interleaved so both sides see
the same drive state.

**Equal-RAM-budget A/B** (`bench/ab-prefetch.ps1`, `-lm mmap+mlock -lzm on`,
48 greedy tokens, ram-eater leaving X GB available before each run; the
explicit cache locks its slots inside that budget):

| budget | page cache | explicit cache | faults/token | |
|---|---|---|---|---|
| 2.5 GB | 1.89 tok/s | **2.92 / 2.98** (32 slots, 1.86 GB) | 24.9k -> ~850 | 1.56x |
| 1.5 GB | 1.07 / 1.17 | **1.51 / 1.68** (16 slots, 0.93 GB) | 47-50k -> ~900 | 1.4x |

(One baseline run hit 28.6 tok/s: Windows had paged other apps out and the
whole model fit — the "available" knob is only approximately a budget, and it
favours the page-cache side, whose standby list can grow into memory freed
from other processes while our locked cache cannot.)

Per token at 32 slots: 108 hits, 11 late hits, 8 misses, 40 prefetched,
185 MB read, main thread waits ~290 ms -> the cache path is purely
bandwidth-bound: 185 MB / 0.65 GB/s = 285 ms. Page faults are gone.

**What that means.** With the I/O owned, tok/s = bytes per token / sustained
bandwidth, overlapped with compute. The remaining levers are all about bytes
and bandwidth, none about faults:
1. bytes per expert: Q4_K_M keeps `down_exps` at Q6_K (1.7 MB vs 1.2 MB);
   a uniform Q4/IQ4 quant is ~15% fewer bytes, Q3 ~30% (quality: measure).
2. K: `--override-kv olmoe.expert_used_count=int:4` halves the bytes with no
   code change — Edge0's K=4 trick; needs a perplexity check.
3. the drive: on a desktop NVMe sustaining 3+ GB/s the same code is ~10 tok/s
   at 50% residency; here a cooling pad would literally be a 2x speedup.
4. prefetch waste (12% wrong predictions) and eviction policy: small.
5. an expert-bundled layout (one 3.8 MB read instead of three 1.2-1.7 MB
   reads per expert) — larger requests are worth ~20-30% at QD1-2 on this
   drive per `disk-bench.py`.

## Results 7 — is K=4 free? Perplexity vs. active experts (2026-09-18)

`bench/ppl-k-sweep.ps1` / `llama-perplexity --override-kv olmoe.expert_used_count=int:K`,
OLMoE-1B-7B-Instruct Q4_K_M, ctx 512. Two corpora: `bench/corpus.txt` (git
documentation prose, built by `bench/make-corpus.py`, 40 chunks) and
wikitext-2-raw test (`C:\models\eval`, 80 chunks) for comparability with
published numbers.

| K | expert bytes/token | PPL git-doc | PPL wikitext-2 |
|---|---|---|---|
| 8 (native) | 100% | 4.85 +/- 0.11 | 11.44 +/- 0.23 |
| 6 | 75% | 5.44 +/- 0.14 (+12%) | 12.50 +/- 0.26 (+9%) |
| 4 | 50% | 9.08 +/- 0.26 (+87%) | 16.17 +/- 0.36 (+41%) |
| 2 | 25% | 48.5 +/- 1.9 (10x) | — |

**No, K=4 is not free** on OLMoE: +41% perplexity on wikitext-2, +87% on
technical prose. K=6 costs 9-12% for 25% fewer bytes — in the same regime as dropping one quant level,
and a legitimate knob. K=2 is dead. The curve is convex: each expert removed
hurts more than the last.

This is the concrete case for Edge0's Recover-LoRA: their K=4 on Qwen3.6 is
only viable because adapters were distilled against the K=4 routing. Raw, the
2x I/O saving from K=4 buys a much worse model. Caveats: OLMoE (64 small
experts, no shared expert) may be more K-sensitive than Qwen3.6-35B-A3B
(256 experts, top-8 + a shared expert that carries a baseline) — the same
sweep costs nothing to run on Qwen3.6 once it is downloaded, and should be.

Ranking of bytes-per-token levers by quality cost, so far:
1. lower-bit quantisation (typically +3-10% PPL for 15-30% fewer bytes),
2. K=6 (+12% for -25%),
3. K=4 (+87% for -50%) — only with recovery training.

## Next step options (updated)

- **K=4 + quant sweep** (cheap, no code): `--override-kv` for K, IQ4_XS/Q3_K_M
  quants of OLMoE (4 and 3 GB downloads), `llama-perplexity` for quality,
  `moe-run --cache-slots` for speed. Establishes the bytes/quality frontier.
- **Real target**: Qwen3.6-35B-A3B (22 GB) with the explicit cache. Needs the
  disk space and ~12 min download; the cache code is model-agnostic
  (`ffn_*_exps` + `ffn_gate_inp`), Qwen3.6's linear-attention layers are
  handled by llama.cpp.
- **Runtime**: the Rust port now has a validated reference: slot cache +
  prediction + urgent-miss queue + unbuffered reads. Add overlapped I/O
  (three chunks in flight per expert), expert-bundled layout, adaptive width.

## Log

- 2026-09-17 — llama.cpp cloned at `972d231` (master, 2026-09-17). Built with
  MSVC 19.44, `/arch:AVX2` + AVX-VNNI, OpenMP 2.0, no OpenSSL. Stopped the Dell
  TechHub suite to free ~1 GB (in-use 8.4 -> 7.4 GB).
- 2026-09-18 — baseline curves, disk bench, routing trace, prefetch spike
  (see Results 1-5). Tools: `tools/moe-trace`, `tools/moe-run`; harness in
  `bench/`. numpy installed for the analysis.
- 2026-09-18 (later) — explicit expert cache built and measured (Results 6):
  1.4-1.56x over the page cache at equal RAM budget, faults/token 25-50k -> ~850,
  bandwidth-bound on a drive that throttles to ~650 MB/s.
