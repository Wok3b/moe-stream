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

Per token at 32 slots, from the two logged runs (`ab-20260918-013008` and
`ab-20260918-013149`): 97.2 and 97.8 hits, 22.7 and 22.2 late hits, 8.2 misses,
41.9 prefetched, 39.4 evictions, 191.3 MB read, main thread blocked 283 and
273 ms. Each of the two I/O threads moved 291 and 299 MB/s, so the drive was
busy for roughly 330 ms of a 343 ms token. The cache path is purely
bandwidth-bound and page faults are gone.

(An earlier draft of this paragraph quoted 108 hits, 11 late hits, 185 MB and
~290 ms. Those were from an unlogged run during development; the figures above
are the ones in `bench/results/`, and they are what the article and the charts
use.)

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

## Results 8 — expert-bundled layout (2026-10-07)

Built (the layout lever from Results 6, item 5):

- `tools/moe-run/expert_bundle.h` — the sidecar format. A GGUF cannot express a
  bundled layout: every tensor there has to be one contiguous blob and bundling
  interleaves bytes from three of them. So the bundled copy lives in a sidecar
  file beside the model; the GGUF is untouched and still loads normally, and
  only the expert cache reads the sidecar. Bundles are 4096-aligned and the last
  one is padded to a full stride, so every read is sector aligned and in bounds.
- `tools/moe-run/moe-bundle.cpp` (`llama-moe-bundle -m model.gguf [--verify]`) —
  writes it. `--verify` compares every expert byte against the GGUF.
- `expert_cache` gained `--experts-file`: one `ReadFile` of the whole bundle per
  expert instead of three. The header is checked against the model (tensor names
  in cache order, bytes per expert, expert count) and rejected rather than
  served if anything disagrees.

OLMoE Q4_K_M: 16 layers x 64 experts = 1024 bundles, stride 3.38-3.89 MiB
(Q4_K_M uses different quant types per layer, so it varies), sidecar 3.63 GiB
next to the 3.92 GiB GGUF. `--verify` passes.

A/B (`bench/ab-layout.ps1`), interleaved split/bundled, 48 greedy tokens,
prefetch depth 2, 2 I/O threads, no memory pinning, 3 rounds each:

| cache | layout | tok/s | requests/token | MB/request | MB/s per I/O thread | main thread wait |
|---|---|---|---|---|---|---|
| 32 slots (1.86 GiB) | split | 6.30 / 6.24 / 6.21 | 150.3 | 1.27 | 649 | 115 ms/token |
| 32 slots | **bundled** | **9.73 / 9.33 / 9.49** | **50.1** | **3.82** | **1031** | **62 ms/token** |
| 16 slots (0.93 GiB) | split | 3.50 / 3.61 / 3.60 | 275.8 | 1.27 | 643 | 229 ms/token |
| 16 slots | **bundled** | **5.71 / 5.76 / 5.79** | **91.9** | **3.81** | **1049** | **127 ms/token** |

**1.52x at 32 slots, 1.61x at 16.** Token hash `3b15dac0721bfcbf` on all twelve
runs, so the layout is bit-exact like the cache itself. Bytes per token are
identical on both sides (191.3 MB at 32 slots, 349.9 MB at 16): the gain is
entirely the drive doing more with the same bytes, 645 -> 1040 MB/s per I/O
thread, which tracks the request-size curve in `disk-bench.json` independently.

More than the 20-30% predicted in Results 6. The guess compared 1.27 MB against
3.8 MB requests at a fixed queue depth; the real pattern is three separate seeks
per expert interleaving across two I/O threads, which costs more than the size
difference alone.

**Why these runs are unpinned.** Both layouts read with `FILE_FLAG_NO_BUFFERING`
and the cache is locked, so neither side depends on the page cache and pinning
changes nothing about what is being compared. It only adds noise, and on this
machine it does not work reliably at these cache sizes:

- The one pinned sequence that completed (2.5 GB available, 32 slots, 4 runs,
  tagged `r8-avail2.5g` in `layout.csv`) read 6.01 split, 4.44 bundled, 2.39
  split, 3.23 bundled, with per-thread throughput falling 632 -> 474 -> 237 ->
  330 MB/s as the drive heated. The thermal drift inside one sequence is larger
  than the effect being measured, so that sequence says nothing either way.
- Every later attempt failed outright. `ram-eater.py` pins whatever is free at
  the time, so on an idle machine it pinned 6.1 to 7.3 GB, and a 2.5 GB budget
  then cannot host the locked cache at all: `failed to allocate buffer of size
  1950351360`. Making the budget a real budget needs cgroup-style limits, which
  is one of the reasons `SETUP.md` argues Linux is the better measurement
  platform.

The unpinned runs are reported as the result because they are the controlled
measurement, not because they are the flattering one.

**Not isolated:** the sidecar is a freshly written file, so some of the gain
could be its physical placement rather than request size. Requests per token
fell exactly 3x and per-request throughput rose in line with the disk benchmark,
which is why I think it is request size, but the two are not separated.

**Harness bugs found on the way.** Both were silent and both predate this
change: (a) the CSV writers formatted with the current culture and then stripped
commas as thousand separators, so on a machine with a decimal comma 6.21 was
written as 621 — they now force `InvariantCulture`; (b) `Start-Process
-ArgumentList` was passing unquoted paths, so any path containing a space (this
repo lives under "AI Engine") broke the `--json` argument and the ram-eater
launch.

## Results 9 — Gemma 4 26B-A4B: the cache on a real target (2026-10-13)

A second machine and the first model that is not a control: **Ryzen 5 7430U** (6C/12T, AVX2, no AVX-VNNI,
so built with `-DGGML_NATIVE=ON` only), 15.4 GB RAM of which ~8 GB is free with nothing loaded, Solidigm
NVMe. `disk-bench-ryzen7430u.json`: 1302-1339 MB/s for 2 MB random reads at QD4-8, 961 MB/s at QD1,
26 MB/s for 4 KB QD1 — the same shape and almost the same numbers as the reference machine, so its
conclusions should carry over.

Model: `gemma-4-26B-A4B-it-UD-IQ3_S.gguf` (unsloth), 10.5 GiB, 30 layers x 128 experts, top-8.
`bench/gguf-experts.py` shows the layout per layer: merged `ffn_gate_up_exps` (IQ2_S, 1240 KiB per
expert) + `ffn_down_exps` (IQ4_NL, 1089 KiB) = **2.33 MiB per expert, 8.9 GiB of the 10.5 are experts**,
plus a per-expert scale `ffn_down_exps.scale`. One decode token touches 30 x 8 x 2.33 = 559 MB of experts.

Built:

- `patches/0003-expert-scale-ids.patch`. `build_lora_mm_id` gets an `ids_e` argument: the per-expert
  scale and LoRA index by real expert ids while the matmul uses slot ids. 0002 instead turned the cache
  off whenever any `*_exps_s` tensor existed, so on Gemma 4 the cache would silently never have run.
- `moe-run` now marks `temp` as user-set. **Harness bug:** a GGUF can carry `general.sampling.temp`
  (Gemma 4: 1.0, top_k 64, top_p 0.95) and the loader applies it over the tool's greedy default, so
  every "greedy" run sampled at temp 1.0 with a random seed and no two hashes matched — including two
  plain mmap runs. OLMoE's GGUF has no such key, so the hashes in Results 6 and 8 stand.
- `bench/chat-prompt.py`: renders a prompt with the model's own chat template. Gemma 4 instruct
  produces garbage on a bare "The capital of France is" in every build including upstream b11580, so
  hash checks need templated prompts (`bench/prompts/`).

**Bit-exact:** token hash `07790982544d57b8` for mmap and for 32, 64 and 96 slots (256 greedy tokens,
story prompt, `-c 16384`, `-t 6`, `-lzm on`, 4 I/O threads, no prefetch, unpinned):

| | experts locked | decode tok/s | misses/token (of 240) | read MB/token | main thread wait |
|---|---|---|---|---|---|
| mmap | - | **6.65** | - | - | - |
| 32 slots | 2.2 GiB | 3.89 | 42.6 (18%) | 102 | 116 ms/token |
| **64 slots** | **4.4 GiB** | **5.79** | **10.3 (4.3%)** | 25 | 31 ms/token |
| 96 slots | 6.6 GiB | 5.60 | 8.8 | 21 | 26 ms/token |

**Gemma 4 has a hot set.** Half the slots catch 96% of lookups, where OLMoE needed 38 of 64 experts to
cover 80% (Results 2). Cache policy matters on this model; on OLMoE it did not.

Single run per row, so read the 64/96 difference as noise; the 32 -> 64 step is not. The mmap row is the
best case for mmap: nothing else running and the model nearly fits.

## Results 10 — layer-streamed prefill, and the cache inside llama-server (2026-10-13)

The first use of the cache outside `moe-run`: a journal grader (`llama-server`, JSON-schema output,
735-token prompt, ~250 tokens out). `patches/0004-server-expert-cache.patch` adds the cache to the
server through environment variables (`MOE_CACHE_SLOTS`, `MOE_IO_THREADS`, `MOE_CACHE_MAX_TOKENS`,
`MOE_STREAM_PREFILL`, `MOE_EXPERTS_FILE`), so `common/arg.cpp` stays untouched.

With only the decode cache, grading got **worse**: 134 s against 76 s, 11.7 GB peak working set, 31 MB
of RAM left. Prompt processing (n_tokens > `max_tokens`) still ran on the mmap, and a 512-token batch
touches nearly every expert, so the OS paged all 8.9 GiB of experts in on top of the 4.4 GiB locked cache.

Built:

- **Layer streaming** (`patches/0005-expert-stream-prefill.patch` + `expert_cache` stream mode,
  `moe-run --stream-prefill`). Each expert tensor gets a second registration, a *stream* view with all
  128 experts, used for batches above `max_tokens`. Two buffers of 344 MiB hold one layer each; MoE layer
  k lives in buffer k % 2. When layer k's `ffn_moe_slot_ids` callback fires, the runtime waits for layer k
  and queues layer k+1 into the other buffer, whose previous layer's matmuls ran before this callback.
  Layers load as two big sequential reads split into 4 MiB requests over the I/O threads, from the GGUF
  even when a sidecar is in use. After the last layer it loads layer 0 for the next batch.
- **`max_tokens` = slots / experts-per-token** when not given (8 at 64 slots), and clamped to it always:
  a batch that needs more distinct experts in a layer than there are slots would wait forever in
  `on_slot_ids`. Batches of 2-8 tokens therefore use the slots instead of streaming 8.9 GiB.
- **Seeding:** when a layer is in its stream buffer, the experts the batch used most are copied into
  the slots (memcpy, no I/O), so decode does not start from an empty cache.
- **Dense prefetch:** with experts streamed, the non-expert weights (569 tensors, 2.5 GiB) are what the
  first prompt faults in; one `PrefetchVirtualMemory` call at init (0.18 s, asynchronous).

**Bit-exact:** the 735-token grading prompt (two ubatches) gives `c3e9a15388bb95a5` on mmap, with
streaming, with seeding, and with prefetch depth 1 and 2. The story run in Results 9 also matches with
streaming on.

`moe-run`, grading prompt, 192 tokens out, 64 slots:

| | prefill tok/s | decode tok/s | misses/token | wait/token | graph waited for streaming |
|---|---|---|---|---|---|
| mmap | 19.8 | **6.70** | - | - | - |
| stream | 20.9 | 4.85 | 21.8 | 61 ms | 1.98 s of 35 s |
| stream + seeding + dense prefetch | **22.8** | 5.10 | 15.2 | 46 ms | 2.10 s of 32 s |

Streaming hides almost all of its I/O behind compute: 18.7 GB read for two ubatches, 2 s of waiting.
Prefill is now compute-bound. Decode is slower than mmap for a short answer because the cache starts
cold for what the *answer* needs; seeding takes misses from 21.8 to 15.2 (1920 experts seeded in 0.55 s),
but an English JSON answer uses different experts than a Serbian entry plus English instructions.

On the story prompt (34-token prefill, 256 tokens out) streaming made decode *faster*, 6.69 tok/s with
5 page faults/token against 5.79 with prefill on the mmap: the cache no longer competes with 8.9 GiB of
paged-in experts.

**The stale-router prefetcher does not pay on Gemma 4.** Gemma 4 routes on `attn_out`, not `ffn_inp`,
through `rms_norm(x) / sqrt(n_embd) * ffn_gate_inp_s`; the prefetcher now does the same for this arch.

| prefetch | prediction hit rate | decode tok/s | read MB/token | evictions/token |
|---|---|---|---|---|
| off | - | **5.10** | 37 | 15 |
| depth 1, width 8 | 0.656 | 4.58 | 78 | 32 |
| depth 2, width 8 | 0.702 | 4.11 | 118 | 49 |

66-70% against OLMoE's 84%, and with 64 of 128 slots every wrong guess evicts an expert that was going
to be used. The routing math also costs 12-23 ms/token on one thread. Off by default.

**In the server** (`bench/journal-ab.ps1`, same entry graded by each config, `journal-ab.csv`):

| | wall s | prompt tok/s | decode tok/s | peak working set | lowest available RAM | grades |
|---|---|---|---|---|---|---|
| mmap | 101 / 86 / 128 | 16.3 / 26.4 / 27.1 | 7.3 / 6.6 / 6.7 | 10.1-10.6 GB | **13 / 76 / 46 MB** | 5/4/5/5 x3 |
| moe-stream, 64 slots | 102 / 114 / 105 | 19.6 / 20.1 / 19.5 | 4.8 / 4.8 / 4.6 | 9.3-9.4 GB | **805 / 835 / 962 MB** | 5/4/5/5 x3 |

**Same wall time; the difference is headroom.** mmap takes the machine down to tens of MB free, so
anything else running pages; moe-stream leaves ~0.9 GB free and its prompt speed does not depend on
what the OS cache still holds (mmap's 16 -> 27 tok/s across rounds is the previous run's pages). Peak is
9.3 GB rather than the ~7 GB I estimated because the non-expert weights are 2.5 GiB, not 1.6.

This is not a speed win at this size because the model nearly fits. It is the setup for what does not:
the same code runs a 13 GiB QAT quant or a 20 GiB Qwen3.5-35B-A3B on this machine.

Caveats: the first moe-stream round ran before seeding and dense prefetch were added (rows 2-3 have
them). Three rounds, one machine, unpinned. The server prints no cache counters yet, so server-side
miss rates are inferred from the matching `moe-run` runs.

## Results 11 — Linux port, and the model under real memory budgets (2026-10-13)

Same laptop as Results 9-10 (Ryzen 5 7430U, 15.4 GB, Solidigm NVMe), booted into Arch instead of
Windows: Linux 7.2, GCC, btrfs, model copied onto the Linux partition. `expert_cache.cpp` and
`moe-run.cpp` now build on both systems (SETUP.md, Linux): `O_DIRECT` + `pread`, `posix_memalign`,
`mlock`, `madvise(WILLNEED)`, `mincore`, and `getrusage` so major faults are counted on their own.
`mlock` fails at Arch's 8 MiB `ulimit -l`; it warns and goes on, as on Windows.

**Bit-exact within the Linux build.** Story prompt, 256 tokens, `-t 6 -c 16384 -lzm on`: mmap, 32, 64
and 96 slots, and 64 slots + `--stream-prefill` all give `090643b6af8e29c9` (MSVC gave `07790982544d57b8`;
GCC rounds differently, as expected). All 15 budget runs below give `aa7be0807a9ce7a9` (128 tokens).

| story, 256 tokens | Windows decode tok/s | Linux decode tok/s | Linux misses/token | Linux major faults/token |
|---|---|---|---|---|
| mmap, warm | 6.65 | 6.77, then **8.73** | - | 2443, then 972 |
| 32 slots | - | 5.78 | 42.2 | 0.3 |
| 64 slots | 5.79 | 9.11 | 10.2 | 1.0 |
| 96 slots | - | 9.03 | 8.6 | 1.0 |
| 64 slots + stream prefill | 6.69 | **9.76** | 6.1 | 0.0 |

Linux is faster on everything, mmap included (+31% warm), and the cache gains more (+46%). The miss
counts are close to Windows' (64 slots: ~10/token both), so the gap is not I/O. Most likely the compiler
(GCC vs MSVC on the ggml CPU kernels) plus less background load; not isolated. I/O per thread: 520-800
MB/s for 1.2 MB `O_DIRECT` reads, with 4 threads.

**Memory budgets** (`bench/mem-budget.sh`): `systemd-run --user --scope -p MemoryMax=X -p
MemorySwapMax=0`, no root. Every run starts cold for the model file (`bench/evict-file.py`,
`posix_fadvise(DONTNEED)`). Story prompt (34 tokens), 128 tokens out, `--io-threads 4`. Page cache counts
against `MemoryMax`, so the mmap run gets exactly X for weights + KV + everything else.

| budget | mmap decode | mmap wall | 32 slots + stream | wall | 64 slots + stream | wall |
|---|---|---|---|---|---|---|
| 6 GB | 1.97 (18063 majflt/tok) | 91 s* | **5.68** (0) | 30 s* | 0.41 (655 majflt/tok) | 320 s |
| 8 GB | 4.88 (4936) | 52 s | 5.94 (0) | 29 s | **9.58** (0) | 21 s |
| 10 GB | 6.31 (3136) | 46 s | 5.93 | 29 s | **9.44** | 21 s |
| 12 GB | 6.35 (3128) | 46 s | 5.95 | 29 s | **9.56** | 21 s |
| none | 6.35 (3128) | 46 s | 5.94 | 29 s | **9.57** | 21 s |

\* `bc` was missing for the first two runs; process time from the log's elapsed stamps instead.
Wall includes model load. Prefill: mmap 1.4 tok/s with ~1.04 M major faults, streaming 6.2 tok/s with
~380, at every budget.

What it shows:

- **At 8 GB, moe-stream is 2x mmap on decode and 2.5x end to end** (21 s vs 52 s), at the same output.
  At 6 GB, 32 slots runs 2.9x mmap and does not fault at all. moe-stream's speed does not move from
  8 GB to unlimited: it only uses what it allocated.
- **mmap above 10 GB is limited by the cold start, not the budget.** 10, 12 GB and unlimited are
  identical (6.3 tok/s, ~3.1k major faults/token): the first touch of each expert is a 4 KiB-granular
  fault, 128 tokens is too short to warm up, and prefill pages in 1 M faults at 1.4 tok/s. Warm, mmap
  reaches 8.7 tok/s (table above), still below 64 slots + stream at 9.6-9.8.
- **Too many slots is worse than mmap.** 64 slots + stream at 6 GB leaves under 1 GB for the 2.5 GiB of
  non-expert weights and the KV cache; those thrash instead (655 major + 33k minor faults/token, 0.41
  tok/s). The budget rule from Results 10 holds: slots x 69 MiB + 688 MiB stream + 2.5 GiB dense + KV.
  A `--cache-slots auto` that sizes from `memory.max` would avoid the cliff.

Caveats: one prompt, short output, one run per cell (the warm mmap runs show run-to-run drift of 30%).
Desktop session running (browser, terminal, ~4 GB). mmap's cold numbers include the model's first read
from disk, which a long-running server pays once; the warm row is the fair decode comparison.

## Next step options (updated)

- **Bundled sidecar for Gemma 4** (needs ~9 GB of disk): one request per miss instead of two, which
  Results 8 measured at 1.5x on OLMoE. Decode still waits 46 ms/token on misses.
- **A model that does not fit**: Gemma 4 QAT UD-Q4_K_XL (13.3 GiB) or Qwen3.5-35B-A3B Q4_K_M
  (20.5 GiB). This is where streaming prefill + slots should beat mmap outright, not just tie it.
- **Cheaper wrong guesses**: prefetch only into free or cold slots, or into a small separate pool, so a
  mispredicted expert cannot evict a hot one; and run the routing math off the graph thread.
- **Server counters**: cache and stream stats per request in the server log.
- **K=4 + quant sweep** (cheap, no code): `--override-kv` for K, IQ4_XS/Q3_K_M
  quants of OLMoE (4 and 3 GB downloads), `llama-perplexity` for quality,
  `moe-run --cache-slots` for speed. Establishes the bytes/quality frontier.
- **Real target**: Qwen3.6-35B-A3B (22 GB) with the explicit cache. Needs the
  disk space and ~12 min download; the cache code is model-agnostic
  (`ffn_*_exps` + `ffn_gate_inp`), Qwen3.6's linear-attention layers are
  handled by llama.cpp.
- **Runtime**: the Rust port now has a validated reference: slot cache +
  prediction + urgent-miss queue + unbuffered reads + bundled layout (Results 8).
  What is left to add is real asynchronous I/O (IOCP / io_uring) so one thread
  can hold the queue depth the drive wants, which `disk-bench.json` says is 2-4,
  and adaptive prefetch width.
- **Upstream**: `0002-expert-cache-hook.patch` is small and architecture
  agnostic, and upstream is already moving this way (`-lm dio`, `-lzm`,
  `TENSOR_READ_LAZY`). Landing the hook there reaches users without maintaining
  a patch against a tree that moves daily.

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
- 2026-10-07 — working tree rebuilt from SETUP.md against the same pin (`972d231`,
  now with MSVC 14.51). Disk bench re-run on a cool drive: 1496 MB/s for random
  2 MB QD1 against 1194 in September, same shape, confirming the throttling
  story. Expert-bundled layout built and measured (Results 8): 1.52-1.61x,
  bit-exact. Article written up in `docs/streaming-moe-on-a-laptop.md`, with
  every figure generated from `bench/results/` by `bench/make-charts.py`.
- 2026-10-13 — second machine (Ryzen 5 7430U, VS 2026, no AVX-VNNI). Gemma 4 26B-A4B
  ported (patch 0003, Results 9), moe-run sampling bug found and fixed, cache added to
  llama-server (0004), layer-streamed prefill + slot seeding + dense prefetch (0005,
  Results 10). Stale-router prefetch measured as a loss on Gemma 4. Raw numbers in
  `bench/results/gemma4/`.
- 2026-10-13 (later) — Arch on the same laptop: tools ported to Linux (O_DIRECT, mlock, madvise,
  mincore, getrusage), bit-exact across mmap / slots / stream. cgroup budget sweep 6-12 GB (Results 11):
  2x mmap's decode at 8 GB, 2.9x at 6 GB with 32 slots. Raw numbers in `bench/results/linux/`.
