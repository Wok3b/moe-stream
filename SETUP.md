# Setup

This repo holds only our own code: benchmark harness, tracing/inference tools,
and patches against llama.cpp. The upstream clone, the build, the models and the
raw traces are **not** included (they are large and reproducible). This file
reassembles a working tree from scratch.

Everything was developed and measured on Windows; the Linux notes at the bottom
say exactly what still needs porting.

---

## Windows

### Prerequisites

| | |
|---|---|
| Visual Studio 2022 or newer | Build Tools are enough; needs "Desktop development with C++" (MSVC + Windows SDK). Built and measured with v143 and with MSVC 14.51 |
| CMake | 3.20+, on PATH |
| Python | 3.10+, on PATH; `pip install numpy matplotlib huggingface_hub` |
| Git | for the clone |

### 1. Clone llama.cpp at the pinned commit

The patches and the in-tree tools are written against **`972d231`** (master,
2026-09-17). Later upstream commits may move `build_moe_ffn` or the loader and
break `patches/0002-expert-cache-hook.patch`.

```bash
git clone https://github.com/ggml-org/llama.cpp llama.cpp
cd llama.cpp && git checkout 972d231 && cd ..
```

### 2. Apply the patches

```bash
cd llama.cpp
git apply ../patches/0001-lazy-experts.patch
git apply ../patches/0002-expert-cache-hook.patch
cd ..
```

- `0001-lazy-experts.patch` — with `--lazy-mode on`, every `*_exps.weight`
  tensor is marked `TENSOR_READ_LAZY`, so the loader does not try to prefetch
  the whole file into RAM at startup.
- `0002-expert-cache-hook.patch` — adds `llama_expert_cache_register()` and
  makes `build_moe_ffn` run the expert matmuls against registered cache tensors
  addressed by a rewritable `ffn_moe_slot_ids-<il>` tensor.

### 3. Copy our tools into the tree and register them

```powershell
.\patches\install-tools.ps1
```

This copies `tools/*` into `llama.cpp/examples/` and appends the
`add_subdirectory(...)` lines. Re-run it after editing anything under `tools/`.

### 4. Configure and build

```powershell
cmake -S llama.cpp -B llama.cpp\build -G "Visual Studio 17 2022" -A x64 -DGGML_NATIVE=ON -DGGML_AVX_VNNI=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF
```

Drop `-G` to let cmake pick the newest Visual Studio it finds, or name yours
(`"Visual Studio 18 2026"`). The generator has to be a Visual Studio one, not
Ninja, because the benchmark scripts expect binaries in `build\bin\Release\`.

```powershell
cmake --build llama.cpp\build --config Release --parallel 8
```

`GGML_NATIVE=ON` lets MSVC pick the host SIMD level. Drop `-DGGML_AVX_VNNI=ON`
on a CPU without AVX-VNNI (pre-Alder Lake). `LLAMA_OPENSSL=OFF` avoids needing
OpenSSL; nothing here fetches over HTTPS.

Binaries land in `llama.cpp\build\bin\Release\`, including our
`llama-moe-run.exe`, `llama-moe-trace.exe` and `llama-moe-bundle.exe`.

### 5. Get a model

Any MoE GGUF works. The reference subject is OLMoE-1B-7B (16 layers, 64
experts, top-8, 4.2 GB at Q4_K_M):

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/OLMoE-1B-7B-0924-Instruct-GGUF', filename='OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf', local_dir=r'C:\models')"
```

Scripts default to `C:\models`; every one of them takes `-Model` explicitly, so
put weights wherever you like.

Optional, for perplexity work — wikitext-2:

```powershell
curl.exe -sSL -o "$env:TEMP\wt2.zip" https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip; python -c "import zipfile,os; zipfile.ZipFile(os.environ['TEMP']+r'\wt2.zip').extractall(r'C:\models\eval')"
```

### 6. Smoke test

Warm run, explicit expert cache on, prefetch two layers ahead:

```powershell
.\llama.cpp\build\bin\Release\llama-moe-run.exe -m C:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf -p "The capital of France is" -n 32 -t 10 -lzm on --cache-slots 32 --prefetch-depth 2 --io-threads 2
```

You should see an `expert_cache:` line reporting slots and locked MiB, then a
`decode ... tok/s` line and a `cache:` line with hits/misses per token.

### 6b. Build the expert-bundled sidecar (optional, worth 1.5x)

`llama-moe-bundle` rewrites the expert weights into a file where each expert's
matrices are contiguous, so the cache fetches one with a single read instead of
three. The GGUF is not modified; the sidecar sits next to it and is about the
size of the expert tensors (3.63 GiB for this model).

```powershell
.\llama.cpp\build\bin\Release\llama-moe-bundle.exe -m C:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf --verify
```

`--verify` re-reads every expert and compares it byte for byte with the GGUF.
Then add `--experts-file C:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf.experts`
to any `llama-moe-run` command that uses `--cache-slots`.

**Correctness check that matters:** the `tokens hash` printed at the end must be
identical with and without `--cache-slots`, and with and without
`--experts-file`. Same prompt, greedy (no `--temp`),
same `-n`. If the hashes differ, the cache is serving wrong expert weights.

```powershell
.\llama.cpp\build\bin\Release\llama-moe-run.exe -m C:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf -p "The capital of France is" -n 32 -t 10 -lzm on
```

### 7. Reproduce the measurements

```powershell
.\bench\machine-info.ps1 | Out-File bench\results\machine.json -Encoding utf8
.\bench\disk-bench.py                      # run via: python bench\disk-bench.py <a big file>
.\bench\run-bench.ps1 -Model C:\models\<model>.gguf -Threads "4,6,8,10,12"
.\bench\sweep-pressure.ps1 -Model C:\models\<model>.gguf -AvailableGB 6,3,2,1.5 -Threads 10
.\bench\ab-prefetch.ps1 -Model C:\models\<model>.gguf -AvailableGB 2.5 -Configs "0 8","2 8"
.\bench\ppl-k-sweep.ps1 -Model C:\models\<model>.gguf -Arch olmoe -Text C:\models\eval\wikitext-2-raw\wiki.test.raw -K 8,6,4
.\bench\ab-layout.ps1 -Model C:\models\<model>.gguf -Slots 32 -Rounds 3
python bench\make-charts.py
```

`make-charts.py` regenerates every figure in
`docs/streaming-moe-on-a-laptop.md` from the files in `bench/results/`.

`sweep-pressure` and `ab-prefetch` pin RAM with `bench/ram-eater.py` to simulate
a smaller machine; the box will feel sluggish while they run, which is expected.

`docs/NOTES.md` has every number we measured and why, in order. Read it before
re-running anything — Results 1-7 explain what each script was built to answer.

---

## Linux

**Status: not ported.** llama.cpp itself builds fine, and the two patches are
platform-neutral (they touch `llama-model-loader.cpp` and `llama-graph.cpp`).
What does not build is our tooling.

Build the upstream part exactly as above but with:

```bash
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF && cmake --build llama.cpp/build -j
```

### What needs porting

`tools/moe-run/expert_cache.cpp` — the I/O path, all inside `#ifdef _WIN32`:

| Windows | Linux |
|---|---|
| `CreateFileA(.., FILE_FLAG_NO_BUFFERING, ..)` | `open(path, O_RDONLY \| O_DIRECT)` |
| `SetFilePointerEx` + `ReadFile` | `pread(fd, buf, len, off)` — simpler and thread-safe, no seek |
| `VirtualAlloc` staging buffer | `posix_memalign(&p, 4096, n)` — O_DIRECT needs aligned buffers |
| `VirtualLock` + `SetProcessWorkingSetSizeEx` | `mlock(base, size)`; needs `RLIMIT_MEMLOCK` raised (`ulimit -l`) or `CAP_IPC_LOCK` |
| `GetLastError()` | `errno` / `strerror` |

`tools/moe-run/moe-bundle.cpp` builds as is: it uses stdio and already maps
`fseek64` to `fseeko`. Only the cache reading the sidecar is Win32.

`tools/moe-run/moe-run.cpp` — the mmap-prefetch path and the counters:

| Windows | Linux |
|---|---|
| `PrefetchVirtualMemory` | `posix_madvise(.., POSIX_MADV_WILLNEED)` |
| `QueryWorkingSetEx` (residency check) | `mincore(addr, len, vec)` |
| `GetProcessMemoryInfo(..).PageFaultCount` | `getrusage(RUSAGE_SELF).ru_majflt` (major faults only — arguably a better metric than what we used) |

Note that the `PrefetchVirtualMemory` path is the one Results 5 showed does
**not** work well; only the `expert_cache` path (Results 6) is worth porting
first. The prefetcher's routing math (`prefetcher::route`) is plain C++ and
already portable.

`bench/*.ps1` — rewrite in bash. `bench/analyze-trace.py` is portable as-is;
`bench/ram-eater.py` and `bench/disk-bench.py` are ctypes-on-Win32 and need
Linux equivalents (`mmap` + `mlock`, and `O_DIRECT`).

### Linux is a better measurement platform

Two of the weaknesses in our Windows data go away:

- **Real memory budgets.** `systemd-run --scope -p MemoryMax=2G -p MemorySwapMax=0 -- <cmd>` enforces a hard cgroup limit. Our `ram-eater.py` only *pins* pages, and Windows responded by paging other processes out mid-run, which drifted the budget between A/B runs (see the Results 6 caveat about a baseline hitting 28.6 tok/s).
- **Cold caches.** `echo 3 | sudo tee /proc/sys/vm/drop_caches` is reliable; our `purge-standby.ps1` needs UAC and an undocumented `NtSetSystemInformation` call.

Also `/usr/bin/time -v`, `perf stat`, and `iostat -x` give per-process major
faults and per-device queue depth directly, which on Windows took performance
counters and guesswork.

---

## Reference machine

All numbers in `docs/NOTES.md` come from:

| | |
|---|---|
| Dell Vostro 15 3530, Windows 11 Pro 24H2 | |
| Intel Core i5-1334U | 2P + 8E, 12 threads, AVX2 + AVX-VNNI, no AVX-512 |
| 16 GB DDR4-3200 dual channel | ~7-8 GB actually available to a model |
| Intel Iris Xe (shared memory) | no dGPU, no NPU |
| Samsung BM9C1 512 GB NVMe | DRAM-less; 1.2-1.4 GB/s cold, **throttles to ~650 MB/s** sustained |

That last line matters more than anything else in the table. Results 6 is
bandwidth-bound, so the same code on a drive that sustains 3 GB/s should be
roughly 3x faster. Re-running `bench/disk-bench.py` on a new machine first tells
you what to expect from everything else.
