#include "expert_cache.h"

#include "ggml-alloc.h"
#include "gguf.h"
#include "llama.h"
#include "llama-model.h" // in-tree: layer tensors
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {
constexpr size_t SECTOR = 4096; // unbuffered I/O alignment

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

expert_cache::~expert_cache() {
    {
        std::lock_guard<std::mutex> lk(mu);
        stop = true;
    }
    cv_jobs.notify_all();
    for (auto & t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    llama_expert_cache_clear();
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

bool expert_cache::init(const llama_model * model, const std::string & gguf_path, const expert_cache_params & params) {
    p    = params;
    path = gguf_path;
    n_expert = (int) model->hparams.n_expert;
    if (p.slots > n_expert) {
        p.slots = n_expert;
    }
    const int n_layer = (int) model->layers.size();
    layers.resize(n_layer);

    // absolute file offsets of the expert tensors
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gp);
    if (!gctx) {
        LOG_ERR("expert_cache: cannot read GGUF metadata from %s\n", path.c_str());
        return false;
    }
    const uint64_t data_off = gguf_get_data_offset(gctx);

    int n_tensors = 0;
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = model->layers[il];
        if (!L.ffn_gate_inp || !L.ffn_down_exps) {
            continue;
        }
        const ggml_tensor * srcs[3] = {};
        int n = 0;
        if (L.ffn_gate_up_exps) {
            srcs[n++] = L.ffn_gate_up_exps;
        } else {
            srcs[n++] = L.ffn_up_exps;
            if (L.ffn_gate_exps) {
                srcs[n++] = L.ffn_gate_exps;
            }
        }
        srcs[n++] = L.ffn_down_exps;
        for (int i = 0; i < n; ++i) {
            const int64_t id = gguf_find_tensor(gctx, ggml_get_name(srcs[i]));
            if (id < 0) {
                LOG_ERR("expert_cache: tensor %s not in %s (split GGUFs are not supported)\n", ggml_get_name(srcs[i]), path.c_str());
                gguf_free(gctx);
                return false;
            }
            tensor_src ts;
            ts.orig             = srcs[i];
            ts.cache            = nullptr;
            ts.file_off         = data_off + gguf_get_tensor_offset(gctx, id);
            ts.bytes_per_expert = srcs[i]->nb[2];
            max_chunk = std::max(max_chunk, ts.bytes_per_expert);
            layers[il].tensors.push_back(ts);
            n_tensors++;
        }
    }
    gguf_free(gctx);
    if (n_tensors == 0) {
        LOG_ERR("expert_cache: no MoE expert tensors found\n");
        return false;
    }

    // cache tensors: same type and matrix shape, `slots` experts deep
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) n_tensors + 1024, nullptr, /*no_alloc*/ true };
    ctx = ggml_init(ip);
    for (auto & L : layers) {
        for (auto & ts : L.tensors) {
            ts.cache = ggml_new_tensor_3d(ctx, ts.orig->type, ts.orig->ne[0], ts.orig->ne[1], p.slots);
            ggml_format_name(ts.cache, "cache.%s", ggml_get_name(ts.orig));
        }
    }
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (!buf) {
        LOG_ERR("expert_cache: cannot allocate cache buffer\n");
        return false;
    }
    bytes_total = ggml_backend_buffer_get_size(buf);
    for (auto & L : layers) {
        for (auto & ts : L.tensors) {
            GGML_ASSERT(ts.cache->nb[2] == ts.bytes_per_expert);
        }
        if (!L.tensors.empty()) {
            L.slot_expert.assign(p.slots, -1);
            L.expert_slot.assign(n_expert, -1);
            L.state.assign(p.slots, FREE);
            L.last_use.assign(p.slots, 0);
            L.used_token.assign(p.slots, 0);
            L.pref_token.assign(p.slots, 0);
        }
    }

#ifdef _WIN32
    if (p.lock_memory) {
        void * base = ggml_backend_buffer_get_base(buf);
        const size_t size = bytes_total;
        // VirtualLock is bounded by the minimum working set, raise it first
        if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), size + (512u << 20), size + (2048u << 20),
                                        QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE) ||
            !VirtualLock(base, size)) {
            LOG_WRN("expert_cache: could not lock %zu MiB of cache memory (error %lu); it may be paged out under pressure\n",
                    size >> 20, GetLastError());
        }
    }
#endif

    for (auto & L : layers) {
        for (auto & ts : L.tensors) {
            llama_expert_cache_register(ts.orig, ts.cache, p.max_tokens);
        }
    }

    for (int i = 0; i < p.io_threads; ++i) {
        threads.emplace_back([this] { io_loop(); });
    }
    LOG_INF("expert_cache: %d slots/layer x %d tensors, %zu MiB locked, %d I/O threads, chunk %zu KiB\n",
            p.slots, n_tensors, bytes_total >> 20, p.io_threads, max_chunk >> 10);
    return true;
}

// caller holds mu
int expert_cache::alloc_slot(layer_state & L, uint64_t token) {
    for (int s = 0; s < p.slots; ++s) {
        if (L.state[s] == FREE) {
            return s;
        }
    }
    // LRU among ready slots not touched this token; then among those only prefetched this token
    int best = -1;
    uint64_t best_use = UINT64_MAX;
    for (int pass = 0; pass < 2 && best < 0; ++pass) {
        for (int s = 0; s < p.slots; ++s) {
            if (L.state[s] != READY || L.used_token[s] == token) {
                continue;
            }
            if (pass == 0 && L.pref_token[s] == token) {
                continue;
            }
            if (L.last_use[s] < best_use) {
                best_use = L.last_use[s];
                best = s;
            }
        }
    }
    if (best >= 0) {
        L.expert_slot[L.slot_expert[best]] = -1;
        L.slot_expert[best] = -1;
        L.state[best] = FREE;
        stats.evictions++;
    }
    return best;
}

// caller holds mu. Misses are needed now: they go to the front of the queue, ahead of prefetches.
void expert_cache::submit(int il, int expert, int slot, bool urgent) {
    auto & L = layers[il];
    L.state[slot]        = LOADING;
    L.slot_expert[slot]  = expert;
    L.expert_slot[expert] = slot;
    if (urgent) {
        jobs.push_front({ il, expert, slot });
    } else {
        jobs.push_back({ il, expert, slot });
    }
    cv_jobs.notify_one();
}

void expert_cache::on_slot_ids(int il, int32_t * ids, int n, uint64_t token) {
    auto & L = layers[il];
    std::unique_lock<std::mutex> lk(mu);
    const int64_t t0 = now_ns();
    std::vector<int32_t> slots(n);
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        int s = L.expert_slot[e];
        if (s < 0) {
            while ((s = alloc_slot(L, token)) < 0) {
                cv.wait(lk); // everything is pinned or in flight; wait for a load to land
            }
            submit(il, e, s, /*urgent*/ true);
            stats.misses++;
        } else if (L.state[s] == LOADING) {
            stats.late_hits++;
        } else {
            stats.hits++;
        }
        L.used_token[s] = token;
        L.last_use[s]   = ++clock;
        slots[i] = s;
    }
    for (int i = 0; i < n; ++i) {
        cv.wait(lk, [&] { return L.state[slots[i]] == READY; });
        ids[i] = slots[i];
    }
    stats.ns_wait += now_ns() - t0;
}

void expert_cache::prefetch(int il, const int32_t * experts, int n, uint64_t token) {
    auto & L = layers[il];
    std::lock_guard<std::mutex> lk(mu);
    for (int i = 0; i < n; ++i) {
        const int e = experts[i];
        int s = L.expert_slot[e];
        if (s >= 0) {
            L.pref_token[s] = token;
            continue;
        }
        s = alloc_slot(L, token);
        if (s < 0) {
            stats.no_slot++;
            continue;
        }
        submit(il, e, s, /*urgent*/ false);
        L.pref_token[s] = token;
        L.last_use[s]   = ++clock;
        stats.prefetched++;
    }
}

bool expert_cache::read_expert(void * handle, char * staging, const layer_state & L, int expert, int slot) {
#ifdef _WIN32
    for (const auto & ts : L.tensors) {
        const uint64_t off = ts.file_off + (uint64_t) expert * ts.bytes_per_expert;
        const uint64_t a0  = off & ~(uint64_t) (SECTOR - 1);
        const uint64_t a1  = (off + ts.bytes_per_expert + SECTOR - 1) & ~(uint64_t) (SECTOR - 1);
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG) a0;
        if (!SetFilePointerEx(handle, li, nullptr, FILE_BEGIN)) {
            return false;
        }
        DWORD got = 0;
        const int64_t r0 = now_ns();
        if (!ReadFile(handle, staging, (DWORD) (a1 - a0), &got, nullptr) || got < (off - a0) + ts.bytes_per_expert) {
            return false;
        }
        const int64_t r1 = now_ns();
        memcpy((char *) ts.cache->data + (size_t) slot * ts.bytes_per_expert, staging + (off - a0), ts.bytes_per_expert);
        const int64_t r2 = now_ns();
        stats.ns_read += r1 - r0;
        stats.ns_copy += r2 - r1;
        stats.bytes_read += ts.bytes_per_expert;
    }
    return true;
#else
    GGML_UNUSED(handle); GGML_UNUSED(staging); GGML_UNUSED(L); GGML_UNUSED(expert); GGML_UNUSED(slot);
    return false;
#endif
}

void expert_cache::bench(int n) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    char * staging = (char *) VirtualAlloc(nullptr, max_chunk + 2 * SECTOR, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    std::vector<int> moe;
    for (int il = 0; il < (int) layers.size(); ++il) {
        if (!layers[il].tensors.empty()) {
            moe.push_back(il);
        }
    }
    uint32_t rng = 12345;
    const long long b0 = stats.bytes_read.load();
    const int64_t t0 = now_ns();
    for (int i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        const int il = moe[(rng >> 8) % moe.size()];
        rng = rng * 1664525u + 1013904223u;
        const int e = (int) ((rng >> 8) % n_expert);
        read_expert(h, staging, layers[il], e, 0);
    }
    const double sec = (now_ns() - t0) / 1e9;
    const double mb  = (stats.bytes_read.load() - b0) / 1e6;
    LOG_INF("expert_cache: io-bench: %d experts, %.0f MB in %.2fs = %.0f MB/s single thread; ReadFile %.2f ms/chunk, memcpy %.2f ms/chunk, chunk %.2f MB\n",
            n, mb, sec, mb / sec, stats.ns_read.load() / 1e6 / (3.0 * n), stats.ns_copy.load() / 1e6 / (3.0 * n), mb / (3.0 * n));
    stats.bytes_read = b0;
    stats.ns_read = 0;
    stats.ns_copy = 0;

    // same handle, plain random reads of fixed sizes: is it the chunk geometry or the call?
    LARGE_INTEGER fsz;
    GetFileSizeEx(h, &fsz);
    char * big = (char *) VirtualAlloc(nullptr, 4u << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    for (size_t len : { (size_t) (1280u << 10), (size_t) (2u << 20), (size_t) (4u << 20) }) {
        const int64_t t1 = now_ns();
        for (int i = 0; i < n; ++i) {
            rng = rng * 1664525u + 1013904223u;
            const uint64_t off = ((uint64_t) rng % (uint64_t) ((fsz.QuadPart - (LONGLONG) len) / SECTOR)) * SECTOR;
            LARGE_INTEGER li;
            li.QuadPart = (LONGLONG) off;
            SetFilePointerEx(h, li, nullptr, FILE_BEGIN);
            DWORD got = 0;
            ReadFile(h, big, (DWORD) len, &got, nullptr);
        }
        const double s2 = (now_ns() - t1) / 1e9;
        LOG_INF("expert_cache: io-bench: random %.2f MB reads x%d: %.0f MB/s, %.2f ms/req\n", len / 1048576.0, n, n * len / 1e6 / s2, s2 * 1e3 / n);
    }
    VirtualFree(big, 0, MEM_RELEASE);
    VirtualFree(staging, 0, MEM_RELEASE);
    CloseHandle(h);
#else
    GGML_UNUSED(n);
#endif
}

void expert_cache::io_loop() {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        LOG_ERR("expert_cache: cannot open %s for unbuffered reads (error %lu)\n", path.c_str(), GetLastError());
        return;
    }
    char * staging = (char *) VirtualAlloc(nullptr, max_chunk + 2 * SECTOR, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv_jobs.wait(lk, [&] { return stop || !jobs.empty(); });
            if (stop) {
                break;
            }
            j = jobs.front();
            jobs.pop_front();
        }
        const int64_t t0 = now_ns();
        const bool ok = read_expert(h, staging, layers[j.il], j.expert, j.slot);
        stats.ns_io += now_ns() - t0;
        if (!ok) {
            LOG_ERR("expert_cache: read failed for layer %d expert %d (error %lu)\n", j.il, j.expert, GetLastError());
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            layers[j.il].state[j.slot] = READY; // on failure the slot holds garbage; the error is logged
        }
        cv.notify_all();
    }
    VirtualFree(staging, 0, MEM_RELEASE);
    CloseHandle(h);
#endif
}
