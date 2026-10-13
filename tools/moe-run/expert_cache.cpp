#include "expert_cache.h"
#include "expert_bundle.h"

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
#else
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
constexpr size_t SECTOR = 4096; // unbuffered I/O alignment
constexpr size_t STREAM_CHUNK = 4u << 20; // layer streaming: request size, one job each

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Unbuffered file reads. The handle is an opaque void * (a HANDLE, or an fd cast through intptr_t);
// offset, length and buffer must all be SECTOR aligned.
#ifdef _WIN32
void * open_direct(const std::string & path) {
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    return h == INVALID_HANDLE_VALUE ? nullptr : h;
}
void close_direct(void * h) {
    CloseHandle(h);
}
// bytes read, -1 on error
int64_t read_at(void * h, void * dst, size_t len, uint64_t off) {
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG) off;
    DWORD got = 0;
    if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN) || !ReadFile(h, dst, (DWORD) len, &got, nullptr)) {
        return -1;
    }
    return got;
}
uint64_t file_size(void * h) {
    LARGE_INTEGER sz;
    return GetFileSizeEx(h, &sz) ? (uint64_t) sz.QuadPart : 0;
}
char * alloc_aligned(size_t size) {
    return (char *) VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
void free_aligned(char * p) {
    VirtualFree(p, 0, MEM_RELEASE);
}
std::string os_error() {
    return "error " + std::to_string(GetLastError());
}
#else
void * open_direct(const std::string & path) {
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    return fd < 0 ? nullptr : (void *) (intptr_t) (fd + 1); // +1: fd 0 must not look like a failure
}
int fd_of(void * h) {
    return (int) (intptr_t) h - 1;
}
void close_direct(void * h) {
    close(fd_of(h));
}
int64_t read_at(void * h, void * dst, size_t len, uint64_t off) {
    size_t got = 0;
    while (got < len) { // short only at EOF; a partial O_DIRECT read ends on a sector boundary
        const ssize_t r = pread(fd_of(h), (char *) dst + got, len - got, (off_t) (off + got));
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (r == 0) {
            break;
        }
        got += (size_t) r;
    }
    return (int64_t) got;
}
uint64_t file_size(void * h) {
    struct stat st;
    return fstat(fd_of(h), &st) == 0 ? (uint64_t) st.st_size : 0;
}
char * alloc_aligned(size_t size) {
    void * p = nullptr;
    return posix_memalign(&p, SECTOR, size) == 0 ? (char *) p : nullptr;
}
void free_aligned(char * p) {
    free(p);
}
std::string os_error() {
    return strerror(errno);
}
#endif
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
    if (sbuf) {
        ggml_backend_buffer_free(sbuf);
    }
    if (sctx) {
        ggml_free(sctx);
    }
}

bool expert_cache::init(const llama_model * model, const std::string & gguf_path, const expert_cache_params & params) {
    p         = params;
    path      = gguf_path;
    gguf_file = gguf_path;
    n_expert = (int) model->hparams.n_expert;
    if (p.slots > n_expert) {
        p.slots = n_expert;
    }
    const int n_layer = (int) model->layers.size();
    layers.resize(n_layer);
    moe_ord.assign(n_layer, -1);

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
            max_read  = std::max(max_read,  ts.bytes_per_expert);
            layers[il].tensors.push_back(ts);
            n_tensors++;
        }
        moe_ord[il] = (int) moe_layers.size();
        moe_layers.push_back(il);
    }
    gguf_free(gctx);
    if (n_tensors == 0) {
        LOG_ERR("expert_cache: no MoE expert tensors found\n");
        return false;
    }

    // a batch must never need more distinct experts in a layer than there are slots, or on_slot_ids
    // waits forever for an evictable slot
    const int n_used   = std::max(1, (int) model->hparams.n_expert_used_arr[moe_layers[0]]);
    const int max_safe = std::max(1, p.slots / n_used);
    p.max_tokens = p.max_tokens <= 0 ? max_safe : std::min(p.max_tokens, max_safe);
    if (p.stream_prefill) {
        max_read = std::max(max_read, STREAM_CHUNK);
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

    if (p.stream_prefill) {
        if (!init_stream()) {
            return false;
        }
        // with the experts streamed, the non-expert weights are what the first prompt would fault in 4 KiB at a time
        prefetch_dense(model);
    }

    bool locked = false;
#ifdef _WIN32
    if (p.lock_memory) {
        const size_t size = bytes_total + bytes_stream;
        // VirtualLock is bounded by the minimum working set, raise it first
        if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), size + (512u << 20), size + (2048u << 20),
                                        QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE) ||
            !VirtualLock(ggml_backend_buffer_get_base(buf), bytes_total) ||
            (sbuf && !VirtualLock(ggml_backend_buffer_get_base(sbuf), bytes_stream))) {
            LOG_WRN("expert_cache: could not lock %zu MiB of cache memory (error %lu); it may be paged out under pressure\n",
                    size >> 20, GetLastError());
        } else {
            locked = true;
        }
    }
#else
    if (p.lock_memory) {
        // bounded by RLIMIT_MEMLOCK: raise it with ulimit -l or LimitMEMLOCK=infinity in the systemd scope
        if (mlock(ggml_backend_buffer_get_base(buf), bytes_total) != 0 ||
            (sbuf && mlock(ggml_backend_buffer_get_base(sbuf), bytes_stream) != 0)) {
            LOG_WRN("expert_cache: could not lock %zu MiB of cache memory (%s); it may be paged out under pressure\n",
                    (bytes_total + bytes_stream) >> 20, strerror(errno));
        } else {
            locked = true;
        }
    }
#endif

    if (!p.bundle.empty()) {
        if (!init_bundle(p.bundle)) {
            return false;
        }
        path    = p.bundle; // the I/O threads read whole bundles from here instead
        bundled = true;
    }

    for (int il = 0; il < n_layer; ++il) {
        for (size_t i = 0; i < layers[il].tensors.size(); ++i) {
            const auto & ts = layers[il].tensors[i];
            llama_expert_cache_register(ts.orig, ts.cache, p.max_tokens);
            if (sbuf) {
                llama_expert_cache_register_stream(ts.orig, stream_t[il][i]);
            }
        }
    }

    for (int i = 0; i < p.io_threads; ++i) {
        threads.emplace_back([this] { io_loop(); });
    }
    LOG_INF("expert_cache: %d slots/layer x %d tensors, %zu MiB %s, %d I/O threads, %s layout, request %zu KiB\n",
            p.slots, n_tensors, bytes_total >> 20, locked ? "locked" : "not locked", p.io_threads,
            bundled ? "bundled" : "split", max_read >> 10);
    LOG_INF("expert_cache: batches up to %d tokens use the slots; larger ones %s\n", p.max_tokens,
            sbuf ? "stream whole layers through 2 buffers" : "use the mmap");
    if (sbuf) {
        LOG_INF("expert_cache: stream buffers 2 x %.1f MiB, %zu MiB %s\n",
                bytes_stream / 2 / 1048576.0, bytes_stream >> 20, locked ? "locked" : "not locked");
    }
    return true;
}

// Two buffers, each big enough for every expert of one layer. MoE layer k lives in buffer k % 2, so
// layer k + 1 can load while layer k computes. The views alias: layers 0, 2, 4, ... share buffer 0.
bool expert_cache::init_stream() {
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t layer_bytes = 0;
    size_t n_tensors   = 0;
    for (int il : moe_layers) {
        size_t sz = 0;
        for (const auto & ts : layers[il].tensors) {
            sz += GGML_PAD(ggml_nbytes(ts.orig), align);
            n_tensors++;
        }
        layer_bytes = std::max(layer_bytes, sz);
    }
    sbuf = ggml_backend_buft_alloc_buffer(buft, 2 * layer_bytes);
    if (!sbuf) {
        LOG_ERR("expert_cache: cannot allocate 2 x %zu MiB of stream buffers\n", layer_bytes >> 20);
        return false;
    }
    bytes_stream = ggml_backend_buffer_get_size(sbuf);

    ggml_init_params ip = { ggml_tensor_overhead() * n_tensors + 1024, nullptr, /*no_alloc*/ true };
    sctx = ggml_init(ip);
    char * base = (char *) ggml_backend_buffer_get_base(sbuf);
    stream_t.assign(layers.size(), {});
    for (size_t k = 0; k < moe_layers.size(); ++k) {
        const int il = moe_layers[k];
        char * at = base + (k % 2) * layer_bytes;
        for (const auto & ts : layers[il].tensors) {
            ggml_tensor * t = ggml_new_tensor_3d(sctx, ts.orig->type, ts.orig->ne[0], ts.orig->ne[1], ts.orig->ne[2]);
            ggml_format_name(t, "stream.%s", ggml_get_name(ts.orig));
            if (ggml_backend_tensor_alloc(sbuf, t, at) != GGML_STATUS_SUCCESS) {
                LOG_ERR("expert_cache: cannot place %s in a stream buffer\n", ggml_get_name(t));
                return false;
            }
            GGML_ASSERT(t->nb[2] == ts.bytes_per_expert && ggml_nbytes(t) == ggml_nbytes(ts.orig));
            stream_t[il].push_back(t);
            at += GGML_PAD(ggml_nbytes(ts.orig), align);
        }
    }
    return true;
}

// Read the sidecar header and adopt its offsets. Everything is checked against
// the expert tensors found in the GGUF: a sidecar built for another model, a
// different quant or a different tensor order is rejected rather than served
// as garbage.
bool expert_cache::init_bundle(const std::string & bundle_path) {
    FILE * f = fopen(bundle_path.c_str(), "rb");
    if (!f) {
        LOG_ERR("expert_cache: cannot open the expert sidecar %s\n", bundle_path.c_str());
        return false;
    }
    expert_bundle_header h;
    if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, EXPERT_BUNDLE_MAGIC, 8) != 0) {
        LOG_ERR("expert_cache: %s is not an expert sidecar\n", bundle_path.c_str());
        fclose(f);
        return false;
    }
    if (h.version != EXPERT_BUNDLE_VERSION || h.n_chunk_max != EXPERT_BUNDLE_MAX_CHUNK) {
        LOG_ERR("expert_cache: sidecar version %u is not supported\n", h.version);
        fclose(f);
        return false;
    }
    if ((int) h.n_expert != n_expert) {
        LOG_ERR("expert_cache: sidecar has %u experts, the model has %d\n", h.n_expert, n_expert);
        fclose(f);
        return false;
    }
    std::vector<expert_bundle_layer> recs(h.n_layer);
    if (h.n_layer && fread(recs.data(), sizeof(expert_bundle_layer), recs.size(), f) != recs.size()) {
        LOG_ERR("expert_cache: short read in the sidecar header\n");
        fclose(f);
        return false;
    }
    fclose(f);

    int matched = 0;
    for (const auto & r : recs) {
        if (r.il >= layers.size()) {
            LOG_ERR("expert_cache: sidecar names layer %u, the model has %zu\n", r.il, layers.size());
            return false;
        }
        auto & L = layers[r.il];
        if (L.tensors.empty()) {
            continue;
        }
        if (r.n_chunk != L.tensors.size()) {
            LOG_ERR("expert_cache: layer %u has %zu expert tensors, the sidecar has %u\n",
                    r.il, L.tensors.size(), r.n_chunk);
            return false;
        }
        for (size_t i = 0; i < L.tensors.size(); ++i) {
            if (strcmp(r.chunk[i].name, ggml_get_name(L.tensors[i].orig)) != 0 ||
                r.chunk[i].bytes != L.tensors[i].bytes_per_expert) {
                LOG_ERR("expert_cache: sidecar chunk %zu of layer %u is %s (%llu bytes), expected %s (%zu bytes)\n",
                        i, r.il, r.chunk[i].name, (unsigned long long) r.chunk[i].bytes,
                        ggml_get_name(L.tensors[i].orig), L.tensors[i].bytes_per_expert);
                return false;
            }
        }
        L.b_base   = r.base;
        L.b_stride = r.stride;
        max_read   = std::max(max_read, (size_t) r.stride);
        matched++;
    }
    for (size_t il = 0; il < layers.size(); ++il) {
        if (!layers[il].tensors.empty() && layers[il].b_stride == 0) {
            LOG_ERR("expert_cache: the sidecar has no bundles for layer %zu\n", il);
            return false;
        }
    }
    LOG_INF("expert_cache: expert sidecar %s: %d layers, bundle %.2f MiB\n",
            bundle_path.c_str(), matched, (double) max_read / 1048576.0);
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

void expert_cache::on_ids(int il, int32_t * ids, int64_t n_used, int64_t n_tokens, uint64_t token) {
    if (n_tokens > p.max_tokens) {
        // build_moe_ffn only routes such batches here when the stream views are registered
        on_stream_ids(il, ids, (int) (n_used * n_tokens), token);
        return;
    }
    on_slot_ids(il, ids, (int) (n_used * n_tokens), token);
}

// The stream views hold every expert at its own index, so the ids stay as they are.
void expert_cache::on_stream_ids(int il, const int32_t * ids, int n, uint64_t token) {
    std::unique_lock<std::mutex> lk(mu);
    const int64_t t0 = now_ns();
    const int k = moe_ord[il];
    const int b = k % 2;
    if (sb[b].il != il) {
        // nothing loaded this layer ahead of time (first layer of a batch): load it now
        cv.wait(lk, [&] { return sb[b].pending == 0; });
        submit_stream(il, b, /*urgent*/ true);
        stats.stream_sync++;
    }
    cv.wait(lk, [&] { return sb[b].pending == 0; });

    // the other buffer holds the previous layer, whose expert matmuls ran before this callback.
    // After the last layer this loads layer 0 for the next batch.
    const int kn = (k + 1) % (int) moe_layers.size();
    const int bn = kn % 2;
    if (bn != b && sb[bn].il != moe_layers[kn] && sb[bn].pending == 0) {
        submit_stream(moe_layers[kn], bn, /*urgent*/ false);
    }
    stats.ns_stream_wait += now_ns() - t0;

    seed_slots(il, ids, n, token, lk);
}

// The prompt's experts are in RAM in the stream buffer anyway: copy the ones it used most into the slots,
// so decode does not start from an empty cache. Most used last, so they are also the most recent.
// caller holds mu through lk; the copies run unlocked on slots marked LOADING.
void expert_cache::seed_slots(int il, const int32_t * ids, int n, uint64_t token, std::unique_lock<std::mutex> & lk) {
    auto & L = layers[il];
    const int64_t t0 = now_ns();
    std::vector<int> count(n_expert, 0);
    for (int i = 0; i < n; ++i) {
        count[ids[i]]++;
    }
    std::vector<int> order;
    for (int e = 0; e < n_expert; ++e) {
        if (count[e] > 0) {
            order.push_back(e);
        }
    }
    std::sort(order.begin(), order.end(), [&](int x, int y) { return count[x] > count[y]; });
    if ((int) order.size() > p.slots) {
        order.resize(p.slots);
    }
    std::reverse(order.begin(), order.end());

    std::vector<std::pair<int, int>> copies; // expert, slot
    for (int e : order) {
        int s = L.expert_slot[e];
        if (s < 0) {
            s = alloc_slot(L, token);
            if (s < 0) {
                continue;
            }
            L.state[s]       = LOADING;
            L.slot_expert[s] = e;
            L.expert_slot[e] = s;
            copies.emplace_back(e, s);
        }
        L.used_token[s] = token;
        L.last_use[s]   = ++clock;
    }
    if (copies.empty()) {
        return;
    }
    lk.unlock();
    for (const auto & [e, s] : copies) {
        for (size_t i = 0; i < L.tensors.size(); ++i) {
            const size_t bpe = L.tensors[i].bytes_per_expert;
            memcpy((char *) L.tensors[i].cache->data + (size_t) s * bpe, (const char *) stream_t[il][i]->data + (size_t) e * bpe, bpe);
        }
    }
    lk.lock();
    for (const auto & [e, s] : copies) {
        L.state[s] = READY;
    }
    stats.seeded  += (long long) copies.size();
    stats.ns_seed += now_ns() - t0;
    cv.notify_all();
}

void expert_cache::prefetch_dense(const llama_model * model) {
#ifdef _WIN32
    std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
    size_t bytes = 0;
    for (const auto & [name, t] : model->tensors_by_name) {
        if (name.find("_exps") != std::string::npos || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
            continue;
        }
        ranges.push_back({ t->data, ggml_nbytes(t) });
        bytes += ggml_nbytes(t);
    }
    const int64_t t0 = now_ns();
    if (!ranges.empty() && !PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0)) {
        LOG_WRN("expert_cache: PrefetchVirtualMemory failed (error %lu)\n", GetLastError());
        return;
    }
    LOG_INF("expert_cache: prefetched %zu non-expert tensors, %zu MiB, in %.2fs\n", ranges.size(), bytes >> 20, (now_ns() - t0) / 1e9);
#else
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    size_t n = 0;
    size_t bytes = 0;
    const int64_t t0 = now_ns();
    for (const auto & [name, t] : model->tensors_by_name) {
        if (name.find("_exps") != std::string::npos || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
            continue;
        }
        // madvise wants a page-aligned start; WILLNEED only queues readahead, it does not wait for it
        const uintptr_t a0 = (uintptr_t) t->data & ~(page - 1);
        const uintptr_t a1 = (uintptr_t) t->data + ggml_nbytes(t);
        if (madvise((void *) a0, a1 - a0, MADV_WILLNEED) != 0) {
            LOG_WRN("expert_cache: madvise(WILLNEED) failed for %s (%s)\n", name.c_str(), strerror(errno));
            return;
        }
        n++;
        bytes += ggml_nbytes(t);
    }
    LOG_INF("expert_cache: prefetch queued for %zu non-expert tensors, %zu MiB, in %.2fs\n", n, bytes >> 20, (now_ns() - t0) / 1e9);
#endif
}

// caller holds mu
void expert_cache::submit_stream(int il, int b, bool urgent) {
    const auto & L = layers[il];
    std::vector<job> js;
    for (size_t i = 0; i < L.tensors.size(); ++i) {
        const auto & ts = L.tensors[i];
        const size_t total = ggml_nbytes(ts.orig);
        for (size_t o = 0; o < total; o += STREAM_CHUNK) {
            job j = { il, -1, -1 };
            j.stream = b;
            j.off    = ts.file_off + o;
            j.len    = std::min(STREAM_CHUNK, total - o);
            j.dst    = (char *) stream_t[il][i]->data + o;
            js.push_back(j);
        }
    }
    sb[b].il      = il;
    sb[b].pending = (int) js.size();
    jobs.insert(urgent ? jobs.begin() : jobs.end(), js.begin(), js.end());
    stats.stream_layers++;
    cv_jobs.notify_all();
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
    if (bundled) {
        // the whole expert is one aligned, in-bounds range: one request, then
        // hand the staging buffer out across the slot tensors
        const uint64_t off = L.b_base + (uint64_t) expert * L.b_stride;
        const int64_t r0 = now_ns();
        if (read_at(handle, staging, L.b_stride, off) < (int64_t) L.b_stride) {
            return false;
        }
        const int64_t r1 = now_ns();
        size_t at = 0;
        for (const auto & ts : L.tensors) {
            memcpy((char *) ts.cache->data + (size_t) slot * ts.bytes_per_expert, staging + at, ts.bytes_per_expert);
            at += ts.bytes_per_expert;
            stats.bytes_read += ts.bytes_per_expert;
        }
        const int64_t r2 = now_ns();
        stats.ns_read += r1 - r0;
        stats.ns_copy += r2 - r1;
        stats.n_reads++;
        return true;
    }
    for (const auto & ts : L.tensors) {
        const uint64_t off = ts.file_off + (uint64_t) expert * ts.bytes_per_expert;
        const uint64_t a0  = off & ~(uint64_t) (SECTOR - 1);
        const uint64_t a1  = (off + ts.bytes_per_expert + SECTOR - 1) & ~(uint64_t) (SECTOR - 1);
        const int64_t r0 = now_ns();
        if (read_at(handle, staging, a1 - a0, a0) < (int64_t) ((off - a0) + ts.bytes_per_expert)) {
            return false;
        }
        const int64_t r1 = now_ns();
        memcpy((char *) ts.cache->data + (size_t) slot * ts.bytes_per_expert, staging + (off - a0), ts.bytes_per_expert);
        const int64_t r2 = now_ns();
        stats.ns_read += r1 - r0;
        stats.ns_copy += r2 - r1;
        stats.bytes_read += ts.bytes_per_expert;
        stats.n_reads++;
    }
    return true;
}

bool expert_cache::read_range(void * handle, char * staging, uint64_t off, size_t len, char * dst) {
    const uint64_t a0 = off & ~(uint64_t) (SECTOR - 1);
    const uint64_t a1 = (off + len + SECTOR - 1) & ~(uint64_t) (SECTOR - 1);
    if (read_at(handle, staging, a1 - a0, a0) < (int64_t) ((off - a0) + len)) {
        return false;
    }
    memcpy(dst, staging + (off - a0), len);
    stats.stream_bytes += len;
    return true;
}

void expert_cache::bench(int n) {
    void * h = open_direct(path);
    char * staging = alloc_aligned(max_read + 2 * SECTOR);
    if (!h || !staging) {
        LOG_ERR("expert_cache: io-bench: cannot open %s for unbuffered reads (%s)\n", path.c_str(), os_error().c_str());
        if (h) {
            close_direct(h);
        }
        free_aligned(staging);
        return;
    }
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
    const double reqs = (double) std::max<long long>(1, stats.n_reads.load());
    LOG_INF("expert_cache: io-bench: %d experts, %.0f MB in %.2fs = %.0f MB/s single thread; %.2f requests/expert, read %.2f ms/request, memcpy %.2f ms/request, request %.2f MB\n",
            n, mb, sec, mb / sec, reqs / n, stats.ns_read.load() / 1e6 / reqs,
            stats.ns_copy.load() / 1e6 / reqs, mb / reqs);
    stats.bytes_read = b0;
    stats.ns_read = 0;
    stats.ns_copy = 0;
    stats.n_reads = 0;

    // same handle, plain random reads of fixed sizes: is it the chunk geometry or the call?
    const uint64_t fsz = file_size(h);
    char * big = alloc_aligned(4u << 20);
    for (size_t len : { (size_t) (1280u << 10), (size_t) (2u << 20), (size_t) (4u << 20) }) {
        const int64_t t1 = now_ns();
        for (int i = 0; i < n; ++i) {
            rng = rng * 1664525u + 1013904223u;
            const uint64_t off = ((uint64_t) rng % ((fsz - len) / SECTOR)) * SECTOR;
            read_at(h, big, len, off);
        }
        const double s2 = (now_ns() - t1) / 1e9;
        LOG_INF("expert_cache: io-bench: random %.2f MB reads x%d: %.0f MB/s, %.2f ms/req\n", len / 1048576.0, n, n * len / 1e6 / s2, s2 * 1e3 / n);
    }
    free_aligned(big);
    free_aligned(staging);
    close_direct(h);
}

void expert_cache::io_loop() {
    void * h = open_direct(path);
    if (!h) {
        LOG_ERR("expert_cache: cannot open %s for unbuffered reads (%s)\n", path.c_str(), os_error().c_str());
        return;
    }
    // with a sidecar, h reads bundles; stream chunks always come from the GGUF
    void * hg = h;
    if (sbuf && gguf_file != path) {
        hg = open_direct(gguf_file);
        if (!hg) {
            LOG_ERR("expert_cache: cannot open %s for unbuffered reads (%s)\n", gguf_file.c_str(), os_error().c_str());
            close_direct(h);
            return;
        }
    }
    char * staging = alloc_aligned(max_read + 2 * SECTOR);
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
        if (j.stream >= 0) {
            if (!read_range(hg, staging, j.off, j.len, j.dst)) {
                LOG_ERR("expert_cache: stream read failed for layer %d at %llu (%s)\n",
                        j.il, (unsigned long long) j.off, os_error().c_str());
            }
            bool done;
            {
                std::lock_guard<std::mutex> lk(mu);
                done = --sb[j.stream].pending == 0;
            }
            if (done) {
                cv.notify_all();
            }
            continue;
        }
        const int64_t t0 = now_ns();
        const bool ok = read_expert(h, staging, layers[j.il], j.expert, j.slot);
        stats.ns_io += now_ns() - t0;
        if (!ok) {
            LOG_ERR("expert_cache: read failed for layer %d expert %d (%s)\n", j.il, j.expert, os_error().c_str());
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            layers[j.il].state[j.slot] = READY; // on failure the slot holds garbage; the error is logged
        }
        cv.notify_all();
    }
    free_aligned(staging);
    if (hg != h) {
        close_direct(hg);
    }
    close_direct(h);
}
