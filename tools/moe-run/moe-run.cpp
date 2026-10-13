// moe-run: prompt + N-token decode with optional predictive expert prefetch.
//
// At layer L, once the residual entering the FFN is known, the routers of
// layers L+1..L+depth are applied to it (the "stale router": the residual
// changes little per layer, see docs/NOTES.md) and the predicted experts'
// weight ranges are handed to PrefetchVirtualMemory (madvise(WILLNEED) on
// Linux), so the kernel issues large reads for them while layers L..L+depth-1
// compute. Without this the experts fault in page by page when mul_mat_id
// touches them.
//
// Usage: llama-moe-run -m model.gguf -p "prompt" -n 128 -t 10 -lzm on
//            [--prefetch-depth D] [--prefetch-width M] [--prefetch-prefill] [--stats] [--no-resident-check]
//            [--prefetch-threads N] [--touch]
//            [--cache-slots C] [--io-threads N] [--cache-max-tokens T] [--no-lock] [--stream-prefill]
//            [--experts-file model.gguf.experts]
//   --cache-slots > 0 replaces the mmap/page-cache path by an explicit expert cache (expert_cache.h);
//            auto takes as many slots as fit next to the non-expert weights (cgroup memory.max or free RAM).
//            The cache prefetches with depth 1 by default (--prefetch-depth 0 turns it off); --stats and
//            --prefetch-prefill apply to the mmap path only
//   --cache-lfu N: eviction keeps the most used experts, uses decayed by half every N tokens (default 8;
//            0 = LRU); --cache-trace out.txt logs every lookup, prefetch and seed for bench/replay-cache.py
//   --stream-prefill: batches above --cache-max-tokens go through two layer buffers instead of paging
//            experts in from the mmap: each layer gets the experts the predictor guessed for the batch
//            plus the ones it picked. --stream-full F: a batch picking at least this share of a layer's
//            experts loads the next layer whole (default 1 with prefetch, else 0.5)
//   --experts-file reads whole experts from a bundled sidecar (llama-moe-bundle): one request
//            per expert instead of three
//            [--temp T -s SEED] [--json out.json]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"
#include "llama-model.h" // in-tree tool: reads the layer weights directly
#include "expert_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

struct mem_range {
    void * addr;
    size_t len;
};

// ask the OS to read the pages in ahead of use
void prefetch_ranges(const std::vector<mem_range> & rs) {
#ifdef _WIN32
    std::vector<WIN32_MEMORY_RANGE_ENTRY> w;
    w.reserve(rs.size());
    for (const auto & r : rs) {
        w.push_back({ r.addr, r.len });
    }
    PrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) w.size(), w.data(), 0);
#else
    // WILLNEED queues readahead on the mapped file and returns; the start must be page-aligned
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    for (const auto & r : rs) {
        const uintptr_t a0 = (uintptr_t) r.addr & ~(page - 1);
        madvise((void *) a0, (uintptr_t) r.addr + r.len - a0, MADV_WILLNEED);
    }
#endif
}

// first and middle page mapped in (Windows: in the working set; Linux: in the page cache)
bool resident(const mem_range & r) {
#ifdef _WIN32
    PSAPI_WORKING_SET_EX_INFORMATION info[2];
    info[0].VirtualAddress = r.addr;
    info[1].VirtualAddress = (char *) r.addr + ((r.len / 2) & ~(SIZE_T) 4095);
    if (!QueryWorkingSetEx(GetCurrentProcess(), info, sizeof(info))) {
        return false;
    }
    return info[0].VirtualAttributes.Valid && info[1].VirtualAttributes.Valid;
#else
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    for (const uintptr_t a : { (uintptr_t) r.addr, (uintptr_t) r.addr + r.len / 2 }) {
        unsigned char v = 0;
        if (mincore((void *) (a & ~(page - 1)), 1, &v) != 0 || !(v & 1)) {
            return false;
        }
    }
    return true;
#endif
}

struct layer_weights {
    const ggml_tensor * exps[3] = {}; // up, gate, down (or gate_up, down)
    int                 n_exps  = 0;
};

// mmap mode: the stale router's picks go to PrefetchVirtualMemory / madvise. With --cache-slots the
// expert cache runs its own predictor (expert_cache::on_router_input) and this one only names the hook.
struct prefetcher {
    expert_router router;
    int  n_layer  = 0;
    int  n_expert = 0;
    int  k        = 0;   // experts used per token

    int  depth   = 0;     // layers ahead to prefetch (0 = off)
    int  width   = 0;     // experts per layer to prefetch (>= k)
    bool prefill = false; // also prefetch whole layers during batched prompt processing
    bool stats   = false;

    std::vector<layer_weights> layers;
    std::vector<std::vector<int32_t>> predicted; // union of this token's predictions per layer
    std::vector<uint64_t> predicted_token;       // token the prediction belongs to

    expert_cache * ec = nullptr; // explicit cache mode: the hook feeds ec->on_router_input
    uint64_t token = 0;          // decode step, set by the main loop

    // stats
    long long n_actual = 0, n_hit = 0, n_pred_layers = 0, n_prefetch_calls = 0, bytes_prefetched = 0;
    double    predict_ms = 0;

    std::vector<float>   xn, logits;
    std::vector<int32_t> order;
    std::vector<mem_range> ranges;

    void init(const llama_model * model) {
        if (!router.init(model) && depth > 0) {
            // unknown router input: the cache still works, prediction cannot
            LOG_WRN("prefetcher: router input unknown for this architecture, prefetch disabled\n");
            depth = 0;
        }
        n_layer  = router.n_layer;
        n_expert = router.n_expert;
        k        = router.k;
        if (width < k) {
            width = k;
        }
        layers.resize(n_layer);
        predicted.assign(n_layer, {});
        predicted_token.assign(n_layer, UINT64_MAX);
        int n_moe = 0;
        for (int il = 0; il < n_layer; ++il) {
            const auto & L = model->layers[il];
            auto & w = layers[il];
            if (!L.ffn_gate_inp || !L.ffn_down_exps) {
                continue; // dense layer
            }
            if (L.ffn_gate_up_exps) {
                w.exps[w.n_exps++] = L.ffn_gate_up_exps;
            } else {
                w.exps[w.n_exps++] = L.ffn_up_exps;
                if (L.ffn_gate_exps) {
                    w.exps[w.n_exps++] = L.ffn_gate_exps;
                }
            }
            w.exps[w.n_exps++] = L.ffn_down_exps;
            n_moe++;
        }
        xn.resize(router.n_embd);
        logits.resize(n_expert);
        order.resize(n_expert);
        LOG_INF("prefetcher: %d MoE layers, %d experts, top-%d, gating %d, depth %d, width %d, eps %g\n",
                n_moe, n_expert, k, router.gating, depth, width, router.eps);
        for (int il = 0; il < n_layer; ++il) {
            if (layers[il].n_exps) {
                const auto * t = layers[il].exps[0];
                LOG_INF("prefetcher: expert bytes per tensor: %zu KiB x %d tensors\n",
                        (size_t) t->nb[2] / 1024, layers[il].n_exps);
                break;
            }
        }
    }

    struct job {
        int layer;
        std::vector<mem_range> ranges;
    };
    // PrefetchVirtualMemory walks every page of every range in the caller's thread (~0.16 ms/MB
    // even when resident), so it runs on a worker: the compute threads only pay for the routing
    // math and an enqueue. Stale jobs (layer already computed) are dropped, and ranges whose
    // first and middle page are already in the working set are skipped.
    struct worker {
        std::vector<std::thread> ths;
        std::mutex              mu;
        std::condition_variable cv;
        std::deque<job>         q;
        bool                    stop = false;
        std::atomic<int>        current_layer{-1};
        bool                    check_resident = true;
        int                     n_threads = 1;
        // optional second stage: touch one byte per page so the pages land in the working set
        // (a soft fault each) on this thread instead of on the compute threads
        bool                    touch = false;
        std::thread             toucher;
        std::mutex              tmu;
        std::condition_variable tcv;
        std::deque<job>         tq;
        // stats
        std::atomic<long long>  bytes_issued{0}, bytes_skipped{0}, calls{0}, stale{0};
        std::atomic<long long>  ns_in_calls{0}, ns_touch{0}, bytes_touched{0};

        void start() {
            for (int i = 0; i < n_threads; ++i) {
                ths.emplace_back([this] { run(); });
            }
            if (touch) {
                toucher = std::thread([this] { run_touch(); });
            }
        }
        void finish() {
            { std::lock_guard<std::mutex> lk(mu); stop = true; }
            cv.notify_all();
            for (auto & t : ths) if (t.joinable()) t.join();
            { std::lock_guard<std::mutex> lk(tmu); }
            tcv.notify_all();
            if (toucher.joinable()) toucher.join();
        }
        void push(job && j) {
            { std::lock_guard<std::mutex> lk(mu); q.push_back(std::move(j)); }
            cv.notify_one();
        }
        void run_touch() {
            for (;;) {
                job j;
                {
                    std::unique_lock<std::mutex> lk(tmu);
                    tcv.wait(lk, [&] { return stop || !tq.empty(); });
                    if (tq.empty()) return;
                    j = std::move(tq.front());
                    tq.pop_front();
                }
                if (j.layer <= current_layer.load()) { continue; }
                const auto t0 = std::chrono::steady_clock::now();
                volatile char sink = 0;
                for (const auto & r : j.ranges) {
                    const char * base = (const char *) r.addr;
                    for (size_t off = 0; off < r.len; off += 4096) {
                        sink += base[off];
                    }
                    bytes_touched += r.len;
                }
                (void) sink;
                ns_touch += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            }
        }
        void run() {
            std::vector<mem_range> todo;
            for (;;) {
                job j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return stop || !q.empty(); });
                    if (q.empty()) return;
                    j = std::move(q.front());
                    q.pop_front();
                }
                if (j.layer <= current_layer.load()) { stale++; continue; }
                todo.clear();
                for (const auto & r : j.ranges) {
                    if (check_resident && resident(r)) { bytes_skipped += r.len; continue; }
                    todo.push_back(r);
                }
                if (todo.empty()) continue;
                const auto t0 = std::chrono::steady_clock::now();
                prefetch_ranges(todo);
                ns_in_calls += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                for (const auto & r : todo) bytes_issued += r.len;
                calls++;
                if (touch) {
                    job t; t.layer = j.layer; t.ranges = todo;
                    { std::lock_guard<std::mutex> lk(tmu); tq.push_back(std::move(t)); }
                    tcv.notify_one();
                }
            }
        }
    } wk;

    void issue(int il, const int32_t * ids, int n) {
        const auto & w = layers[il];
        job j;
        j.layer = il;
        j.ranges.reserve((size_t) n * w.n_exps);
        for (int i = 0; i < n; ++i) {
            for (int t = 0; t < w.n_exps; ++t) {
                const ggml_tensor * e = w.exps[t];
                j.ranges.push_back({ (char *) e->data + (size_t) ids[i] * e->nb[2], e->nb[2] });
                bytes_prefetched += e->nb[2];
            }
        }
        wk.push(std::move(j));
        n_prefetch_calls++;
    }

    // whole-layer prefetch for batched prompt processing (every expert gets used anyway)
    void issue_layer(int il) {
        const auto & w = layers[il];
        ranges.clear();
        for (int t = 0; t < w.n_exps; ++t) {
            const ggml_tensor * e = w.exps[t];
            ranges.push_back({ e->data, ggml_nbytes(e) });
            bytes_prefetched += ggml_nbytes(e);
        }
        job j;
        j.layer = il;
        j.ranges = ranges;
        wk.push(std::move(j));
        n_prefetch_calls++;
    }

    // the residual entering layer il's FFN is ready: predict and prefetch ahead
    void on_ffn_inp(int il, const float * x, int n_tokens) {
        if (depth <= 0) {
            return;
        }
        wk.current_layer.store(il);
        if (n_tokens > 1) {
            if (prefill) {
                for (int d = 1; d <= depth; ++d) {
                    if (il + d < n_layer && router.layers[il + d].is_moe) {
                        issue_layer(il + d);
                    }
                }
            }
            return;
        }
        predict(il, x, token);
    }

    void predict(int il, const float * x, uint64_t tok) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int d = 1; d <= depth; ++d) {
            const int lp = il + d;
            if (lp >= n_layer || !router.layers[lp].is_moe) {
                continue;
            }
            router.route(lp, x, xn.data(), logits.data());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + width, order.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            // prefetch what this prediction adds; the union over depths is what was prefetched for lp
            auto & pred = predicted[lp];
            if (predicted_token[lp] != tok) { // first prediction for lp in this token
                pred.clear();
                predicted_token[lp] = tok;
            }
            std::vector<int32_t> fresh;
            for (int i = 0; i < width; ++i) {
                if (std::find(pred.begin(), pred.end(), order[i]) == pred.end()) {
                    pred.push_back(order[i]);
                    fresh.push_back(order[i]);
                }
            }
            if (!fresh.empty()) {
                issue(lp, fresh.data(), (int) fresh.size());
            }
            n_pred_layers++;
        }
        predict_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // the router's real choice for layer il: score the prediction that was made for it
    void on_topk(int il, const int32_t * ids, int n_tokens) {
        if (n_tokens != 1 || predicted[il].empty() || predicted_token[il] != token) {
            return;
        }
        for (int i = 0; i < k; ++i) {
            n_actual++;
            if (std::find(predicted[il].begin(), predicted[il].end(), ids[i]) != predicted[il].end()) {
                n_hit++;
            }
        }
    }
};

// "<base>-<layer>" with base in {<inp_name>, ffn_moe_topk, ffn_moe_slot_ids}; rejects derived names like "... (reshaped)"
bool classify(const char * name, const std::string & inp_name, int & kind, int & layer) {
    const char * dash = strrchr(name, '-');
    if (!dash || dash[1] == 0) {
        return false;
    }
    const std::string base(name, dash - name);
    if (base == inp_name) {
        kind = 0;
    } else if (base == "ffn_moe_topk") {
        kind = 1;
    } else if (base == "ffn_moe_slot_ids") {
        kind = 2;
    } else {
        return false;
    }
    for (const char * c = dash + 1; *c; ++c) {
        if (*c < '0' || *c > '9') {
            return false;
        }
    }
    layer = atoi(dash + 1);
    return true;
}

bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * pf = (prefetcher *) user_data;
    int kind, layer;
    if (!classify(t->name, pf->router.inp_name, kind, layer)) {
        return false;
    }
    if (ask) {
        switch (kind) {
            case 0:  return pf->depth > 0 || (pf->ec && pf->ec->router_input());
            case 1:  return pf->stats;
            default: return pf->ec != nullptr;
        }
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(t->buffer));
    const int n_tokens = (int) (t->ne[1] * t->ne[2] * t->ne[3]);
    if (kind == 0) {
        if (pf->ec) {
            if (ggml_is_contiguous(t)) {
                pf->ec->on_router_input(layer, (const float *) t->data, n_tokens, pf->token);
            }
        } else {
            pf->on_ffn_inp(layer, (const float *) t->data, n_tokens);
        }
    } else if (kind == 1) {
        // topk is a strided view; a single token is one dense row
        pf->on_topk(layer, (const int32_t *) t->data, n_tokens);
    } else {
        // contiguous copy of the selected expert ids [n_used, n_tokens]: resolved to cache slots in place,
        // or for a large batch with --stream-prefill, a wait for the layer to be streamed in
        pf->ec->on_ids(layer, (int32_t *) t->data, t->ne[0], t->ne[1], pf->token);
    }
    return true;
}

// Windows only reports hard + soft faults together (major stays 0); soft faults are cheap, so on
// Linux the major ones (read from disk) are counted on their own
struct fault_count {
    size_t total = 0;
    size_t major = 0;
};

fault_count page_faults() {
    fault_count f;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        f.total = pmc.PageFaultCount;
    }
#else
    rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        f.total = (size_t) (ru.ru_majflt + ru.ru_minflt);
        f.major = (size_t) ru.ru_majflt;
    }
#endif
    return f;
}

} // namespace

int main(int argc, char ** argv) {
    prefetcher pf;
    std::string json_out;
    bool no_resident_check = false;
    int  prefetch_threads = 1;
    bool touch = false;
    expert_cache_params cp;
    cp.slots = 0; // off unless --cache-slots
    int io_bench = 0;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--prefetch-depth" && i + 1 < argc) {
            pf.depth = cp.predict_depth = atoi(argv[++i]);
        } else if (a == "--prefetch-width" && i + 1 < argc) {
            pf.width = cp.predict_width = atoi(argv[++i]);
        } else if (a == "--prefetch-prefill") {
            pf.prefill = true;
        } else if (a == "--stats") {
            pf.stats = true;
        } else if (a == "--no-resident-check") {
            no_resident_check = true;
        } else if (a == "--prefetch-threads" && i + 1 < argc) {
            prefetch_threads = atoi(argv[++i]);
        } else if (a == "--touch") {
            touch = true;
        } else if (a == "--cache-slots" && i + 1 < argc) {
            const std::string v = argv[++i];
            cp.slots = v == "auto" ? expert_cache_params::SLOTS_AUTO : atoi(v.c_str());
        } else if (a == "--io-threads" && i + 1 < argc) {
            cp.io_threads = atoi(argv[++i]);
        } else if (a == "--cache-max-tokens" && i + 1 < argc) {
            cp.max_tokens = atoi(argv[++i]);
        } else if (a == "--stream-prefill") {
            cp.stream_prefill = true;
        } else if (a == "--stream-full" && i + 1 < argc) {
            cp.stream_full = (float) atof(argv[++i]);
        } else if (a == "--no-lock") {
            cp.lock_memory = false;
        } else if (a == "--experts-file" && i + 1 < argc) {
            cp.bundle = argv[++i];
        } else if (a == "--cache-lfu" && i + 1 < argc) {
            cp.lfu_half_life = atoi(argv[++i]);
        } else if (a == "--cache-trace" && i + 1 < argc) {
            cp.trace = argv[++i];
        } else if (a == "--io-bench" && i + 1 < argc) {
            io_bench = atoi(argv[++i]);
        } else if (a == "--json" && i + 1 < argc) {
            json_out = argv[++i];
        } else {
            args.push_back(argv[i]);
        }
    }

    common_params params;
    params.n_predict = 64;
    params.sampling.temp = 0.0f;
    // mark temp as user-set: a GGUF can carry general.sampling.temp (Gemma 4: 1.0), which would replace
    // our greedy default at load and make the tokens hash differ from run to run
    params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    if (cp.slots != 0) {
        pf.depth = 0; // the cache predicts on its own thread (expert_cache::on_router_input)
    }
    if (pf.depth > 0 || pf.stats || cp.slots != 0) {
        params.cb_eval           = cb_eval;
        params.cb_eval_user_data = &pf;
    }
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }
    pf.init(model);
    expert_cache ec;
    if (cp.slots != 0) {
        if (!ec.init(model, params.model.path, cp)) {
            return 1;
        }
        if (io_bench > 0) {
            ec.bench(io_bench);
        }
        pf.ec = &ec;
    }
    pf.wk.check_resident = !no_resident_check;
    pf.wk.n_threads      = std::max(1, prefetch_threads);
    pf.wk.touch          = touch;
    if (pf.depth > 0 && !pf.ec) {
        pf.wk.start();
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }

    using clock = std::chrono::steady_clock;
    const fault_count pf0 = page_faults();
    const auto tp0 = clock::now();
    pf.token = 0;
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("%s: prefill failed\n", __func__);
        return 1;
    }
    const double prefill_s = std::chrono::duration<double>(clock::now() - tp0).count();
    const fault_count pf1 = page_faults();

    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.sampling.temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.sampling.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    std::string text;
    uint64_t tok_hash = 1469598103934665603ull; // FNV-1a over the generated token ids
    int n_gen = 0;
    const auto tg0 = clock::now();
    for (int i = 0; i < params.n_predict; ++i) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, tok);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        text += common_token_to_piece(ctx, tok);
        tok_hash = (tok_hash ^ (uint64_t) (uint32_t) tok) * 1099511628211ull;
        pf.token = (uint64_t) i + 1;
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("%s: decode failed at token %d\n", __func__, i);
            return 1;
        }
        n_gen++;
    }
    const double gen_s = std::chrono::duration<double>(clock::now() - tg0).count();
    const fault_count pf2 = page_faults();
    pf.wk.finish();
    const double wk_ms  = pf.wk.ns_in_calls.load() / 1e6;
    const double wk_mb  = pf.wk.bytes_issued.load() / 1e6;
    const double skp_mb = pf.wk.bytes_skipped.load() / 1e6;
    const double tch_ms = pf.wk.ns_touch.load() / 1e6;
    const double tch_mb = pf.wk.bytes_touched.load() / 1e6;
    const long long stale = pf.wk.stale.load();

    const double hit = pf.n_actual ? (double) pf.n_hit / pf.n_actual : 0.0;
    const double faults_per_token = n_gen ? (double) (pf2.total - pf1.total) / n_gen : 0.0;
    const double majflt_per_token = n_gen ? (double) (pf2.major - pf1.major) / n_gen : 0.0;
    const double gen_tps = n_gen ? n_gen / gen_s : 0.0;
    LOG_INF("\n%s: prefill %zu tokens in %.2fs (%.1f tok/s), %zu page faults (%zu major)\n", __func__,
            tokens.size(), prefill_s, tokens.size() / prefill_s, pf1.total - pf0.total, pf1.major - pf0.major);
    LOG_INF("%s: decode %d tokens in %.2fs (%.2f tok/s), %.0f page faults/token (%.1f major)\n", __func__,
            n_gen, gen_s, gen_tps, faults_per_token, majflt_per_token);
    if (pf.depth > 0) {
        LOG_INF("%s: prefetch depth %d width %d: predict %.2f ms/token, %.1f MB/token requested, hit rate %.3f (%lld/%lld)\n",
                __func__, pf.depth, pf.width, n_gen ? pf.predict_ms / n_gen : 0.0,
                n_gen ? pf.bytes_prefetched / 1e6 / n_gen : 0.0, hit, pf.n_hit, pf.n_actual);
        LOG_INF("%s: worker x%d: %.1f MB/token issued, %.1f MB/token skipped (resident), %lld stale jobs, %.1f ms/token inside the prefetch call; toucher: %.1f MB/token in %.1f ms/token\n",
                __func__, pf.wk.n_threads, n_gen ? wk_mb / n_gen : 0.0, n_gen ? skp_mb / n_gen : 0.0, stale, n_gen ? wk_ms / n_gen : 0.0,
                n_gen ? tch_mb / n_gen : 0.0, n_gen ? tch_ms / n_gen : 0.0);
    }
    if (pf.ec) {
        const auto & cs = ec.stats;
        const double per = n_gen ? 1.0 / n_gen : 0.0;
        if (ec.predict_depth() > 0) {
            LOG_INF("%s: prefetch depth %d width %d: predict %.2f ms/token on the predictor thread\n", __func__,
                    ec.predict_depth(), ec.predict_width(), cs.ns_predict.load() / 1e6 * per);
        }
        LOG_INF("%s: cache: %zu MiB, %s layout, per token: hits %.1f, late hits %.1f, misses %.1f, prefetched %.1f, evictions %.1f, no-slot %.1f; "
                "read %.1f MB/token in %.1f requests (%.2f MB each), wait %.1f ms/token, I/O %.0f MB/s per thread\n",
                __func__, ec.bytes_total >> 20, cp.bundle.empty() ? "split" : "bundled",
                cs.hits.load() * per, cs.late_hits.load() * per, cs.misses.load() * per,
                cs.prefetched.load() * per, cs.evictions.load() * per, cs.no_slot.load() * per,
                cs.bytes_read.load() / 1e6 * per, cs.n_reads.load() * per,
                cs.n_reads.load() ? cs.bytes_read.load() / 1e6 / cs.n_reads.load() : 0.0,
                cs.ns_wait.load() / 1e6 * per,
                cs.ns_io.load() ? cs.bytes_read.load() / (cs.ns_io.load() / 1e9) / 1e6 : 0.0);
    }
    if (pf.ec && ec.bytes_stream) {
        const auto & cs = ec.stats;
        LOG_INF("%s: stream: %zu MiB buffers, %lld layers loaded whole, %lld partly; experts %lld predicted, %lld read late, %lld from slots; %.0f MB read, graph waited %.2fs in total; %lld experts seeded into the slots in %.2fs\n",
                __func__, ec.bytes_stream >> 20, cs.stream_layers.load(), cs.stream_partial.load(),
                cs.stream_predicted.load(), cs.stream_late.load(), cs.stream_from_slots.load(),
                cs.stream_bytes.load() / 1e6, cs.ns_stream_wait.load() / 1e9, cs.seeded.load(), cs.ns_seed.load() / 1e9);
    }
    for (auto & c : text) { if (c == '\n' || c == '\r') c = ' '; }
    LOG_INF("%s: tokens hash %016llx, text: %.160s\n", __func__, (unsigned long long) tok_hash, text.c_str());

    if (!json_out.empty()) {
        if (FILE * f = fopen(json_out.c_str(), "wb")) {
            fprintf(f, "{\"n_prompt\": %zu, \"prefill_s\": %.4f, \"n_gen\": %d, \"gen_s\": %.4f, \"gen_tps\": %.4f, "
                       "\"prefill_faults\": %zu, \"prefill_majflt\": %zu, \"faults_per_token\": %.1f, \"majflt_per_token\": %.2f, \"prefetch_depth\": %d, \"prefetch_width\": %d, "
                       "\"prefetch_mb_per_token\": %.2f, \"hit_rate\": %.4f, \"predict_ms_per_token\": %.3f, "
                       "\"issued_mb_per_token\": %.2f, \"skipped_mb_per_token\": %.2f, \"stale_jobs\": %lld, \"worker_ms_per_token\": %.2f, "
                       "\"prefetch_threads\": %d, \"touch\": %d, \"touch_ms_per_token\": %.2f, "
                       "\"cache_slots\": %d, \"cache_hits\": %.2f, \"cache_late_hits\": %.2f, \"cache_misses\": %.2f, "
                       "\"cache_read_mb_per_token\": %.2f, \"cache_wait_ms_per_token\": %.2f, \"cache_io_mbps\": %.0f, "
                       "\"cache_reads_per_token\": %.2f, \"cache_layout\": \"%s\"}\n",
                    tokens.size(), prefill_s, n_gen, gen_s, gen_tps, pf1.total - pf0.total, pf1.major - pf0.major,
                    faults_per_token, majflt_per_token, pf.ec ? ec.predict_depth() : pf.depth, pf.ec ? ec.predict_width() : pf.width,
                    n_gen ? pf.bytes_prefetched / 1e6 / n_gen : 0.0, hit,
                    n_gen ? (pf.ec ? ec.stats.ns_predict.load() / 1e6 : pf.predict_ms) / n_gen : 0.0,
                    n_gen ? wk_mb / n_gen : 0.0, n_gen ? skp_mb / n_gen : 0.0, stale, n_gen ? wk_ms / n_gen : 0.0,
                    prefetch_threads, touch ? 1 : 0, n_gen ? tch_ms / n_gen : 0.0,
                    pf.ec ? ec.slots() : 0, n_gen ? (double) ec.stats.hits.load() / n_gen : 0.0, n_gen ? (double) ec.stats.late_hits.load() / n_gen : 0.0,
                    n_gen ? (double) ec.stats.misses.load() / n_gen : 0.0, n_gen ? ec.stats.bytes_read.load() / 1e6 / n_gen : 0.0,
                    n_gen ? ec.stats.ns_wait.load() / 1e6 / n_gen : 0.0,
                    ec.stats.ns_io.load() ? ec.stats.bytes_read.load() / (ec.stats.ns_io.load() / 1e9) / 1e6 : 0.0,
                    n_gen ? (double) ec.stats.n_reads.load() / n_gen : 0.0,
                    cp.bundle.empty() ? "split" : "bundled");
            fclose(f);
        }
    }

    llama_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
