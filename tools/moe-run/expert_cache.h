// Expert cache: MoE expert weights live in slots in our own (locked) memory, filled by unbuffered
// reads from the GGUF file on an I/O thread pool, instead of being demand-paged from an mmap.
// Pairs with the "ffn_moe_slot_ids" hook in build_moe_ffn (llama_expert_cache_register).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_model;

struct expert_cache_params {
    int  slots          = 32;    // per layer
    int  io_threads     = 2;
    int  max_tokens     = 1;     // batches up to this size use the slots; 0 = slots / experts per token
    bool lock_memory    = true;  // VirtualLock the cache so the OS cannot page it out under pressure
    bool stream_prefill = false; // larger batches stream whole layers through two buffers, else they use the mmap
    std::string bundle;          // expert-bundled sidecar (expert_bundle.h); empty = read the GGUF
};

struct expert_cache_stats {
    std::atomic<long long> hits{0};       // expert was ready in a slot when the router asked
    std::atomic<long long> late_hits{0};  // prefetch was in flight; had to wait for it
    std::atomic<long long> misses{0};     // not present; loaded synchronously
    std::atomic<long long> prefetched{0}; // reads issued ahead of use
    std::atomic<long long> evictions{0};
    std::atomic<long long> no_slot{0};    // prefetch skipped: nothing evictable
    std::atomic<long long> bytes_read{0};
    std::atomic<long long> n_reads{0};    // read requests issued (3 per expert split, 1 bundled)
    std::atomic<long long> ns_wait{0};    // main thread blocked in on_slot_ids
    std::atomic<long long> ns_io{0};      // summed over I/O threads
    std::atomic<long long> ns_read{0};    // inside ReadFile
    std::atomic<long long> ns_copy{0};    // memcpy staging -> slot
    // layer streaming (batches above max_tokens)
    std::atomic<long long> stream_layers{0};  // layers loaded into a stream buffer
    std::atomic<long long> stream_sync{0};    // ... of which on demand, nothing overlapped
    std::atomic<long long> stream_bytes{0};
    std::atomic<long long> ns_stream_wait{0}; // graph thread blocked waiting for a layer
    std::atomic<long long> seeded{0};         // experts copied from a stream buffer into the slots
    std::atomic<long long> ns_seed{0};
};

class expert_cache {
public:
    ~expert_cache();

    bool init(const llama_model * model, const std::string & gguf_path, const expert_cache_params & params);

    // eval-callback entry points (called on the graph thread)
    // "ffn_moe_slot_ids-<il>" of shape [n_used, n_tokens]: picks on_slot_ids or on_stream_ids by batch size
    void on_ids(int il, int32_t * ids, int64_t n_used, int64_t n_tokens, uint64_t token);
    void on_slot_ids(int il, int32_t * ids, int n, uint64_t token);           // resolve, load misses, rewrite ids -> slots
    void on_stream_ids(int il, const int32_t * ids, int n, uint64_t token);  // wait for layer il, start the next, seed the slots
    void prefetch(int il, const int32_t * experts, int n, uint64_t token);   // start loading experts for a later layer

    // read n random experts on the calling thread and report MB/s: isolates raw I/O from inference
    void bench(int n);

    bool has_layer(int il) const { return il >= 0 && il < (int) layers.size() && !layers[il].tensors.empty(); }

    int max_tokens() const { return p.max_tokens; }

    expert_cache_stats stats;
    size_t bytes_total  = 0; // cache memory
    size_t bytes_stream = 0; // stream buffers

private:
    enum slot_state : uint8_t { FREE = 0, LOADING = 1, READY = 2 };

    struct tensor_src {
        const ggml_tensor * orig;
        ggml_tensor *       cache;
        uint64_t            file_off;         // absolute offset of the tensor's data in the GGUF
        size_t              bytes_per_expert; // orig->nb[2]
    };
    struct layer_state {
        std::vector<tensor_src> tensors;      // up, gate, down (or gate_up, down)
        uint64_t                b_base   = 0; // bundled layout: expert 0's bundle
        uint64_t                b_stride = 0; // bundled layout: bytes per expert bundle
        std::vector<int32_t>    slot_expert;  // [slots]  expert in the slot, -1 if free
        std::vector<int32_t>    expert_slot;  // [n_expert] slot of the expert, -1 if absent
        std::vector<uint8_t>    state;        // [slots]
        std::vector<uint64_t>   last_use;     // [slots]  LRU clock
        std::vector<uint64_t>   used_token;   // [slots]  token that last used it (protected from eviction)
        std::vector<uint64_t>   pref_token;   // [slots]  token that prefetched it
    };
    struct job {
        int il, expert, slot;
        // stream chunk (stream >= 0): read [off, off + len) of the GGUF into dst, for stream buffer `stream`
        int      stream = -1;
        uint64_t off    = 0;
        size_t   len    = 0;
        char *   dst    = nullptr;
    };
    // one of the two stream buffers: the layer it holds or is loading, and the chunks still in flight
    struct stream_buf {
        int il      = -1;
        int pending = 0;
    };

    bool init_bundle(const std::string & bundle_path); // validate the sidecar and take its offsets
    bool init_stream();                                // allocate the two layer buffers and register the views
    void prefetch_dense(const llama_model * model);    // read the non-expert weights in big requests, not by faults
    void seed_slots(int il, const int32_t * ids, int n, uint64_t token, std::unique_lock<std::mutex> & lk);
    int  alloc_slot(layer_state & L, uint64_t token); // caller holds mu; -1 if nothing evictable
    void submit(int il, int expert, int slot, bool urgent); // caller holds mu; urgent = front of the queue
    void submit_stream(int il, int b, bool urgent);         // caller holds mu
    void io_loop();
    bool read_expert(void * handle, char * staging, const layer_state & L, int expert, int slot);
    bool read_range(void * handle, char * staging, uint64_t off, size_t len, char * dst);

    expert_cache_params p;
    std::string path;
    std::string gguf_file;                  // stream reads come from the GGUF, also with a sidecar
    std::vector<int> moe_layers;            // MoE layer indices in order
    std::vector<int> moe_ord;               // [n_layer] position in moe_layers, -1 for dense layers
    std::vector<std::vector<ggml_tensor *>> stream_t; // [n_layer] views into a stream buffer
    stream_buf sb[2];
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    int n_expert = 0;
    std::vector<layer_state> layers;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    size_t max_chunk = 0;
    size_t max_read  = 0; // largest single request: a chunk, or a whole bundle
    bool   bundled   = false;
    uint64_t clock = 0;

    std::mutex mu;
    std::condition_variable cv;      // a slot became READY
    std::condition_variable cv_jobs; // work for the I/O threads
    std::deque<job> jobs;
    bool stop = false;
    std::vector<std::thread> threads;
};
