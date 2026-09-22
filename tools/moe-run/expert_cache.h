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
    int  slots       = 32;   // per layer
    int  io_threads  = 2;
    int  max_tokens  = 1;    // batches larger than this use the mmap path (prefill)
    bool lock_memory = true; // VirtualLock the cache so the OS cannot page it out under pressure
};

struct expert_cache_stats {
    std::atomic<long long> hits{0};       // expert was ready in a slot when the router asked
    std::atomic<long long> late_hits{0};  // prefetch was in flight; had to wait for it
    std::atomic<long long> misses{0};     // not present; loaded synchronously
    std::atomic<long long> prefetched{0}; // reads issued ahead of use
    std::atomic<long long> evictions{0};
    std::atomic<long long> no_slot{0};    // prefetch skipped: nothing evictable
    std::atomic<long long> bytes_read{0};
    std::atomic<long long> ns_wait{0};    // main thread blocked in on_slot_ids
    std::atomic<long long> ns_io{0};      // summed over I/O threads
    std::atomic<long long> ns_read{0};    // inside ReadFile
    std::atomic<long long> ns_copy{0};    // memcpy staging -> slot
};

class expert_cache {
public:
    ~expert_cache();

    bool init(const llama_model * model, const std::string & gguf_path, const expert_cache_params & params);

    // eval-callback entry points (called on the graph thread)
    void on_slot_ids(int il, int32_t * ids, int n, uint64_t token);           // resolve, load misses, rewrite ids -> slots
    void prefetch(int il, const int32_t * experts, int n, uint64_t token);   // start loading experts for a later layer

    // read n random experts on the calling thread and report MB/s: isolates raw I/O from inference
    void bench(int n);

    bool has_layer(int il) const { return il >= 0 && il < (int) layers.size() && !layers[il].tensors.empty(); }

    expert_cache_stats stats;
    size_t bytes_total = 0; // cache memory

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
        std::vector<int32_t>    slot_expert;  // [slots]  expert in the slot, -1 if free
        std::vector<int32_t>    expert_slot;  // [n_expert] slot of the expert, -1 if absent
        std::vector<uint8_t>    state;        // [slots]
        std::vector<uint64_t>   last_use;     // [slots]  LRU clock
        std::vector<uint64_t>   used_token;   // [slots]  token that last used it (protected from eviction)
        std::vector<uint64_t>   pref_token;   // [slots]  token that prefetched it
    };
    struct job {
        int il, expert, slot;
    };

    int  alloc_slot(layer_state & L, uint64_t token); // caller holds mu; -1 if nothing evictable
    void submit(int il, int expert, int slot, bool urgent); // caller holds mu; urgent = front of the queue
    void io_loop();
    bool read_expert(void * handle, char * staging, const layer_state & L, int expert, int slot);

    expert_cache_params p;
    std::string path;
    int n_expert = 0;
    std::vector<layer_state> layers;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    size_t max_chunk = 0;
    uint64_t clock = 0;

    std::mutex mu;
    std::condition_variable cv;      // a slot became READY
    std::condition_variable cv_jobs; // work for the I/O threads
    std::deque<job> jobs;
    bool stop = false;
    std::vector<std::thread> threads;
};
