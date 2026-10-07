// Expert-bundled layout.
//
// A GGUF stores the MoE experts as three stacked tensors (ffn_up_exps,
// ffn_gate_exps, ffn_down_exps), so one expert is three byte ranges far apart
// in the file and costs three reads. A valid GGUF cannot express the layout we
// want, because every tensor there has to be one contiguous blob and bundling
// interleaves bytes from three of them. So the bundled copy lives in a sidecar
// file next to the model: the GGUF stays untouched and loadable, and the expert
// cache reads whole experts from the sidecar with one request each.
//
// File layout:
//   expert_bundle_header
//   expert_bundle_layer   x header.n_layer
//   padding to EXPERT_BUNDLE_ALIGN
//   bundles: for layer L, expert e, at (layer.base + e * layer.stride), the
//            layer's chunks back to back in the order given by layer.chunk[],
//            zero padded up to layer.stride
//
// base and stride are both multiples of EXPERT_BUNDLE_ALIGN, and the last
// bundle is padded to a full stride, so every read is sector aligned and in
// bounds: the reader can use unbuffered I/O without any fix-up.
#pragma once

#include <cstdint>

#define EXPERT_BUNDLE_MAGIC     "MOEBUND1"
#define EXPERT_BUNDLE_VERSION   1u
#define EXPERT_BUNDLE_ALIGN     4096u
#define EXPERT_BUNDLE_MAX_CHUNK 4

struct expert_bundle_header {
    char     magic[8];      // EXPERT_BUNDLE_MAGIC, not NUL terminated
    uint32_t version;
    uint32_t n_layer;       // number of expert_bundle_layer records that follow
    uint32_t n_expert;
    uint32_t n_chunk_max;   // EXPERT_BUNDLE_MAX_CHUNK
    uint64_t data_offset;   // first bundle in the file
    uint64_t total_bytes;   // size of the finished file
};

struct expert_bundle_chunk {
    uint64_t bytes;         // bytes per expert for this tensor
    char     name[64];      // source tensor name, NUL padded; the reader checks it
};

struct expert_bundle_layer {
    uint32_t il;            // layer index in the model
    uint32_t n_chunk;
    uint64_t base;          // absolute offset of expert 0's bundle
    uint64_t stride;        // bytes from one expert's bundle to the next
    expert_bundle_chunk chunk[EXPERT_BUNDLE_MAX_CHUNK];
};

static inline uint64_t expert_bundle_align(uint64_t v) {
    return (v + EXPERT_BUNDLE_ALIGN - 1) & ~(uint64_t) (EXPERT_BUNDLE_ALIGN - 1);
}
