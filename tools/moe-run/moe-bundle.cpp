// moe-bundle: write a sidecar file holding the model's MoE expert weights in
// expert-bundled order, so that fetching one expert is one read instead of
// three. See expert_bundle.h for the format and why it is not a GGUF.
//
// Usage: llama-moe-bundle -m model.gguf [-o model.experts] [--verify]
//
// The GGUF is not modified. Point llama-moe-run at the result with
// --experts-file; it falls back to reading the GGUF when the flag is absent.

#include "expert_bundle.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#define fseek64 _fseeki64
#else
#define fseek64 fseeko
#endif

namespace {

// the three (or two) expert tensors of one layer, in the order the expert cache
// walks them: the merged gate_up tensor if the model has one, else up then
// gate, and down last
const char * CHUNK_ORDER[] = {
    "ffn_gate_up_exps.weight",
    "ffn_up_exps.weight",
    "ffn_gate_exps.weight",
    "ffn_down_exps.weight",
};

struct src_chunk {
    std::string name;
    uint64_t    file_off;         // absolute offset in the GGUF
    uint64_t    bytes_per_expert; // nb[2]
    int64_t     n_expert;
};

bool parse_name(const char * name, int & il, std::string & suffix) {
    int n = 0;
    if (sscanf(name, "blk.%d.%n", &il, &n) != 1 || n <= 0) {
        return false;
    }
    suffix = name + n;
    return true;
}

int order_of(const std::string & suffix) {
    for (int i = 0; i < (int) (sizeof(CHUNK_ORDER) / sizeof(*CHUNK_ORDER)); ++i) {
        if (suffix == CHUNK_ORDER[i]) {
            return i;
        }
    }
    return -1;
}

void die(const char * msg, const char * arg = nullptr) {
    fprintf(stderr, "moe-bundle: %s%s%s\n", msg, arg ? ": " : "", arg ? arg : "");
    exit(1);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model, out;
    bool verify = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) {
            model = argv[++i];
        } else if ((a == "-o" || a == "--out") && i + 1 < argc) {
            out = argv[++i];
        } else if (a == "--verify") {
            verify = true;
        } else {
            fprintf(stderr, "usage: %s -m model.gguf [-o model.experts] [--verify]\n", argv[0]);
            return 1;
        }
    }
    if (model.empty()) {
        die("no model given (-m)");
    }
    if (out.empty()) {
        out = model + ".experts";
    }

    ggml_context * mctx = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &mctx };
    gguf_context * gctx = gguf_init_from_file(model.c_str(), gp);
    if (!gctx) {
        die("cannot read GGUF metadata from", model.c_str());
    }
    const uint64_t data_off = gguf_get_data_offset(gctx);

    // collect the expert tensors, grouped by layer and ordered as above
    std::map<int, std::map<int, src_chunk>> layers;
    for (ggml_tensor * t = ggml_get_first_tensor(mctx); t; t = ggml_get_next_tensor(mctx, t)) {
        int il = 0;
        std::string suffix;
        if (!parse_name(ggml_get_name(t), il, suffix)) {
            continue;
        }
        const int ord = order_of(suffix);
        if (ord < 0) {
            continue;
        }
        if (ggml_n_dims(t) != 3) {
            die("expert tensor is not 3d", ggml_get_name(t));
        }
        const int64_t id = gguf_find_tensor(gctx, ggml_get_name(t));
        if (id < 0) {
            die("tensor missing from the GGUF index", ggml_get_name(t));
        }
        src_chunk c;
        c.name             = ggml_get_name(t);
        c.file_off         = data_off + gguf_get_tensor_offset(gctx, id);
        c.bytes_per_expert = t->nb[2];
        c.n_expert         = t->ne[2];
        layers[il][ord]    = c;
    }
    if (layers.empty()) {
        die("no MoE expert tensors (ffn_*_exps.weight) in", model.c_str());
    }

    const int64_t n_expert = layers.begin()->second.begin()->second.n_expert;

    // header + one record per layer, then the bundles
    std::vector<expert_bundle_layer> recs;
    uint64_t cursor = 0; // filled in once the header size is known
    for (const auto & [il, chunks] : layers) {
        if (chunks.size() > EXPERT_BUNDLE_MAX_CHUNK) {
            die("too many expert tensors in one layer");
        }
        expert_bundle_layer r = {};
        r.il      = (uint32_t) il;
        r.n_chunk = (uint32_t) chunks.size();
        uint64_t sum = 0;
        int k = 0;
        for (const auto & [ord, c] : chunks) {
            if (c.n_expert != n_expert) {
                die("layers disagree on the expert count", c.name.c_str());
            }
            if (c.name.size() >= sizeof(r.chunk[k].name)) {
                die("tensor name too long for the header", c.name.c_str());
            }
            r.chunk[k].bytes = c.bytes_per_expert;
            snprintf(r.chunk[k].name, sizeof(r.chunk[k].name), "%s", c.name.c_str());
            sum += c.bytes_per_expert;
            k++;
        }
        r.stride = expert_bundle_align(sum);
        recs.push_back(r);
    }

    expert_bundle_header h = {};
    memcpy(h.magic, EXPERT_BUNDLE_MAGIC, 8);
    h.version     = EXPERT_BUNDLE_VERSION;
    h.n_layer     = (uint32_t) recs.size();
    h.n_expert    = (uint32_t) n_expert;
    h.n_chunk_max = EXPERT_BUNDLE_MAX_CHUNK;
    h.data_offset = expert_bundle_align(sizeof(h) + recs.size() * sizeof(expert_bundle_layer));

    cursor = h.data_offset;
    for (auto & r : recs) {
        r.base = cursor;
        cursor += r.stride * (uint64_t) n_expert;
    }
    h.total_bytes = cursor;

    printf("moe-bundle: %s\n", model.c_str());
    printf("  %u MoE layers, %u experts, bundle stride %.2f MiB, sidecar %.2f GiB\n",
           h.n_layer, h.n_expert, recs[0].stride / 1048576.0, h.total_bytes / 1073741824.0);

    FILE * in = fopen(model.c_str(), "rb");
    if (!in) {
        die("cannot open the model for reading", model.c_str());
    }
    FILE * fo = fopen(out.c_str(), "wb");
    if (!fo) {
        die("cannot open the sidecar for writing", out.c_str());
    }

    if (fwrite(&h, sizeof(h), 1, fo) != 1 ||
        fwrite(recs.data(), sizeof(expert_bundle_layer), recs.size(), fo) != recs.size()) {
        die("short write in the header");
    }
    std::vector<char> zero(EXPERT_BUNDLE_ALIGN, 0);
    uint64_t pos = sizeof(h) + recs.size() * sizeof(expert_bundle_layer);
    while (pos < h.data_offset) {
        const uint64_t n = (h.data_offset - pos) < zero.size() ? (h.data_offset - pos) : zero.size();
        fwrite(zero.data(), 1, (size_t) n, fo);
        pos += n;
    }

    // copy expert by expert: the write side is sequential, so this runs at
    // whatever the drive does for streaming writes
    std::vector<char> buf;
    uint64_t written = 0;
    size_t   n_bundles = 0;
    for (size_t li = 0; li < recs.size(); ++li) {
        const auto & r  = recs[li];
        const auto & cs = layers[(int) r.il];
        buf.assign((size_t) r.stride, 0);
        for (int64_t e = 0; e < n_expert; ++e) {
            size_t at = 0;
            int k = 0;
            for (const auto & [ord, c] : cs) {
                (void) ord;
                if (fseek64(in, (long long) (c.file_off + (uint64_t) e * c.bytes_per_expert), SEEK_SET) != 0) {
                    die("seek failed in the model");
                }
                if (fread(buf.data() + at, 1, (size_t) c.bytes_per_expert, in) != c.bytes_per_expert) {
                    die("short read in the model");
                }
                at += (size_t) c.bytes_per_expert;
                k++;
            }
            memset(buf.data() + at, 0, (size_t) r.stride - at); // pad, so reads stay in bounds
            if (fwrite(buf.data(), 1, (size_t) r.stride, fo) != (size_t) r.stride) {
                die("short write in the sidecar");
            }
            written += r.stride;
            n_bundles++;
        }
        printf("  layer %2u: %d chunks, %.2f MiB/expert\r", r.il, (int) r.n_chunk, r.stride / 1048576.0);
        fflush(stdout);
    }
    printf("\n  wrote %zu bundles, %.2f GiB -> %s\n", n_bundles, written / 1073741824.0, out.c_str());
    fclose(fo);

    if (verify) {
        printf("  verifying every expert against the GGUF...\n");
        FILE * fv = fopen(out.c_str(), "rb");
        if (!fv) {
            die("cannot reopen the sidecar", out.c_str());
        }
        std::vector<char> a, b;
        uint64_t checked = 0;
        for (const auto & r : recs) {
            const auto & cs = layers[(int) r.il];
            for (int64_t e = 0; e < n_expert; ++e) {
                uint64_t at = r.base + (uint64_t) e * r.stride;
                for (const auto & [ord, c] : cs) {
                    (void) ord;
                    a.resize((size_t) c.bytes_per_expert);
                    b.resize((size_t) c.bytes_per_expert);
                    fseek64(in, (long long) (c.file_off + (uint64_t) e * c.bytes_per_expert), SEEK_SET);
                    fseek64(fv, (long long) at, SEEK_SET);
                    if (fread(a.data(), 1, a.size(), in) != a.size() ||
                        fread(b.data(), 1, b.size(), fv) != b.size()) {
                        die("short read while verifying");
                    }
                    if (memcmp(a.data(), b.data(), a.size()) != 0) {
                        fprintf(stderr, "moe-bundle: MISMATCH at layer %u expert %lld tensor %s\n",
                                r.il, (long long) e, c.name.c_str());
                        return 1;
                    }
                    at      += c.bytes_per_expert;
                    checked += c.bytes_per_expert;
                }
            }
        }
        fclose(fv);
        printf("  verify OK: %.2f GiB identical to the GGUF\n", checked / 1073741824.0);
    }

    fclose(in);
    gguf_free(gctx);
    ggml_free(mctx);
    return 0;
}
