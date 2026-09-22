// moe-trace: run a prompt, then greedy-decode N tokens, and dump per layer and
// per decode step what the MoE router did: the selected expert ids
// (ffn_moe_topk), the router probabilities (ffn_moe_probs) and the pre-norm
// FFN input (ffn_inp). bench/analyze-trace.py turns the stream into expert
// popularity, temporal reuse and stale-router prediction accuracy.
//
// Usage: llama-moe-trace -m model.gguf -p "prompt" -n 64 [--trace-out trace.bin] [common options]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

enum rec_kind : uint32_t { REC_FFN_INP = 0, REC_TOPK = 1, REC_PROBS = 2 };

struct rec_hdr {
    uint32_t magic;    // 'MOET'
    uint32_t kind;     // rec_kind
    int32_t  layer;
    int32_t  ne0;      // n_embd | n_expert_used | n_expert
    int32_t  ne1;      // n_tokens in this decode call
    int32_t  type;     // ggml_type of the payload
    int32_t  step;     // 0 = prefill, i = i-th generated token
    int32_t  reserved;
};

struct trace_state {
    FILE *  f = nullptr;
    int     step = 0;
    size_t  n_records = 0;
    std::vector<uint8_t> buf;
};

// "<base>-<layer>" -> kind, layer; false when the node is not one we record
bool classify(const char * name, rec_kind & kind, int & layer) {
    const char * dash = strrchr(name, '-');
    if (!dash) {
        return false;
    }
    const std::string base(name, dash - name);
    if (base == "ffn_inp") {
        kind = REC_FFN_INP;
    } else if (base == "ffn_moe_topk") {
        kind = REC_TOPK;
    } else if (base == "ffn_moe_probs") {
        kind = REC_PROBS;
    } else {
        return false;
    }
    // require "<base>-<digits>" exactly: ggml views inherit names like "ffn_moe_probs-3 (reshaped)"
    const char * q = dash + 1;
    if (*q == 0) {
        return false;
    }
    for (const char * c = q; *c; ++c) {
        if (*c < '0' || *c > '9') {
            return false;
        }
    }
    layer = atoi(q);
    return true;
}

bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    rec_kind kind;
    int layer;
    if (!classify(t->name, kind, layer)) {
        return false; // not observed, no graph split needed
    }
    if (ask) {
        return true;
    }
    auto * st = (trace_state *) user_data;
    GGML_ASSERT(ggml_type_size(t->type) == t->nb[0]); // rows are dense; only the outer dims may be strided

    // flatten dims 1..3 into rows of ne0 elements (views into argsort etc. can be strided)
    const size_t  row    = (size_t) t->ne[0] * ggml_type_size(t->type);
    const int64_t n_rows = t->ne[1] * t->ne[2] * t->ne[3];
    st->buf.resize(row * n_rows);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        size_t r = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1, ++r) {
                    const char * src = (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    memcpy(st->buf.data() + r * row, src, row);
                }
            }
        }
    } else {
        GGML_ASSERT(ggml_is_contiguous(t));
        ggml_backend_tensor_get(t, st->buf.data(), 0, st->buf.size());
    }

    const rec_hdr h = { 0x54454F4Du, kind, layer, (int32_t) t->ne[0], (int32_t) n_rows, (int32_t) t->type, st->step, 0 };
    fwrite(&h, sizeof(h), 1, st->f);
    fwrite(st->buf.data(), 1, st->buf.size(), st->f);
    st->n_records++;
    return true; // false would abort the graph computation
}

} // namespace

int main(int argc, char ** argv) {
    // pull our own option out before common_params_parse sees it
    std::string trace_out = "moe-trace.bin";
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--trace-out") == 0 && i + 1 < argc) {
            trace_out = argv[++i];
        } else {
            args.push_back(argv[i]);
        }
    }

    common_params params;
    params.n_predict = 32;
    params.sampling.temp = 0.0f;

    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    trace_state st;
    st.f = fopen(trace_out.c_str(), "wb");
    if (!st.f) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, trace_out.c_str());
        return 1;
    }
    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup            = false; // a warmup decode would pollute the trace

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt, use -p\n", __func__);
        return 1;
    }
    LOG_INF("%s: n_layer = %d, n_embd = %d, prompt tokens = %zu, n_predict = %d\n", __func__,
            llama_model_n_layer(model), llama_model_n_embd(model), tokens.size(), params.n_predict);

    // step 0: prefill as one batch
    st.step = 0;
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("%s: prefill failed\n", __func__);
        return 1;
    }

    // steps 1..n: greedy decode, one token per call (the regime that matters for streaming)
    // --temp 0 (default here) = greedy; --temp T -s SEED = sampled, to avoid greedy repetition loops
    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.sampling.temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.sampling.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    std::string text;
    for (int i = 0; i < params.n_predict; ++i) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, tok);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        text += common_token_to_piece(ctx, tok);
        st.step = i + 1;
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("%s: decode failed at step %d\n", __func__, st.step);
            return 1;
        }
    }
    // the full text goes to a sidecar file; the log only gets a preview
    if (FILE * ft = fopen((trace_out + ".txt").c_str(), "wb")) {
        fwrite(text.data(), 1, text.size(), ft);
        fclose(ft);
    }
    LOG_INF("%s: generated %d tokens: %.120s...\n", __func__, st.step, text.c_str());
    LOG_INF("%s: %zu records -> %s\n", __func__, st.n_records, trace_out.c_str());

    fclose(st.f);
    llama_sampler_free(smpl);
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
