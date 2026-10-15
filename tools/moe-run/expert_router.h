// Stale-router expert prediction: the residual entering layer L's FFN, pushed through the router of a
// later layer, names experts that layer will likely pick (docs/NOTES.md, Results 2). Shared by moe-run
// and the expert cache's predictor.
#pragma once

#include "ggml.h"
#include "llama-model.h" // in-tree: layer tensors

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// convert any ggml tensor to a flat f32 vector (row-major over ne[0])
inline std::vector<float> to_f32(const ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        GGML_ASSERT(ggml_is_contiguous(t));
        memcpy(out.data(), t->data, n * sizeof(float));
        return out;
    }
    const auto * traits = ggml_get_type_traits(t->type);
    GGML_ASSERT(traits->to_float && ggml_is_contiguous(t));
    const int64_t rows = n / t->ne[0];
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float((const char *) t->data + r * t->nb[1], out.data() + r * t->ne[0], t->ne[0]);
    }
    return out;
}

struct expert_router {
    struct layer {
        const float *      gate = nullptr; // [n_expert, n_embd]: the model's own f32 tensor, or gate_own
        std::vector<float> gate_own;       // converted copy when the router is not stored as f32
        std::vector<float> norm;           // [n_embd]
        std::vector<float> probs_b;        // [n_expert] or empty
        bool               is_moe = false;
    };

    int   n_layer  = 0;
    int   n_embd   = 0;
    int   n_expert = 0;
    int   k        = 0; // experts used per token
    int   gating   = 0;
    float eps      = 1e-6f;
    bool  layer_norm = false; // LayerNorm (subtract the mean) instead of RMSNorm
    std::string inp_name = "ffn_inp"; // graph tensor holding the residual the router reads
    std::vector<layer> layers;

    // false: this architecture's router input is unknown, nothing can be predicted
    bool init(const llama_model * model) {
        const auto & hp = model->hparams;
        n_layer  = (int) model->layers.size(); // hparams.n_layer() is not exported from the DLL
        const bool qwen_post_norm = model->arch == LLM_ARCH_QWEN35MOE || model->arch == LLM_ARCH_QWEN3NEXT;
        // cohere2moe runs attention and the MoE in parallel on attn_norm(layer input), so the previous
        // layer's output routes the next layer exactly (models/cohere2moe.cpp)
        const bool parallel_block = model->arch == LLM_ARCH_COHERE2MOE;
        inp_name = model->arch == LLM_ARCH_GEMMA4 ? "attn_out" :
                   qwen_post_norm                ? "attn_residual" :
                   parallel_block                ? "l_out" : "ffn_inp";
        n_embd   = (int) hp.n_embd;
        n_expert = (int) hp.n_expert;
        k        = (int) hp.n_expert_used_arr[0];
        layer_norm = parallel_block && hp.f_norm_rms_eps == 0.0f;
        eps      = layer_norm ? hp.f_norm_eps : hp.f_norm_rms_eps;
        gating   = (int) hp.expert_gating_func;
        layers.assign(n_layer, {});
        for (int il = 0; il < n_layer; ++il) {
            const auto & L = model->layers[il];
            auto & w = layers[il];
            if (!L.ffn_gate_inp || !L.ffn_down_exps) {
                continue; // dense layer
            }
            if (model->arch == LLM_ARCH_GEMMA4) {
                // gemma4 routes on rms_norm(attn_out) / sqrt(n_embd) * ffn_gate_inp_s (models/gemma4.cpp)
                w.norm = to_f32(L.ffn_gate_inp_s);
                for (auto & v : w.norm) {
                    v /= sqrtf((float) n_embd);
                }
            } else if (qwen_post_norm) {
                w.norm = to_f32(L.attn_post_norm); // models/qwen35moe.cpp, qwen3next.cpp: rms_norm(attn_residual)
            } else if (parallel_block) {
                w.norm = to_f32(L.attn_norm);
            } else if (L.ffn_norm) {
                w.norm = to_f32(L.ffn_norm);
            } else {
                return false;
            }
            if (L.ffn_gate_inp->type == GGML_TYPE_F32 && ggml_is_contiguous(L.ffn_gate_inp)) {
                w.gate = (const float *) L.ffn_gate_inp->data;
            } else {
                w.gate_own = to_f32(L.ffn_gate_inp);
                w.gate     = w.gate_own.data();
            }
            if (L.ffn_exp_probs_b) {
                w.probs_b = to_f32(L.ffn_exp_probs_b);
            }
            w.is_moe = true;
        }
        return true;
    }

    // scores of layer il's router for the pre-norm residual x; xn is [n_embd] scratch, logits [n_expert]
    void route(int il, const float * x, float * xn, float * logits) const {
        const auto & w = layers[il];
        float mean = 0.0f;
        if (layer_norm) {
            double sum = 0.0;
            for (int i = 0; i < n_embd; ++i) {
                sum += x[i];
            }
            mean = (float) (sum / n_embd);
        }
        double ss = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            ss += (double) (x[i] - mean) * (x[i] - mean);
        }
        const float scale = 1.0f / sqrtf((float) (ss / n_embd) + eps);
        for (int i = 0; i < n_embd; ++i) {
            xn[i] = (x[i] - mean) * scale * w.norm[i];
        }
        // independent lanes: a single float accumulator cannot be vectorized without -ffast-math
        constexpr int LANES = 16;
        const int n_main = n_embd - n_embd % LANES;
        for (int e = 0; e < n_expert; ++e) {
            const float * row = w.gate + (size_t) e * n_embd;
            float acc[LANES] = {};
            for (int i = 0; i < n_main; i += LANES) {
                for (int l = 0; l < LANES; ++l) {
                    acc[l] += row[i + l] * xn[i + l];
                }
            }
            float sum = 0.0f;
            for (int l = 0; l < LANES; ++l) {
                sum += acc[l];
            }
            for (int i = n_main; i < n_embd; ++i) {
                sum += row[i] * xn[i];
            }
            logits[e] = sum;
        }
        if (gating == 2 /* sigmoid */) {
            for (int e = 0; e < n_expert; ++e) {
                logits[e] = 1.0f / (1.0f + expf(-logits[e]));
            }
        }
        if (!w.probs_b.empty()) { // selection bias is added after the gating function
            for (int e = 0; e < n_expert; ++e) {
                logits[e] += w.probs_b[e];
            }
        }
    }
};
