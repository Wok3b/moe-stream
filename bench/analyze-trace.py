"""Analyse a llama-moe-trace dump.

    python analyze-trace.py trace.bin model.gguf [--lookahead 3] [--json out.json]

Reports, for the decode steps (prefill is excluded unless --include-prefill):
  * expert popularity per layer: how many experts cover 50/80/90/95% of selections
  * temporal reuse: fraction of a token's experts that the previous token also used
  * stale-router prediction: apply layer L+d's norm + router to the residual
    entering layer L's FFN and compare its top-k with what L+d really chose.
    This is a zero-training prerouter; its accuracy bounds what prefetch can do.
"""
import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

REC_FFN_INP, REC_TOPK, REC_PROBS = 0, 1, 2
HDR = struct.Struct("<IIiiiiii")
NP_TYPES = {0: np.float32, 1: np.float16, 26: np.int32, 27: np.int64, 30: np.uint16}  # ggml_type -> numpy


def read_trace(path):
    """-> {(kind, layer): [(step, array[ne1, ne0])...]}"""
    out = defaultdict(list)
    with open(path, "rb") as f:
        while True:
            raw = f.read(HDR.size)
            if len(raw) < HDR.size:
                break
            magic, kind, layer, ne0, ne1, gtype, step, _ = HDR.unpack(raw)
            assert magic == 0x54454F4D, "bad record magic"
            dt = NP_TYPES[gtype]
            arr = np.frombuffer(f.read(ne0 * ne1 * np.dtype(dt).itemsize), dtype=dt).reshape(ne1, ne0)
            out[(kind, layer)].append((step, arr))
    return out


def load_routers(gguf_path):
    """-> arch, eps, n_expert_used, {layer: (W_gate[n_expert, n_embd], w_norm[n_embd], bias|None)}"""
    r = GGUFReader(gguf_path)
    arch = r.get_field("general.architecture").contents()
    eps = float(r.get_field(f"{arch}.attention.layer_norm_rms_epsilon").contents())
    k = int(r.get_field(f"{arch}.expert_used_count").contents())
    tensors = {t.name: t for t in r.tensors}

    def as_f32(t):
        d = t.data
        if t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
            d = dequantize(d, t.tensor_type)
        return np.asarray(d, dtype=np.float32)

    routers = {}
    layer = 0
    while f"blk.{layer}.ffn_gate_inp.weight" in tensors:
        w = as_f32(tensors[f"blk.{layer}.ffn_gate_inp.weight"])       # numpy shape (n_expert, n_embd)
        n = as_f32(tensors[f"blk.{layer}.ffn_norm.weight"])
        b = tensors.get(f"blk.{layer}.exp_probs_b.bias")
        routers[layer] = (w, n, as_f32(b) if b is not None else None)
        layer += 1
    return arch, eps, k, routers


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def topk_ids(logits, k):
    return np.argpartition(-logits, k - 1, axis=-1)[..., :k]


def overlap(a, b):
    """mean |a ∩ b| / k over rows"""
    return np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("model")
    ap.add_argument("--lookahead", type=int, default=3)
    ap.add_argument("--include-prefill", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    trace = read_trace(args.trace)
    arch, eps, k, routers = load_routers(args.model)
    layers = sorted({l for (kind, l) in trace if kind == REC_TOPK})
    n_expert = trace[(REC_PROBS, layers[0])][0][1].shape[1]
    n_layer = len(layers)

    def stack(kind, layer):
        recs = [(s, a) for s, a in trace[(kind, layer)] if args.include_prefill or s > 0]
        recs.sort(key=lambda r: r[0])
        return np.concatenate([a for _, a in recs], axis=0)  # [n_tokens_total, ne0]

    topk = {l: stack(REC_TOPK, l) for l in layers}
    ffn_inp = {l: stack(REC_FFN_INP, l) for l in layers}
    n_tok = topk[layers[0]].shape[0]
    print(f"{arch}: {n_layer} MoE layers, {n_expert} experts, top-{k}, {n_tok} decode tokens analysed\n")

    # --- popularity -------------------------------------------------------
    print("expert popularity (decode): experts needed to cover X% of selections, per layer")
    print("layer   50%   80%   90%   95%   unique")
    cover = {}
    for l in layers:
        counts = np.bincount(topk[l].ravel(), minlength=n_expert)
        cum = np.cumsum(np.sort(counts)[::-1]) / counts.sum()
        need = {p: int(np.searchsorted(cum, p) + 1) for p in (0.5, 0.8, 0.9, 0.95)}
        cover[l] = need
        print(f"{l:5d} {need[0.5]:5d} {need[0.8]:5d} {need[0.9]:5d} {need[0.95]:5d} {int((counts > 0).sum()):8d}")
    avg = {p: np.mean([cover[l][p] for l in layers]) for p in (0.5, 0.8, 0.9, 0.95)}
    print(f"  avg {avg[0.5]:5.1f} {avg[0.8]:5.1f} {avg[0.9]:5.1f} {avg[0.95]:5.1f}   (of {n_expert})\n")

    # --- temporal reuse ---------------------------------------------------
    reuse = {l: overlap(topk[l][1:], topk[l][:-1]) for l in layers}
    print(f"temporal reuse (token t vs t-1, same layer): mean {np.mean(list(reuse.values())):.3f}, "
          f"min {min(reuse.values()):.3f}, max {max(reuse.values()):.3f}")

    # --- stale-router prediction -----------------------------------------
    print("\nstale-router prediction: router of layer L+d applied to the residual entering FFN of layer L")
    print("  d   top-k hit   recall@2k   recall@4k    (mean over layers, decode tokens)")
    pred = {}
    for d in range(1, args.lookahead + 1):
        hit, r2, r4 = [], [], []
        for l in layers:
            if l + d not in routers or (l + d) not in topk:
                continue
            w, wn, bias = routers[l + d]
            x = rms_norm(ffn_inp[l].astype(np.float32), wn, eps)
            logits = x @ w.T
            if bias is not None:
                logits = logits + bias   # selection uses the biased scores (DeepSeek-style)
            actual = topk[l + d]
            hit.append(overlap(topk_ids(logits, k), actual))
            r2.append(overlap(actual, topk_ids(logits, min(2 * k, n_expert))))
            r4.append(overlap(actual, topk_ids(logits, min(4 * k, n_expert))))
        pred[d] = (float(np.mean(hit)), float(np.mean(r2)), float(np.mean(r4)))
        print(f"  {d}     {pred[d][0]:.3f}       {pred[d][1]:.3f}       {pred[d][2]:.3f}")
    # sanity: d=0 must be ~1.0 (same layer, same input) or the bookkeeping is wrong
    w, wn, bias = routers[layers[0]]
    logits0 = rms_norm(ffn_inp[layers[0]].astype(np.float32), wn, eps) @ w.T
    if bias is not None:
        logits0 = logits0 + bias
    print(f"  0     {overlap(topk_ids(logits0, k), topk[layers[0]]):.3f}   (self-check, should be ~1.0)")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "arch": arch, "n_layer": n_layer, "n_expert": n_expert, "k": k, "n_tokens": int(n_tok),
            "coverage_avg": {str(p): float(v) for p, v in avg.items()},
            "temporal_reuse": float(np.mean(list(reuse.values()))),
            "stale_router": {str(d): {"topk_hit": v[0], "recall_2k": v[1], "recall_4k": v[2]} for d, v in pred.items()},
        }, indent=2))


if __name__ == "__main__":
    main()
