"""Regenerate the figures used in docs/streaming-moe-on-a-laptop.md.

    python bench/make-charts.py [--results bench/results] [--out docs/img]

Reads only the measurement files committed under bench/results: llama-bench
JSON, the per-run moe-run JSON summarised in ab.csv, the trace analysis JSON,
and the small CSVs the PowerShell harness appends to. Every value plotted comes
from one of those files; nothing is hardcoded here except axis labels.
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette: fixed slot order, never cycled. Checked for colour-vision
# separation and for contrast against this surface; the aqua slot falls below
# 3:1, so every mark drawn in it also carries a direct numeric label.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"
AXIS = "#c9c8c3"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.dpi": 160, "savefig.bbox": "tight",
    })


def grid_y(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def grid_x(ax):
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def label_bars(ax, bars, fmt="{:.2f}", dy=3):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8, color=INK2)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
    return path


# --------------------------------------------------------------- figures

def fig_headline(res, out):
    """The three measured pairs the README leads with.

    Each panel is one controlled comparison against its own baseline, with its
    own conditions in the subtitle. They are not stages of a single run and do
    not multiply together.
    """
    disk = sorted(glob.glob(os.path.join(res, "disk-bench*.json")))
    if not disk:
        return None
    big = {t["pattern"]: t["mb_s"] for t in json.load(open(disk[-1]))["tests"]}
    expert_read = next(v for k, v in big.items() if "2 MB" in k and "QD1" in k)

    press = read_csv(os.path.join(res, "summary-v1.csv"))
    paging = [float(r["disk_read_MB"]) / float(r["run_s"])
              for r in press if r["tag"] in ("avail2g", "avail1.5g") and r["test"] == "tg64"]
    paged = sum(paging) / len(paging)

    ab = [r for r in read_csv(os.path.join(res, "ab.csv")) if r["tag"].startswith("v5")]
    at25 = [r for r in ab if r["avail_gb"] == "2.5" and float(r["faults_per_token"]) > 100]
    base25 = [float(r["gen_tps"]) for r in at25 if "--cache-slots" not in r["extra"]]
    cache25 = [float(r["gen_tps"]) for r in at25 if "--cache-slots" in r["extra"]]

    lay = [r for r in read_csv(os.path.join(res, "layout.csv"))
           if float(r["avail_gb"]) == 0 and int(r["slots"]) == 32]
    split = [float(r["gen_tps"]) for r in lay if r["layout"] == "split"]
    bund = [float(r["gen_tps"]) for r in lay if r["layout"] == "bundled"]
    if not (base25 and cache25 and split and bund):
        return None

    def mean(v):
        return sum(v) / len(v)

    panels = (
        ("Reading the weights", "MB/s", "{:.0f}",
         paged, expert_read,
         "page faults vs whole-expert reads, same drive"),
        ("Decoding, 2.5 GB of RAM", "tok/s", "{:.2f}",
         mean(base25), mean(cache25),
         "OS page cache vs an expert cache we own"),
        ("Decoding, 1.86 GiB cache", "tok/s", "{:.2f}",
         mean(split), mean(bund),
         "three reads per expert vs one"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for ax, (title, ylab, fmt, before, after, sub) in zip(axes, panels):
        bars = ax.bar([0, 1], [before, after], width=0.52, color=[S1, S2], linewidth=0)
        label_bars(ax, bars, fmt)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["", ""])
        ax.set_title(title, fontsize=10.5, pad=20)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, after * 1.35)
        ax.annotate("%.1fx" % (after / before), (0.5, after * 1.17), ha="center",
                    fontsize=12, color=INK, fontweight="bold")
        ax.annotate(sub, (0.5, 1.0), xycoords="axes fraction", textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7.5, color=INK2)
        grid_y(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=S1), plt.Rectangle((0, 0), 1, 1, color=S2)]
    fig.legend(handles, ["what the machine gives you", "moe-stream"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=9)
    fig.text(0.5, -0.14, "OLMoE-1B-7B Q4_K_M on a Core i5-1334U with 16 GB and a Samsung BM9C1 NVMe, CPU only. "
                         "Three separate A/B comparisons, each against its own baseline; they are not stages of one run.",
             ha="center", fontsize=7.5, color=INK2)
    return save(fig, out, "fig0-headline.png")


def fig_threads(res, out):
    """tok/s vs thread count, model fully resident."""
    src = glob.glob(os.path.join(res, "*threads.json"))
    if not src:
        return None
    rows = json.load(open(sorted(src)[-1]))
    pp = sorted((r["n_threads"], r["avg_ts"], r["stddev_ts"]) for r in rows if r["n_prompt"])
    tg = sorted((r["n_threads"], r["avg_ts"], r["stddev_ts"]) for r in rows if r["n_gen"])

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for data, color, name in ((pp, S1, "prompt (pp512)"), (tg, S2, "decode (tg128)")):
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        e = [d[2] for d in data]
        ax.errorbar(x, y, yerr=e, color=color, linewidth=2, marker="o",
                    markersize=5, capsize=3, elinewidth=1, label=name)
        ax.annotate("%.1f" % y[-1], (x[-1], y[-1]), textcoords="offset points",
                    xytext=(9, -3), fontsize=8, color=INK2)
    ax.set_xlabel("threads")
    ax.set_ylabel("tok/s")
    ax.set_title("Thread scaling, model fully in RAM")
    ax.set_xticks([d[0] for d in pp])
    ax.set_ylim(0, 66)
    grid_y(ax)
    ax.legend(loc="center right")
    return save(fig, out, "fig1-threads.png")


def fig_pressure(res, out):
    """Throughput and disk traffic vs the RAM the model is allowed.

    Two panels, one measure each: tok/s and GiB share no scale.
    """
    rows = read_csv(os.path.join(res, "summary-v1.csv"))
    order = ["avail6g", "avail3g", "avail2g", "avail1.5g"]
    labels = ["6 GB\n(model fits)", "3 GB", "2 GB", "1.5 GB"]
    by = {(r["tag"], r["test"]): r for r in rows}
    if not all((t, "tg64") in by for t in order):
        return None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    x = list(range(len(order)))
    for test, color, name in (("pp256", S1, "prompt (pp256)"), ("tg64", S2, "decode (tg64)")):
        y = [float(by[(t, test)]["avg_ts"]) for t in order]
        e = [float(by[(t, test)]["stddev_ts"]) for t in order]
        a1.errorbar(x, y, yerr=e, color=color, linewidth=2, marker="o",
                    markersize=5, capsize=3, elinewidth=1, label=name)
    for i, t in enumerate(order):
        v = float(by[(t, "tg64")]["avg_ts"])
        a1.annotate("%.1f" % v, (i, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=INK2)
    a1.set_xticks(x)
    a1.set_xticklabels(labels)
    a1.set_ylabel("tok/s")
    a1.set_ylim(0, 60)
    a1.set_title("Throughput vs available RAM")
    grid_y(a1)
    a1.legend(loc="lower left")

    gb = [float(by[(t, "tg64")]["disk_read_MB"]) / 1024 for t in order]
    bars = a2.bar(x, gb, width=0.56, color=S1, linewidth=0)
    label_bars(a2, bars, "{:.1f}")
    a2.set_xticks(x)
    a2.set_xticklabels(labels)
    a2.set_ylabel("GiB read from disk per run")
    a2.set_ylim(0, max(gb) * 1.2)
    a2.set_title("Disk traffic for the same work")
    grid_y(a2)
    fig.text(0.5, -0.05, "OLMoE-1B-7B Q4_K_M, 10 threads, mmap; RAM pinned with ram-eater.py",
             ha="center", fontsize=7.5, color=INK2)
    return save(fig, out, "fig2-pressure.png")


def fig_disk(res, out):
    """What the drive delivers by request size, against the rate the
    page-fault path actually achieved on the same drive."""
    src = sorted(glob.glob(os.path.join(res, "disk-bench*.json")))
    if not src:
        return None
    d = json.load(open(src[-1]))
    tests = d["tests"]
    names = [t["pattern"] for t in tests][::-1]
    mbs = [t["mb_s"] for t in tests][::-1]

    rows = read_csv(os.path.join(res, "summary-v1.csv"))
    paging = [float(r["disk_read_MB"]) / float(r["run_s"])
              for r in rows if r["tag"] in ("avail2g", "avail1.5g") and r["test"] == "tg64"]
    paged = sum(paging) / len(paging)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    big = [("2 MB" in n) or ("sequential" in n) for n in names]
    colors = [S1 if b else S3 for b in big]
    bars = ax.barh(range(len(names)), mbs, height=0.6, color=colors, linewidth=0)
    for i, v in enumerate(mbs):
        ax.annotate("%.0f" % v, (v + max(mbs) * 0.015, i), va="center",
                    fontsize=8, color=INK2)
    ax.axvline(paged, color=S2, linewidth=2, linestyle="--")
    ax.annotate("demand paging on the\nsame drive: %.0f MB/s" % paged,
                (paged + max(mbs) * 0.04, 1.5), fontsize=8, color=S2, va="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("MB/s, unbuffered reads")
    ax.set_xlim(0, max(mbs) * 1.18)
    ax.set_title("The drive is not the bottleneck; the request size is")
    grid_x(ax)
    fig.text(0.5, -0.04, d.get("note", ""), ha="center", fontsize=7.5, color=INK2)
    return save(fig, out, "fig3-disk-patterns.png")


def fig_router(res, out):
    """Stale-router prediction accuracy by lookahead distance."""
    d = json.load(open(os.path.join(res, "trace-olmoe-256.analysis.json")))
    sr = d["stale_router"]
    ks = sorted(int(k) for k in sr)
    k = d["k"]
    series = [("topk_hit", S1, "exact top-%d" % k),
              ("recall_2k", S2, "recall, %d experts fetched" % (2 * k)),
              ("recall_4k", S3, "recall, %d experts fetched" % (4 * k))]

    fig, ax = plt.subplots(figsize=(5.9, 3.4))
    for key, color, name in series:
        y = [sr[str(i)][key] for i in ks]
        ax.plot(ks, y, color=color, linewidth=2, marker="o", markersize=5, label=name)
        ax.annotate("%.2f" % y[0], (ks[0], y[0]), textcoords="offset points",
                    xytext=(-26, -3), fontsize=8, color=INK2)
        ax.annotate("%.2f" % y[-1], (ks[-1], y[-1]), textcoords="offset points",
                    xytext=(10, -3), fontsize=8, color=INK2)
    ax.set_xlabel("layers of lookahead")
    ax.set_ylabel("fraction of the real top-k recovered")
    ax.set_title("A router run one layer early already knows the answer")
    ax.set_xticks(ks)
    ax.set_xlim(0.5, 4.8)
    ax.set_ylim(0.6, 1.03)
    grid_y(ax)
    ax.legend(loc="lower left")
    fig.text(0.5, -0.03, "%s, %d layers, %d experts, top-%d, %d sampled decode tokens"
             % (d["arch"], d["n_layer"], d["n_expert"], k, d["n_tokens"]),
             ha="center", fontsize=7.5, color=INK2)
    return save(fig, out, "fig4-stale-router.png")


def fig_popularity(res, out):
    """How many experts you must hold to cover a share of all selections."""
    d = json.load(open(os.path.join(res, "trace-olmoe-256.analysis.json")))
    cov = d["coverage_avg"]
    ks = sorted(cov, key=float)
    y = [cov[k] for k in ks]
    n = d["n_expert"]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(range(len(ks)), y, width=0.54, color=S1, linewidth=0)
    label_bars(ax, bars, "{:.1f}")
    ax.axhline(n, color=INK2, linewidth=1, linestyle=":")
    ax.annotate("all %d experts" % n, (len(ks) - 0.5, n), textcoords="offset points",
                xytext=(0, 4), ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(["%.0f%%" % (float(k) * 100) for k in ks])
    ax.set_xlabel("share of expert selections covered")
    ax.set_ylabel("experts needed (of %d)" % n)
    ax.set_ylim(0, n * 1.14)
    ax.set_title("There is no small hot set to pin")
    grid_y(ax)
    return save(fig, out, "fig5-expert-popularity.png")


def fig_ab(res, out):
    """Page cache vs explicit expert cache at the same RAM budget."""
    rows = [r for r in read_csv(os.path.join(res, "ab.csv")) if r["tag"].startswith("v5")]
    if not rows:
        return None
    budgets = ["2.5", "1.5"]
    series = ((False, S1, "page cache (mmap)"), (True, S2, "explicit expert cache"))

    def pick(b, cache):
        rr = [r for r in rows if r["avail_gb"] == b
              and (("--cache-slots" in r["extra"]) == cache)]
        # One 2.5 GB baseline run is left out of the mean: Windows had paged
        # other processes out, the whole model was resident, and it recorded
        # 31 faults/token against 25k for every other baseline run.
        return [r for r in rr if float(r["faults_per_token"]) > 100]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    w = 0.3
    for i, b in enumerate(budgets):
        for j, (cache, color, name) in enumerate(series):
            rr = pick(b, cache)
            if not rr:
                continue
            pos = i + (j - 0.5) * (w + 0.03)
            tps = [float(r["gen_tps"]) for r in rr]
            bars = a1.bar([pos], [sum(tps) / len(tps)], width=w, color=color,
                          linewidth=0, label=name if i == 0 else None)
            label_bars(a1, bars, "{:.2f}", dy=9)
            jitter = [pos + (n - (len(tps) - 1) / 2) * 0.1 for n in range(len(tps))]
            a1.plot(jitter, tps, "o", color=SURFACE, markersize=5,
                    markeredgewidth=1.3, markeredgecolor=INK2, zorder=3)

            flt = [float(r["faults_per_token"]) for r in rr]
            mean_f = sum(flt) / len(flt)
            a2.bar([pos], [mean_f], width=w, color=color, linewidth=0,
                   label=name if i == 0 else None)
            a2.annotate(("%.0f" % mean_f if mean_f < 10000 else "%.0fk" % (mean_f / 1000)),
                        (pos, mean_f * 1.1), ha="center", fontsize=8, color=INK2)

    for ax, title, ylab in ((a1, "Decode throughput at equal RAM budget", "tok/s"),
                            (a2, "Page faults per generated token", "faults/token (log scale)")):
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels(["%s GB available" % b for b in budgets])
        ax.set_title(title)
        ax.set_ylabel(ylab)
        grid_y(ax)
    a1.set_ylim(0, 4.3)
    a1.legend(loc="upper right")
    a2.set_yscale("log")
    a2.set_ylim(100, 300000)
    fig.text(0.5, -0.05, "bars are the mean of the interleaved runs, dots the individual runs; "
                         "48 greedy tokens, identical prompt and identical output",
             ha="center", fontsize=7.5, color=INK2)
    return save(fig, out, "fig6-cache-ab.png")


def fig_ppl(res, out):
    """Quality cost of routing to fewer experts."""
    rows = read_csv(os.path.join(res, "ppl.csv"))
    corpora = (("wikitext2", S1, "wikitext-2 test"), ("gitdoc", S2, "technical prose"))
    fig, ax = plt.subplots(figsize=(5.7, 3.4))
    for tag, color, name in corpora:
        rr = sorted((int(r["k"]), float(r["ppl"]), float(r["ppl_err"]))
                    for r in rows if r["tag"] == tag and int(r["k"]) >= 4)
        if not rr:
            continue
        ax.errorbar([r[0] for r in rr], [r[1] for r in rr], yerr=[r[2] for r in rr],
                    color=color, linewidth=2, marker="o", markersize=5,
                    capsize=3, elinewidth=1, label=name)
        for kk, p, _ in rr:
            ax.annotate("%.2f" % p, (kk, p), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=INK2)
    k2 = [float(r["ppl"]) for r in rows if r["tag"] == "gitdoc" and r["k"] == "2"]
    ax.set_xlabel("experts used per token (K)")
    ax.set_ylabel("perplexity, ctx 512")
    ax.set_title("Halving the bytes by halving K is not free")
    ax.set_xticks([4, 6, 8])
    ax.set_xlim(8.6, 3.4)
    ax.set_ylim(0, 19)
    grid_y(ax)
    ax.legend(loc="upper left")
    note = "native K is 8; lower K set with --override-kv, no retraining"
    if k2:
        note += ". K=2 on technical prose reaches %.1f, off this scale" % k2[0]
    fig.text(0.5, -0.03, note, ha="center", fontsize=7.5, color=INK2)
    return save(fig, out, "fig7-ppl-vs-k.png")


def fig_layout(res, out):
    """Results 8: one read per expert against three.

    Only the unpinned runs are plotted. The runs under pinned memory are in
    layout.csv too, but there the drive throttled across the sequence and the
    layout effect is not separable from it.
    """
    path = os.path.join(res, "layout.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in read_csv(path) if float(r["avail_gb"]) == 0]
    if not rows:
        return None
    sizes = sorted({int(r["slots"]) for r in rows})
    series = (("split", S1, "3 reads/expert (GGUF)"),
              ("bundled", S2, "1 read/expert (sidecar)"))
    panels = (("gen_tps", "Decode throughput", "tok/s", "{:.2f}", True),
              ("reads_per_token", "Read requests per token", "requests", "{:.0f}", False),
              ("io_mbps", "Read throughput per I/O thread", "MB/s", "{:.0f}", False))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    w = 0.32
    for ax, (col, title, ylab, fmt, dots) in zip(axes, panels):
        top = 0.0
        for i, n in enumerate(sizes):
            for j, (layout, color, name) in enumerate(series):
                vals = [float(r[col]) for r in rows
                        if int(r["slots"]) == n and r["layout"] == layout]
                if not vals:
                    continue
                pos = i + (j - 0.5) * (w + 0.03)
                mean = sum(vals) / len(vals)
                top = max(top, max(vals))
                bars = ax.bar([pos], [mean], width=w, color=color, linewidth=0,
                              label=name if i == 0 else None)
                label_bars(ax, bars, fmt, dy=9 if dots else 3)
                if dots:
                    jit = [pos + (m - (len(vals) - 1) / 2) * 0.1 for m in range(len(vals))]
                    ax.plot(jit, vals, "o", color=SURFACE, markersize=5,
                            markeredgewidth=1.3, markeredgecolor=INK2, zorder=3)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(["%d slots/layer" % n for n in sizes])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, top * 1.25)
        grid_y(ax)
    axes[0].legend(loc="upper left")
    fig.text(0.5, -0.05, "same model, prompt, cache budget and prefetch depth; runs interleaved, "
                         "bars the mean and dots the individual runs; identical output on all of them",
             ha="center", fontsize=7.5, color=INK2)
    return save(fig, out, "fig8-bundled-layout.png")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "bench", "results"))
    ap.add_argument("--out", default=os.path.join(here, "docs", "img"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    for f in (fig_headline, fig_threads, fig_pressure, fig_disk, fig_router,
              fig_popularity, fig_ab, fig_ppl, fig_layout):
        if f(a.results, a.out) is None:
            print("skipped", f.__name__, "- input not present")


if __name__ == "__main__":
    main()
