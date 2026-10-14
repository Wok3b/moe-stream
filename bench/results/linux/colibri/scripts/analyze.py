"""Table of the comparison rounds: medians over rounds of decode, time to first output and peak memory.
    analyze.py RESULTS_DIR"""
import json
import re
import statistics as st
import sys
from pathlib import Path

R = Path(sys.argv[1])
NAMES = ["story", "code", "notes", "licence"]
BUDGETS = ["6G", "8G", "10G", "12G", "infinity"]
rows = {}
for p in sorted(R.glob("*-r[0-9].json")):
    m = re.fullmatch(r"(moe|coli)-(\w+)-r(\d)\.json", p.name)
    if not m or m[3] == "0":
        continue
    eng, bud = m[1], m[2]
    rep = json.loads(p.read_text())
    reqs = rep["requests"]
    dec = [(r["completion_tokens"] - 1) / (r["duration_seconds"] - r["first_output_seconds"])
           if r["success"] else None for r in reqs]
    ttft = [r["first_output_seconds"] if r["success"] else None for r in reqs]
    summ = next((l for l in (R / "summary.txt").read_text().splitlines() if l.startswith(p.stem + ":")), "")
    peak = float(re.search(r"peak ([\d.]+) GiB", summ)[1]) if "peak" in summ else None
    ready = float(re.search(r"ready (\d+)s", summ)[1]) if "ready" in summ else None
    rows.setdefault((eng, bud), []).append({"dec": dec, "ttft": ttft, "peak": peak, "ready": ready,
                                            "total": sum(r["duration_seconds"] for r in reqs)})


def med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def fmt(v, d=2):
    return "-" if v is None else f"{v:.{d}f}"


print(f"{'budget':9} {'engine':6} {'n':>2}  " + "  ".join(f"dec {n:7}" for n in NAMES)
      + "  " + "  ".join(f"ttft {n:7}" for n in NAMES) + "   total  peak  ready")
for b in BUDGETS:
    for e in ("moe", "coli"):
        rs = rows.get((e, b))
        if not rs:
            continue
        dec = [med([r["dec"][i] for r in rs]) for i in range(4)]
        ttft = [med([r["ttft"][i] for r in rs]) for i in range(4)]
        spread = [max(v for v in (r["dec"][i] for r in rs) if v) - min(v for v in (r["dec"][i] for r in rs) if v)
                  for i in range(4)] if len(rs) > 1 else None
        print(f"{b:9} {e:6} {len(rs):>2}  " + "  ".join(f"{fmt(v):>11}" for v in dec) + "  "
              + "  ".join(f"{fmt(v, 1):>12}" for v in ttft)
              + f"  {fmt(med([r['total'] for r in rs]), 0):>6}  {fmt(med([r['peak'] for r in rs])):>4}  {fmt(med([r['ready'] for r in rs]), 0):>5}"
              + (f"   dec spread {' '.join(fmt(s) for s in spread)}" if spread else ""))
