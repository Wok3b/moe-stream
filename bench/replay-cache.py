"""Replay an expert-cache trace (llama-moe-run --cache-trace) against eviction policies.

    python bench/replay-cache.py trace.txt [--slots 25,54] [--policies lru,opt,...]

The trace is the order the cache saw lookups (D), prefetches (P) and prefill seeds (S) in; routing does
not depend on the cache (the output is bit-exact), so one trace replays every policy. The slot logic
mirrors expert_cache.cpp: experts used this token are never evicted, ones prefetched this token only
when nothing else is left. Reported per decode token: demand misses (the graph waits for these) and
reads (misses + prefetch loads; the drive's work).

Policies:
  lru        current cache: least recently used; a prefetch counts as a use
  lru-cold   a prefetched expert is not a use until the graph asks for it: wrong guesses go first
  lfu-N      least uses, exponentially decayed with a half-life of N tokens; ties by LRU
  lru2       LRU-K with K=2: evict the oldest second-to-last use; experts used once go first
  opt        Belady: evict the expert asked for furthest in the future (upper bound, not buildable)
"""

import bisect
import sys
from collections import defaultdict


def load(path):
    events = []
    slots = experts = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                w = line.split()
                slots, experts = int(w[2]), int(w[4])
                continue
            w = line.split()
            events.append((w[0], int(w[1]), int(w[2]), [int(x) for x in w[3:]]))
    return events, slots, experts


class Layer:
    def __init__(self):
        self.slot = {}  # expert -> entry dict


def replay(events, n_slots, policy, next_use=None):
    layers = defaultdict(Layer)
    clock = 0
    misses = reads = pref_reads = no_slot = 0
    tokens = set()
    half = None
    if policy.startswith("lfu-"):
        half = float(policy.split("-")[1])
    for idx, (kind, il, token, ids) in enumerate(events):
        L = layers[il]
        decode = token > 0
        if decode:
            tokens.add(token)

        def score(e, ent):
            if policy == "opt":
                # furthest next demand first; never asked again = infinity
                pos = next_use[il].get(e)
                if pos is None:
                    return -float("inf")
                i = bisect.bisect_right(pos, idx)
                return -(pos[i] if i < len(pos) else float("inf"))
            if policy == "lru-cold":
                return ent["use"]
            if policy == "lru2":
                return (ent["prev"], ent["use"])
            if half is not None:
                f = ent["freq"] * 0.5 ** ((token - ent["ftok"]) / half)
                return (f, ent["use"])
            return ent["use"]

        def alloc(tok):
            if len(L.slot) < n_slots:
                return True
            best = None
            for pass_ in (0, 1):
                for e, ent in L.slot.items():
                    if ent["used"] == tok:
                        continue
                    if pass_ == 0 and ent["pref"] == tok:
                        continue
                    s = score(e, ent)
                    if best is None or s < best[0]:
                        best = (s, e)
                if best is not None:
                    break
            if best is None:
                return False
            del L.slot[best[1]]
            return True

        def bump(ent, tok):
            ent["prev"] = ent["use"]
            if half is not None:
                ent["freq"] = ent["freq"] * 0.5 ** ((tok - ent["ftok"]) / half) + 1.0
                ent["ftok"] = tok

        if kind == "S":
            for e in ids:
                ent = L.slot.get(e)
                if ent is None:
                    if not alloc(token):
                        continue
                    ent = L.slot[e] = {"used": -1, "pref": -1, "use": 0, "freq": 0.0, "ftok": token, "prev": -1}
                ent["used"] = token
                bump(ent, token)
                clock += 1
                ent["use"] = clock
        elif kind == "P":
            for e in ids:
                ent = L.slot.get(e)
                if ent is not None:
                    ent["pref"] = token
                    continue
                if not alloc(token):
                    no_slot += decode
                    continue
                ent = L.slot[e] = {"used": -1, "pref": token, "use": 0, "freq": 0.0, "ftok": token, "prev": -1}
                clock += 1
                # lru-cold: an unconfirmed guess sorts as older than everything the graph used
                ent["use"] = -clock if policy == "lru-cold" else clock
                if decode:
                    reads += 1
                    pref_reads += 1
        else:  # D
            for e in ids:
                ent = L.slot.get(e)
                if ent is None:
                    alloc(token)  # the graph waits until something is evictable; never fails here
                    ent = L.slot[e] = {"used": -1, "pref": -1, "use": 0, "freq": 0.0, "ftok": token, "prev": -1}
                    if decode:
                        misses += 1
                        reads += 1
                ent["used"] = token
                bump(ent, token)
                clock += 1
                ent["use"] = clock
    n = max(1, len(tokens))
    return misses / n, reads / n, pref_reads / n, no_slot / n


def main():
    args = sys.argv[1:]
    opts = {}
    for k in ("--slots", "--policies"):
        if k in args:
            i = args.index(k)
            opts[k] = args[i + 1]
            del args[i:i + 2]
    policies = opts.get("--policies", "lru,lru-cold,lru2,lfu-8,lfu-32,opt").split(",")
    for path in args:
        events, slots, experts = load(path)
        slot_list = [int(s) for s in opts["--slots"].split(",")] if "--slots" in opts else [slots]
        for kind_filter in ("with prefetch", "no prefetch"):
            ev = events if kind_filter == "with prefetch" else [x for x in events if x[0] != "P"]
            next_use = defaultdict(dict)
            for idx, (kind, il, token, ids) in enumerate(ev):
                if kind == "D":
                    for e in ids:
                        next_use[il].setdefault(e, []).append(idx)
            print(f"{path}: {experts} experts, {kind_filter}")
            for s in slot_list:
                for pol in policies:
                    m, r, pr, ns = replay(ev, s, pol, next_use if pol == "opt" else None)
                    print(f"  slots {s:3d} {pol:9s} misses {m:6.2f}  reads {r:6.2f}  (prefetch {pr:6.2f}, no-slot {ns:5.2f})")


if __name__ == "__main__":
    main()
