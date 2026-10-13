"""List the MoE expert tensors of one layer of a GGUF: names, types, shapes and bytes per expert.

    python bench/gguf-experts.py C:\\models\\model.gguf [--layer 0]

Uses the gguf-py package from the llama.cpp clone, so it reads exactly what llama.cpp reads.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    r = GGUFReader(args.model)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    print(f"arch {arch}, {len(r.tensors)} tensors")
    for key in ("expert_count", "expert_used_count", "block_count", "expert_feed_forward_length"):
        f = r.fields.get(f"{arch}.{key}")
        if f is not None:
            print(f"  {key}: {f.parts[-1][0]}")

    prefix = f"blk.{args.layer}."
    total = 0
    for t in r.tensors:
        if t.name.startswith(prefix) and ("exps" in t.name or "gate_inp" in t.name):
            shape = list(map(int, t.shape))
            per_expert = int(t.n_bytes) // shape[-1] if "exps" in t.name and t.name.endswith(".weight") else 0
            total += per_expert
            print(f"  {t.name:36} {t.tensor_type.name:8} {shape}  {int(t.n_bytes) / 2**20:8.1f} MiB"
                  + (f"  {per_expert / 1024:7.0f} KiB/expert" if per_expert else ""))
    print(f"bytes per expert (all weight chunks): {total / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
