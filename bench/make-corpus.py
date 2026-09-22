"""Build a plain-English evaluation corpus from documentation already on the machine.

Used for *relative* perplexity comparisons (K=8 vs K=4, quant vs quant), where the
absolute number does not matter but the text must be the same across runs.

    python make-corpus.py [out.txt]
"""
import re
import sys
from pathlib import Path

GIT_DOC = Path(r"C:\Program Files\Git\mingw64\share\doc\git-doc")
FILES = [
    "user-manual.adoc", "gittutorial.adoc", "gittutorial-2.adoc", "giteveryday.adoc",
    "gitglossary.adoc", "gitworkflows.adoc", "gitcore-tutorial.adoc", "MyFirstContribution.adoc",
    "DecisionMaking.adoc", "ReviewingGuidelines.adoc", "SubmittingPatches.adoc", "CodingGuidelines.adoc",
    "gitcredentials.adoc", "gitrevisions.adoc", "gitsubmodules.adoc", "git-rebase.adoc",
    "git-merge.adoc", "git-bisect.adoc", "git-stash.adoc", "githooks.adoc",
]

LINK = re.compile(r"link(?:git)?:[^\[]+\[([^\]]*)\]")
XREF = re.compile(r"<<[^,>]+,\s*([^>]*)>>|<<([^>]+)>>")
ANCHOR = re.compile(r"\[\[[^\]]*\]\]")


def clean(text: str) -> str:
    out = []
    in_block = False
    for line in text.splitlines():
        s = line.rstrip()
        if s in ("----", "....", "====", "++++", "////"):
            in_block = not in_block
            continue
        if in_block:
            continue
        if not s or s.startswith((":", "[", "//", "ifdef", "ifndef", "endif", "include::", "|")):
            continue
        if re.fullmatch(r"[-=~^+]{3,}", s):  # setext-style underlines
            continue
        s = LINK.sub(r"\1", s)
        s = XREF.sub(lambda m: m.group(1) or m.group(2), s)
        s = ANCHOR.sub("", s)
        s = s.replace("`", "").replace("*", "").replace("'", "'")
        s = re.sub(r"^=+\s*", "", s)  # heading markers
        out.append(s)
    return "\n".join(out)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("corpus.txt"))
    parts = []
    for name in FILES:
        p = GIT_DOC / name
        if p.exists():
            parts.append(clean(p.read_text(encoding="utf-8", errors="replace")))
    text = "\n\n".join(parts)
    out.write_text(text, encoding="utf-8")
    print(f"{out}: {len(parts)} files, {len(text):,} chars, {len(text.split()):,} words")


if __name__ == "__main__":
    main()
