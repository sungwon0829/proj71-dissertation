"""Leakage audit: red-team eval suite vs. every training corpus.

Which paper number this feeds: it does not produce a table cell -- it protects
every cell. Item-level overlap between the frozen eval suite and any training
set invalidates the headline ASR comparison (CLAUDE.md "THE FREEZE";
CLAUDE_safeguards_v2.md section 3, which requires overlap statistics to be
reported in Methods even when provenance is independent by construction).

Three tests, in increasing strictness:
  T1 EXACT      normalised full-text equality between any red-team message and
                any training message.
  T2 VERBATIM   any shared contiguous 12-word span (the strong leakage signal;
                natural language almost never shares a 12-gram by chance).
  T3 NEAR-DUP   max word-5-gram Jaccard similarity per red-team item against
                every training record. Reported as a distribution.

Memory strategy: index the SMALL side (300 eval items) and stream the LARGE side
(pref_helpful.jsonl is ~120 MB), so peak memory stays flat.

Also reports T4: internal near-duplicate pairs WITHIN the suite, because
near-duplicate attacks inflate apparent robustness.

Usage:
  C:\\proj71\\env\\Scripts\\python.exe C:\\proj71\\scripts\\check_redteam_leakage.py
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "data" / "redteam" / "redteam_suite.jsonl"
TRAIN_FILES = [
    REPO / "data" / "processed" / "sft_train.jsonl",
    REPO / "data" / "processed" / "sft_esconv.jsonl",
    REPO / "data" / "processed" / "sft_counsel.jsonl",
    REPO / "data" / "processed" / "pref_safety.jsonl",
    REPO / "data" / "processed" / "pref_helpful.jsonl",
]

NGRAM = 5
SPAN = 12
NEAR_DUP_REPORT = 0.30
INTERNAL_REPORT = 0.40

# T1 only counts messages long enough for a match to be meaningful. Single-token
# demonstration turns in many_shot items ("Yes.", "Fine.", "Correct.") collide
# with thousands of ordinary training utterances; those are token collisions,
# not leakage, and are reported separately as a diagnostic.
MIN_EXACT_WORDS = 8

_punct = re.compile(r"[^a-z0-9\s]+")
_ws = re.compile(r"\s+")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = _punct.sub(" ", text)
    return _ws.sub(" ", text).strip()


def words(text: str) -> list[str]:
    return norm(text).split()


def shingles(toks: list[str], n: int) -> set[int]:
    if len(toks) < n:
        return {hash(" ".join(toks))} if toks else set()
    return {hash(" ".join(toks[i : i + n])) for i in range(len(toks) - n + 1)}


def record_texts(rec: dict) -> list[str]:
    """Every free-text field of a training record, whatever the schema."""
    out: list[str] = []
    for key in ("messages", "prompt", "chosen", "rejected"):
        val = rec.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            out.append(val)
        elif isinstance(val, list):
            for m in val:
                if isinstance(m, dict) and m.get("role") != "system":
                    out.append(str(m.get("content", "")))
                elif isinstance(m, str):
                    out.append(m)
        elif isinstance(val, dict):
            out.append(str(val.get("content", "")))
    return [t for t in out if t and t.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(SUITE))
    args = ap.parse_args()

    suite = [json.loads(l) for l in Path(args.suite).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"suite items: {len(suite)}")

    # --- index the small side -------------------------------------------------
    item_tokens: dict[str, list[str]] = {}
    item_5g: dict[str, set[int]] = {}
    item_12g: dict[str, set[int]] = {}
    exact_norms: dict[str, set[str]] = defaultdict(set)  # normalised msg -> ids (>= MIN_EXACT_WORDS)
    short_norms: dict[str, set[str]] = defaultdict(set)  # normalised msg -> ids (short, diagnostic)
    inv5: dict[int, set[str]] = defaultdict(set)
    inv12: dict[int, set[str]] = defaultdict(set)

    for rec in suite:
        rid = rec["id"]
        joined = " ".join(m["content"] for m in rec["messages"])
        toks = words(joined)
        item_tokens[rid] = toks
        item_5g[rid] = shingles(toks, NGRAM)
        item_12g[rid] = shingles(toks, SPAN)
        for sh in item_5g[rid]:
            inv5[sh].add(rid)
        for sh in item_12g[rid]:
            inv12[sh].add(rid)
        for m in rec["messages"]:
            n = norm(m["content"])
            if not n:
                continue
            if len(n.split()) >= MIN_EXACT_WORDS:
                exact_norms[n].add(rid)
            else:
                short_norms[n].add(rid)

    best_jaccard: dict[str, tuple[float, str]] = {rid: (0.0, "-") for rid in item_5g}
    exact_hits: list[tuple[str, str]] = []
    short_collisions: dict[str, int] = defaultdict(int)
    verbatim_hits: list[tuple[str, str, str]] = []

    # --- stream the large side ------------------------------------------------
    for path in TRAIN_FILES:
        if not path.exists():
            print(f"  [skip] {path.name} (does not exist)")
            continue
        n_rec = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                n_rec += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for text in record_texts(rec):
                    nt = norm(text)
                    if nt in exact_norms:
                        for rid in exact_norms[nt]:
                            exact_hits.append((rid, path.name))
                    if nt in short_norms:
                        short_collisions[nt] += 1
                    toks = nt.split()
                    if len(toks) < NGRAM:
                        continue

                    t12 = shingles(toks, SPAN)
                    for sh in t12 & inv12.keys():
                        for rid in inv12[sh]:
                            verbatim_hits.append((rid, path.name, text[:160]))

                    t5 = shingles(toks, NGRAM)
                    cand: dict[str, int] = defaultdict(int)
                    for sh in t5:
                        for rid in inv5.get(sh, ()):
                            cand[rid] += 1
                    for rid, inter in cand.items():
                        union = len(t5) + len(item_5g[rid]) - inter
                        j = inter / union if union else 0.0
                        if j > best_jaccard[rid][0]:
                            best_jaccard[rid] = (j, path.name)
        print(f"  scanned {path.name}: {n_rec} records")

    # --- report ---------------------------------------------------------------
    print(f"\n=== T1 EXACT normalised message match (>= {MIN_EXACT_WORDS} words) ===")
    print(f"  hits: {len(exact_hits)}")
    for rid, src in exact_hits[:20]:
        print(f"    {rid}  <-  {src}")
    print(f"  [diagnostic] short-utterance token collisions (< {MIN_EXACT_WORDS} words, not leakage):")
    for text, n in sorted(short_collisions.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {n:>6}x  {text!r}  (ids: {sorted(short_norms[text])})")

    print(f"\n=== T2 VERBATIM shared {SPAN}-word span ===")
    print(f"  hits: {len(verbatim_hits)}")
    for rid, src, snippet in verbatim_hits[:20]:
        print(f"    {rid}  <-  {src}: {snippet!r}")

    print(f"\n=== T3 NEAR-DUP max word-{NGRAM}-gram Jaccard per item ===")
    vals = sorted(((j, rid, src) for rid, (j, src) in best_jaccard.items()), reverse=True)
    top = vals[0][0] if vals else 0.0
    mean = sum(v[0] for v in vals) / len(vals) if vals else 0.0
    print(f"  max over all 300 items: {top:.4f}")
    print(f"  mean over all 300 items: {mean:.4f}")
    over = [v for v in vals if v[0] >= NEAR_DUP_REPORT]
    print(f"  items with Jaccard >= {NEAR_DUP_REPORT}: {len(over)}")
    print("  top 10:")
    for j, rid, src in vals[:10]:
        print(f"    {j:.4f}  {rid}  (best match in {src})")

    print(f"\n=== T4 INTERNAL near-duplicates within the suite (>= {INTERNAL_REPORT}) ===")
    ids = list(item_5g)
    pairs: list[tuple[float, str, str]] = []
    for a_i in range(len(ids)):
        a = ids[a_i]
        sa = item_5g[a]
        for b_i in range(a_i + 1, len(ids)):
            b = ids[b_i]
            sb = item_5g[b]
            inter = len(sa & sb)
            if not inter:
                continue
            j = inter / (len(sa) + len(sb) - inter)
            if j >= INTERNAL_REPORT:
                pairs.append((j, a, b))
    pairs.sort(reverse=True)
    print(f"  pairs above threshold: {len(pairs)}")
    for j, a, b in pairs[:15]:
        print(f"    {j:.4f}  {a}  ~  {b}")

    print("\n=== VERDICT ===")
    clean = not exact_hits and not verbatim_hits
    print("  T1 exact overlap:    " + ("NONE" if not exact_hits else f"{len(exact_hits)} HITS"))
    print("  T2 verbatim spans:   " + ("NONE" if not verbatim_hits else f"{len(verbatim_hits)} HITS"))
    print(f"  T3 max similarity:   {top:.4f}")
    print(f"  T4 internal dup pairs: {len(pairs)}")
    print("  RESULT: " + ("NO LEAKAGE DETECTED" if clean else "LEAKAGE DETECTED - INVESTIGATE"))


if __name__ == "__main__":
    main()
