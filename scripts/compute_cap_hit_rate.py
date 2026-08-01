"""
Project 71 -- retrospective cap-hit rate over a full-suite generations.jsonl.

cap-hit = a response that reaches max_new_tokens (per configs/eval_generation.yaml,
currently 512) WITHOUT ever emitting EOS -- i.e. failure to terminate, not verbosity.

WHY RETROSPECTIVE, NOT LOGGED AT GENERATION TIME: eval_generate.py decodes with
skip_special_tokens=True and slices `out[i, prompt_len:]` to the batch's full generated
width; both a genuine EOS token (151645) and any trailing pad-after-EOS filler
(pad_token_id=151643, which is ITSELF one of the two configured eos_token_ids) are
stripped by skip_special_tokens. So the two cases -- "hit EOS early" and "ran the full
512 steps and never emitted EOS" -- are NOT distinguishable from a literal special-token
marker left in the saved text; no per-item generated-token-count field was persisted
either. The only retrospective signal is re-tokenizing the saved response text and
counting tokens.

CPU-only, no GPU required: loads the same tokenizer (base Qwen2.5-7B-Instruct at the
pinned revision) used for generation and re-encodes response_continuation with
add_special_tokens=False.

KNOWN LIMITATION, stated plainly rather than hidden in a threshold choice: encode(decode(ids))
is not guaranteed to reproduce the original per-step token IDs exactly for a BPE tokenizer
(re-merging at the boundary between two adjacent generated tokens can occasionally collapse
them into fewer sub-word units). This tool therefore uses a small tolerance band around the
configured cap rather than requiring an exact match, and reports the raw token-count
distribution near the cap so a reader can see whether the mass sits at the boundary
(consistent with true cap-hits) or is spread out (which would suggest the tolerance is
doing real work rather than absorbing noise).
"""

import argparse
import json
from collections import Counter

import yaml
from transformers import AutoTokenizer


def load_generation_records(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("record_type") == "generation":
                rows.append(r)
    if not rows:
        raise RuntimeError(f"Loaded 0 generation records from {path} -- fail loudly, not silently.")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=str, required=True)
    ap.add_argument("--gen_config", type=str, default="configs/eval_generation.yaml")
    ap.add_argument("--tolerance", type=int, default=4,
                     help="cap-hit if re-tokenized length >= max_new_tokens - tolerance")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    with open(args.gen_config, "r", encoding="utf-8") as f:
        gen_cfg = yaml.safe_load(f)
    max_new_tokens = gen_cfg["decoding"]["max_new_tokens"]
    model_name = gen_cfg["model"]["name_or_path"]
    revision = gen_cfg["model"]["revision"]
    cache_dir = gen_cfg["model"]["cache_dir"]

    print(f"[tokenizer] {model_name} rev={revision} (CPU, no GPU)")
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)

    rows = load_generation_records(args.generations)
    threshold = max_new_tokens - args.tolerance

    results = []
    for r in rows:
        text = r.get("response_continuation", "")
        n_tok = len(tok.encode(text, add_special_tokens=False)) if text else 0
        cap_hit = n_tok >= threshold
        results.append({"id": r["id"], "category": r["category"], "n_tok": n_tok, "cap_hit": cap_hit})

    n_total = len(results)
    n_cap_hit = sum(1 for x in results if x["cap_hit"])
    pct = round(100.0 * n_cap_hit / n_total, 1)

    by_cat = Counter()
    tot_by_cat = Counter()
    for x in results:
        tot_by_cat[x["category"]] += 1
        if x["cap_hit"]:
            by_cat[x["category"]] += 1

    # gap check: how many items terminate in [384, threshold) -- a near-empty band here is
    # the signature the coordinator's parallel analysis reported for B2 v3 (bimodal: clean
    # short completions vs. full-length runaway, nothing in between).
    n_in_gap = sum(1 for x in results if 384 <= x["n_tok"] < threshold)

    print(f"[source] {args.generations}")
    print(f"[max_new_tokens] {max_new_tokens}  [tolerance] {args.tolerance}  [threshold] >= {threshold}")
    print(f"[CAP-HIT] {n_cap_hit}/{n_total} ({pct}%)")
    for cat in sorted(tot_by_cat):
        d, t = by_cat[cat], tot_by_cat[cat]
        print(f"    {cat}: {d}/{t} ({100*d/t:.1f}%)")
    print(f"[gap check] items terminating in [384, {threshold}): {n_in_gap}/{n_total}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "source": args.generations,
                "max_new_tokens": max_new_tokens,
                "tolerance": args.tolerance,
                "threshold": threshold,
                "n_total": n_total,
                "n_cap_hit": n_cap_hit,
                "pct_cap_hit": pct,
                "by_category": {c: {"n_cap_hit": by_cat[c], "n_total": tot_by_cat[c]} for c in tot_by_cat},
                "n_in_gap_384_to_threshold": n_in_gap,
                "items": results,
            }, f, indent=2)
        print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
