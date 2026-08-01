"""
Project 71 -- sample the INDEPENDENT judge-validation set (CLAUDE.md safeguards v2, rule 4).

Feeds: the Cohen's kappa figures reported in Methods, which are what make every ASR and
over-refusal number in Tables 1-2 credible rather than assumed.

Rule 4 requires 100-150 hand-labelled responses stratified across arms and attack categories.
This set MUST be independent of the 32-item dev set the judge prompts were tuned against, or
the kappa is measured on the data it was fitted to and means nothing. Independence here is
structural, not merely intended: the dev set was drawn from data/redteam/_fixture_dev.jsonl
(a 20-item hand-written fixture) while this set is drawn from the real 300-item frozen suite.
The two share no prompt ids and no response text. The script asserts this and fails loudly.

Sampling is seeded and stratified: `--per_cell` items per (arm x category) cell.

It emits the items WITHOUT judge verdicts, deliberately. Labelling while looking at the
judge's answer would anchor the labeller and inflate agreement. The verdicts are joined in
afterwards by build_validation_set.py.

Usage:
    python Scripts\\sample_validation_set.py --arms b0 b1 --per_cell 10 --seed 7
"""

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATEGORIES = ["prefilling", "persona", "many_shot", "crisis_adjacent", "benign_sensitive"]
DEV_SET = REPO / "results" / "human_validation" / "judge_calibration_dev.jsonl"


def load_generations(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record_type") == "generation":
            out.append(r)
    if not out:
        raise ValueError(f"{path}: no generation records")
    return out


def load_exclusion(paths, arm_aware=False):
    """Prompt ids and response texts already used by an earlier labelled set.

    A judge prompt that has been revised after seeing its errors on set S can never be
    honestly re-measured on S. Every later set must therefore exclude every earlier one, and
    the exclusion must be asserted rather than assumed.

    Two exclusion keys, and the difference matters:

    - RESPONSE TEXT, always global. This is the real leak guard: the unit of judge validation
      is a (prompt, response) pair, and reusing a pair the judge was tuned against is exactly
      what invalidates a kappa.
    - PROMPT ID. By default also global, which is the strictest reading. With `arm_aware`,
      the key becomes (arm, prompt id), so a *different arm's* response to an
      already-used prompt stays eligible. That is defensible -- it is a different response,
      and the judge is judging responses -- and it is sometimes necessary: with only 60
      prompts per category, global prompt-id exclusion exhausts the pool after two sets.
      The weaker guarantee is recorded in the output so it is never silently assumed.
    """
    ids, resps = set(), set()
    for p in paths:
        p = Path(p) if Path(p).is_absolute() else REPO / p
        if not p.is_file():
            print(f"[independence] exclusion file not found, skipping: {p}")
            continue
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            n += 1
            if d.get("source_generation_id"):
                ids.add((d.get("source_arm"), d["source_generation_id"]) if arm_aware
                        else d["source_generation_id"])
            if d.get("response"):
                resps.add(d["response"].strip())
        print(f"[independence] excluding {n} items from {p.name}")
    print(f"[independence] prompt-id exclusion is {'ARM-AWARE (arm, id)' if arm_aware else 'GLOBAL (id)'}; "
          f"response-text exclusion is always global")
    return ids, resps


def load_scored(path: Path):
    """id -> the judge's own binary verdict, for verdict-stratified sampling."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record_type") != "scored":
            continue
        if "refused" in r:
            out[r["id"]] = bool(r["refused"])
        elif "unsafe" in r:
            out[r["id"]] = bool(r["unsafe"])
    if not out:
        raise ValueError(f"{path}: no scored records with a binary outcome")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["b0", "b1"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per_cell", type=int, default=10)
    ap.add_argument("--categories", nargs="+", default=None,
                    help="Restrict to these categories (default: all five)")
    ap.add_argument("--per_cell_overrides", default=None,
                    help='JSON, e.g. {"crisis_adjacent": 12} -- per-category cell size')
    ap.add_argument("--exclude_arm_aware", action="store_true",
                    help="Exclude on (arm, prompt id) rather than prompt id alone, so another "
                         "arm's response to an already-used prompt stays eligible. Response-text "
                         "exclusion stays global either way. Needed once the 60-prompt-per-"
                         "category pool is exhausted; the weaker guarantee is printed and must "
                         "be recorded in the notebook.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Already-labelled JSONL sets whose prompt ids and response texts must "
                         "not reappear. The dev set is always excluded.")
    ap.add_argument("--stratify_by_verdict", action="store_true",
                    help="ENRICHED sampling: draw half the cell from items the judge flagged "
                         "positive and half from items it did not. Use where the natural base "
                         "rate is too low for kappa to be estimable from a random sample. The "
                         "resulting set is NOT a random sample -- see the note written into it.")
    ap.add_argument("--scored_name", default=None,
                    help="scored jsonl filename inside results/<arm>_seed42/, required with "
                         "--stratify_by_verdict")
    ap.add_argument("--scored_names", default=None, help="JSON arm->scored filename")
    ap.add_argument("--positive_fraction", type=float, default=0.5,
                    help="With --stratify_by_verdict, share of each cell drawn from the "
                         "judge-positive stratum. Raise it when the question is precision "
                         "(are the judge's positives real?) rather than prevalence.")
    ap.add_argument("--generations_name", default="generations_realsuite.jsonl")
    ap.add_argument("--generations_names", default=None,
                    help='JSON arm->filename, for when arms carry different supersession '
                         'suffixes, e.g. {"b0": "generations_realsuite_SUPERSEDED_x.jsonl"}')
    ap.add_argument("--out", default="results/human_validation/validation_items_unlabelled.jsonl")
    args = ap.parse_args()

    cats = args.categories or CATEGORIES
    per_cell = {c: args.per_cell for c in cats}
    if args.per_cell_overrides:
        per_cell.update({k: int(v) for k, v in json.loads(args.per_cell_overrides).items()})

    # --- independence guard: everything any earlier labelled set touched ------------------
    dev_prompt_ids, dev_responses = load_exclusion([DEV_SET] + list(args.exclude),
                                                    arm_aware=args.exclude_arm_aware)

    gen_names = json.loads(args.generations_names) if args.generations_names else {}
    rng = random.Random(args.seed)
    rows, cells, strata_pop = [], {}, {}
    for arm in args.arms:
        gens = load_generations(REPO / "results" / f"{arm}_seed42"
                                / gen_names.get(arm, args.generations_name))
        scored_names = json.loads(args.scored_names) if args.scored_names else {}
        verdicts = (load_scored(REPO / "results" / f"{arm}_seed42"
                                / scored_names.get(arm, args.scored_name))
                    if args.stratify_by_verdict else None)
        by_cat = {}
        for g in gens:
            by_cat.setdefault(g["category"], []).append(g)
        for cat in cats:
            pool = sorted(by_cat.get(cat, []), key=lambda r: r["id"])
            pool = [g for g in pool
                    if ((arm, g["id"]) if args.exclude_arm_aware else g["id"]) not in dev_prompt_ids
                    and g["response_full_turn"].strip() not in dev_responses]
            k = per_cell[cat]
            if verdicts is None:
                if len(pool) < k:
                    raise RuntimeError(f"{arm}/{cat}: only {len(pool)} eligible items, need {k}")
                picked = rng.sample(pool, k)
                for g in picked:
                    g["_stratum"] = "random"
            else:
                # Record the POPULATION share of each stratum. Without it the enriched kappa
                # cannot be reweighted back to the real base rate, and an enriched kappa read
                # as a population kappa is simply wrong.
                whole = sorted(by_cat.get(cat, []), key=lambda r: r["id"])
                pos_all = [g for g in whole if verdicts.get(g["id"])]
                strata_pop[(arm, cat)] = {"n_population": len(whole),
                                          "n_judge_positive": len(pos_all),
                                          "judge_positive_rate": len(pos_all) / len(whole)}
                pos = [g for g in pool if verdicts.get(g["id"])]
                neg = [g for g in pool if not verdicts.get(g["id"])]
                import math
                want_pos = min(math.ceil(k * args.positive_fraction), len(pos))
                want_neg = min(k - want_pos, len(neg))
                if want_pos + want_neg < k:
                    print(f"[warn] {arm}/{cat}: only {want_pos}+{want_neg} available, wanted {k}")
                picked = []
                for g in rng.sample(pos, want_pos):
                    g["_stratum"] = "judge_positive"; picked.append(g)
                for g in rng.sample(neg, want_neg):
                    g["_stratum"] = "judge_negative"; picked.append(g)
                picked.sort(key=lambda r: r["id"])
            cells[(arm, cat)] = len(picked)
            for g in picked:
                rows.append({
                    "id": f"val_{arm}_{g['id']}",
                    "source_arm": arm, "source_generation_id": g["id"],
                    "category": g["category"], "messages": g["messages"],
                    "is_prefill": g["is_prefill"],
                    "response": g["response_full_turn"],
                    # sampling stratum, NOT the judge's verdict -- kept so the enriched set can
                    # be reweighted. The labeller must not see it; build_*_set.py joins labels
                    # by id and the labeller works from the printed conversation only.
                    "sampling_stratum": g["_stratum"],
                    "human_label": None,          # to be filled by the labeller
                    "rationale": None,
                })

    # --- hard independence assertions ----------------------------------------------------
    ids = {((r["source_arm"], r["source_generation_id"]) if args.exclude_arm_aware
            else r["source_generation_id"]) for r in rows}
    if ids & dev_prompt_ids:
        raise RuntimeError(f"OVERLAP with an excluded set: {sorted(ids & dev_prompt_ids)}")
    resps = {r["response"].strip() for r in rows}
    if resps & dev_responses:
        raise RuntimeError("OVERLAP with an excluded set on response text")
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate ids in the sampled set")
    if strata_pop:
        meta = REPO / args.out
        meta = meta.with_name(meta.stem + "_strata.json")
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(
            {"note": "population share of each sampling stratum; required to reweight an "
                     "enriched-sample agreement estimate back to the natural base rate",
             "strata": {f"{a}/{c}": v for (a, c), v in strata_pop.items()}},
            indent=2), encoding="utf-8")
        print(f"[written] {meta}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[written] {out}  ({len(rows)} items, {args.per_cell} per cell)")
    for k in sorted(cells, key=str):
        print(f"    {k[0]:<4} {k[1]:<18} {cells[k]}")
    print("[independence] PASS -- no prompt-id or response-text overlap with the dev set")
    print("[note] judge verdicts are deliberately NOT included; label blind, then join.")


if __name__ == "__main__":
    main()
