"""
Project 71 -- Cohen's kappa for the human ASR worksheet (results/human_validation/), plus
the gold-item attention-check pass rate, computed and reported SEPARATELY as required.

WHICH PAPER NUMBER: the headline HUMAN-vs-judge kappa reported in Methods, alongside the
inter-model kappas already pinned in configs/judges_pinned.lock.json (0.521 ASR /
0.583 crisis / 0.074 refusal -- all against an LLM reference labeller, per
notebook/preregistration.md CORRECTION 2026-08-01). This script produces the first genuine
human number for the ASR judge. It does not run until
results/human_validation/human_asr_labels.json has been filled in by the owner and unsealed
with `scripts/dump_human_asr_worksheet.py --unseal`.

WHY A SEPARATE SCRIPT FROM `eval_score.py --calibrate`: eval_score.py's run_calibration()
already computes per-judge, per-category kappa/CI/confusion against any hand-labelled
validation set, and THIS SCRIPT REUSES IT (imported, never reimplemented). What is specific
to this worksheet and not provided by the generic calibration path is:
  (a) the gold-item pass-rate check (this worksheet's only internal check on labelling
      attention -- preregistration_amendments.md Amendment 10 recorded its absence in the
      first build; this script is what makes the fix load-bearing rather than decorative);
  (b) EXCLUDING the 6 gold items from the headline kappa (they were deliberately chosen to
      be unambiguous and would inflate agreement relative to a representative sample --
      "the gold-item pass rate reported separately from kappa" is not optional phrasing);
  (c) failing loudly, before any model loads, on an incomplete label file, a missing gold
      item, or a non-human labeller -- none of which the generic calibration path checks
      for, because it has no concept of "this specific worksheet's gold items".

FAIL-LOUD CONTRACT
  - Refuses to run (clear, instructive error, no GPU touched) if the unsealed validation set
    does not exist yet, is incomplete, contains a non-'safe'/'unsafe' label, or has ANY row
    with labeller_is_human != True. This project has already had one incident of agent
    labels reported as human labels (results/human_validation/LABEL_PROVENANCE_CORRECTION.md);
    this script is a second, independent gate against a repeat.
  - Computes the gold-item pass rate BEFORE loading any judge model (it needs only the
    owner's given labels and the pre-registered gold answer key, not a judge verdict), and
    prints it first.
  - By DEFAULT refuses to compute or report kappa if any gold item failed -- a labeller who
    misses an unambiguous item is a reason to distrust their labels on the ambiguous ones,
    which is the entire point of embedding the check. `--allow_gold_failures` overrides
    this, matching the project's existing smoke-test-abort convention
    (scripts/probe_gold_items.py), and stamps every number that follows NOT_TRUSTWORTHY.

Usage:
    python Scripts\\compute_kappa.py
    python Scripts\\compute_kappa.py --seed 7
    python Scripts\\compute_kappa.py --allow_gold_failures   # only after a failed gold check
"""

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_score import (  # noqa: E402
    ATTACK_CATEGORIES, agreement_block, build_judges, cohens_kappa, judge_independence_block,
    kappa_ci, load_yaml, resolve, run_calibration, sha256_file, verify_judge_pin,
)

REPO = Path(__file__).resolve().parent.parent
HV = REPO / "results" / "human_validation"


def sha256_path(path: Path) -> str:
    return sha256_file(path) if path.is_file() else None


# --------------------------------------------------------------------------------------
def load_validation_set(path: Path) -> list:
    """FAIL LOUDLY on anything short of a complete, genuinely-human-labelled set."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. This worksheet has not been labelled and unsealed yet. "
            f"Run:\n"
            f"  1. Fill in results/human_validation/human_asr_labels.json (60 items).\n"
            f"  2. python Scripts/dump_human_asr_worksheet.py --unseal --labels "
            f"results/human_validation/human_asr_labels.json\n"
            f"then re-run this script. There is nothing to compute yet.")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rows.append((lineno, json.loads(line)))
    if not rows:
        raise ValueError(f"{path} exists but contains zero items.")

    problems = []
    for lineno, r in rows:
        rid = r.get("id", f"line {lineno}")
        for fld in ("id", "item_key", "category", "reference_label", "labeller_is_human"):
            if fld not in r:
                problems.append(f"{rid}: missing required field {fld!r}")
        if r.get("reference_label") not in ("safe", "unsafe"):
            problems.append(f"{rid}: reference_label is {r.get('reference_label')!r}, "
                            f"must be 'safe' or 'unsafe' -- refusing to silently coerce it")
        if r.get("labeller_is_human") is not True:
            problems.append(
                f"{rid}: labeller_is_human={r.get('labeller_is_human')!r}, not True. This "
                f"worksheet exists SPECIFICALLY to produce a genuine human kappa "
                f"(preregistration.md CORRECTION 2026-08-01); a non-human row here would "
                f"repeat the exact incident that correction documents. Refusing to compute "
                f"anything until every row is labeller_is_human=true.")
        if not str(r.get("id", "")).startswith("human_"):
            problems.append(f"{rid}: id does not start with 'human_' -- not produced by "
                            f"dump_human_asr_worksheet.py --unseal as expected")
    if problems:
        raise ValueError(f"{path}: {len(problems)} problem(s), refusing to proceed:\n  " +
                         "\n  ".join(problems[:30]) +
                         (f"\n  ... and {len(problems) - 30} more" if len(problems) > 30 else ""))
    return [r for _, r in rows]


def load_gold_answer_key(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. It is written by scripts/dump_human_asr_worksheet.py "
            f"at build time; if the worksheet exists but this file does not, something has "
            f"gone wrong with the build, not with labelling. Refusing to proceed without it "
            f"-- there is no way to check the gold items otherwise.")
    return json.loads(path.read_text(encoding="utf-8"))


def check_sealed_key_counts(sealed_key_path: Path, n_rows: int, n_gold: int):
    """Cross-check against the sealed key's own recorded n / n_gold_items, if present."""
    if not sealed_key_path.is_file():
        print(f"[warn] {sealed_key_path} not found; skipping the sealed-key count cross-check "
              f"(not fatal -- the sealed key is only needed to UNSEAL labels, already done).")
        return
    key = json.loads(sealed_key_path.read_text(encoding="utf-8"))
    if key.get("n") is not None and key["n"] != n_rows:
        raise ValueError(f"validation set has {n_rows} rows but the sealed key that produced "
                         f"it records n={key['n']}. Refusing to proceed on a mismatched set.")
    if key.get("n_gold_items") is not None and key["n_gold_items"] != n_gold:
        raise ValueError(f"gold answer key has {n_gold} items but the sealed key records "
                         f"n_gold_items={key['n_gold_items']}. Refusing to proceed.")


def compute_gold_block(rows: list, gold_key: dict) -> dict:
    """The attention-check pass rate. Needs only the owner's given labels and the
    pre-registered gold reference -- no judge, no GPU."""
    by_item_key = {r["item_key"]: r for r in rows}
    gold_items = gold_key["items"]
    missing = [k for k in gold_items if k not in by_item_key]
    if missing:
        raise ValueError(
            f"{len(missing)} gold item_key(s) from {gold_key.get('n_gold_items')} declared "
            f"gold items are absent from the validation set: {missing}. The validation set "
            f"does not match the worksheet this gold answer key was built for. Refusing to "
            f"score a mismatched pair.")
    results = []
    for k, gmeta in gold_items.items():
        row = by_item_key[k]
        given = row["reference_label"]
        expected = gmeta["reference_label"]
        results.append({
            "item_key": k, "category": gmeta["category"], "given_label": given,
            "expected_label": expected, "passed": given == expected,
            "rubric_clause_quoted": gmeta.get("rubric_clause_quoted"),
        })
    n = len(results)
    n_passed = sum(r["passed"] for r in results)
    return {
        "n_gold": n, "n_passed": n_passed,
        "gold_pass_rate": round(n_passed / n, 4) if n else None,
        "all_passed": n_passed == n,
        "results": results,
        "failed_items": [r for r in results if not r["passed"]],
        "note": ("This is a labelling-ATTENTION check, not a judge-performance measurement: "
                "n is small by design and every item was deliberately chosen to be "
                "unambiguous. A failure here is evidence about the LABELLING PASS, not "
                "about the judge."),
    }


def per_judge_per_category(rows: list, judge_names: list, seed: int) -> dict:
    """Cohen's kappa, CI and confusion, cross-tabulated by judge AND by category, reusing
    eval_score.agreement_block()/cohens_kappa()/kappa_ci() -- never reimplemented."""
    out = {}
    cats = sorted({r["category"] for r in rows})
    off_scope = sorted(set(cats) - set(ATTACK_CATEGORIES))
    if off_scope:
        raise ValueError(f"unexpected categories in the validation set: {off_scope}. This "
                         f"worksheet only samples {ATTACK_CATEGORIES}.")
    for jn in judge_names:
        out[jn] = {}
        for cat in cats:
            sel = [r for r in rows if r["category"] == cat]
            human = [r["reference_label"] for r in sel]
            judge = [r["per_judge"][jn]["verdict"] for r in sel]
            positive = next(r["per_judge"][jn].get("_positive_label") for r in sel) \
                if sel and "_positive_label" in sel[0]["per_judge"][jn] else "unsafe"
            blk = agreement_block(human, judge, positive, [r["id"] for r in sel], seed=seed)
            out[jn][cat] = blk
    return out


def pooled_per_judge(rows: list, judge_names: list, positive_by_judge: dict, seed: int) -> dict:
    out = {}
    for jn in judge_names:
        human = [r["reference_label"] for r in rows]
        judge = [r["per_judge"][jn]["verdict"] for r in rows]
        blk = agreement_block(human, judge, positive_by_judge[jn], [r["id"] for r in rows],
                              seed=seed)
        blk["categories_covered"] = sorted({r["category"] for r in rows})
        out[jn] = blk
    return out


def pooled_per_category(rows: list, seed: int) -> dict:
    out = {}
    for cat in sorted({r["category"] for r in rows}):
        sel = [r for r in rows if r["category"] == cat]
        blk = agreement_block([r["reference_label"] for r in sel],
                              [r["combined_verdict"] for r in sel], "unsafe",
                              [r["id"] for r in sel], seed=seed)
        out[cat] = blk
    return out


def main():
    ap = argparse.ArgumentParser(description="Project 71 -- human ASR worksheet kappa + gold check")
    ap.add_argument("--validation_set", default="results/human_validation/human_asr_validation_set.jsonl")
    ap.add_argument("--gold_answer_key", default="results/human_validation/human_asr_GOLD_ANSWER_KEY.json")
    ap.add_argument("--sealed_key", default="results/human_validation/human_asr_SEALED_KEY.json")
    ap.add_argument("--config", default="configs/judges.yaml")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--audit_text", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap CI seed for the "
                    "gold-excluded kappa numbers this script computes itself.")
    ap.add_argument("--allow_gold_failures", action="store_true",
                    help="Proceed even if a gold item was mislabelled. Every number in the "
                         "output is then stamped NOT_TRUSTWORTHY.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow_overwrite", action="store_true")
    args = ap.parse_args()

    t0 = datetime.datetime.now().astimezone()
    vs_path = resolve(args.validation_set)
    gold_path = resolve(args.gold_answer_key)
    sealed_path = resolve(args.sealed_key)

    # ---- stage 1: everything that needs no model, no GPU, and must fail loudly ----------
    rows = load_validation_set(vs_path)
    gold_key = load_gold_answer_key(gold_path)
    check_sealed_key_counts(sealed_path, len(rows), len(gold_key["items"]))
    print(f"[labels] {len(rows)} items loaded from {vs_path}, all labeller_is_human=true")

    gold_block = compute_gold_block(rows, gold_key)
    print("\n===== GOLD-ITEM ATTENTION CHECK (n={}) =====".format(gold_block["n_gold"]))
    print(f"  passed: {gold_block['n_passed']}/{gold_block['n_gold']}  "
          f"pass_rate={gold_block['gold_pass_rate']}")
    for r in gold_block["failed_items"]:
        print(f"  *** FAILED *** {r['item_key']} ({r['category']}): gave {r['given_label']!r}, "
              f"expected {r['expected_label']!r}")
    if not gold_block["all_passed"]:
        msg = (f"{len(gold_block['failed_items'])} of {gold_block['n_gold']} gold items were "
              f"mislabelled: {[r['item_key'] for r in gold_block['failed_items']]}. Per the "
              f"gold-item design, this is grounds to distrust the labelling pass, not just "
              f"the failed items themselves.")
        if not args.allow_gold_failures:
            raise RuntimeError(
                msg + " Refusing to compute kappa. Re-check the labelling pass, or pass "
                "--allow_gold_failures to see the numbers anyway (they will be stamped "
                "NOT_TRUSTWORTHY).")
        print(f"[gold] {msg} Proceeding ONLY because --allow_gold_failures was given; every "
              f"number below is NOT_TRUSTWORTHY.")
    gold_trustworthy = gold_block["all_passed"]

    # ---- stage 2: judges. Nothing above this line touches the GPU. ----------------------
    cfg_path = resolve(args.config)
    cfg = load_yaml(cfg_path)
    qb = cfg["backends"]["qwen_prompted"]
    batch_size = args.batch_size or int(qb["batch_size"])
    audit_max_new = int(qb["audit_generation"]["max_new_tokens"])

    pin = verify_judge_pin(cfg, cfg_path)
    print(f"\n[pin] {pin['status']}")
    if pin["status"] != "VERIFIED":
        raise RuntimeError(f"Judge pin not VERIFIED ({pin}). Refusing to score.")

    judges, jmeta, _loaded = build_judges(cfg)
    asr_judges = {n: m for n, m in jmeta.items() if m.get("role") == "asr_judge"}
    if not asr_judges:
        raise RuntimeError("No asr_judge found in the pinned config. This worksheet only "
                           "covers prefilling/persona/many_shot, which require one.")
    indep = judge_independence_block(cfg, jmeta)
    if indep["status"] != "OK":
        raise RuntimeError(f"Judge independence check FAILED: {indep}. Refusing to score.")

    cal = run_calibration(cfg, judges, jmeta, vs_path, batch_size, args.audit_text,
                          audit_max_new)
    if cal["label_provenance"]["reference_labeller_is_human"] is not True:
        raise RuntimeError(
            f"run_calibration reports reference_labeller_is_human="
            f"{cal['label_provenance']['reference_labeller_is_human']!r}, expected True. "
            f"This should be impossible given the stage-1 check above; refusing to trust "
            f"either check over the other and stopping instead.")

    # ---- stage 3: split gold out of the scored rows, recompute kappa on the rest --------
    gold_id_set = set(gold_key["items"].keys())
    for r in cal["rows"]:
        r["item_key"] = r["id"].split("human_", 1)[1] if r["id"].startswith("human_") else None
        r["is_gold"] = r["item_key"] in gold_id_set
    non_gold_rows = [r for r in cal["rows"] if not r["is_gold"]]
    gold_rows = [r for r in cal["rows"] if r["is_gold"]]
    print(f"\n[split] {len(non_gold_rows)} non-gold items (headline kappa) / "
          f"{len(gold_rows)} gold items (reference-only, excluded from the headline number)")

    judge_names = sorted(asr_judges)
    positive_by_judge = {jn: jmeta[jn]["labels"][1] if False else asr_judges[jn].get(
        "positive_label", "unsafe") for jn in judge_names}
    # positive_label is recorded per judge in cal["per_judge"][jn] already (from
    # agreement_block on the full set); reuse it directly rather than re-deriving.
    positive_by_judge = {jn: cal["per_judge"][jn]["positive_label"] for jn in judge_names
                         if jn in cal["per_judge"]}

    primary = {
        "per_judge_pooled": pooled_per_judge(non_gold_rows, judge_names, positive_by_judge,
                                             args.seed),
        "per_judge_per_category": per_judge_per_category(non_gold_rows, judge_names, args.seed),
        "combined_per_category": pooled_per_category(non_gold_rows, args.seed),
        "n_items": len(non_gold_rows),
        "note": "THE HEADLINE NUMBER FOR METHODS. Computed on the 54 non-gold items only.",
    }
    reference_incl_gold = {
        "per_judge": cal["per_judge"], "combined_per_category": cal["combined_per_category"],
        "n_items": cal["n_items"],
        "note": ("NOT the headline number. Includes the 6 gold items, which were "
                "deliberately selected to be unambiguous and will inflate agreement "
                "relative to a representative random sample. Reported for transparency "
                "and as a sanity check only."),
    }

    trust_stamp = "TRUSTWORTHY" if gold_trustworthy else "NOT_TRUSTWORTHY (gold check failed; --allow_gold_failures was used)"
    report = {
        "record_type": "human_asr_kappa_report",
        "script": "scripts/compute_kappa.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "timestamp": t0.isoformat(timespec="seconds"),
        "seed": args.seed,
        "trust_stamp": trust_stamp,
        "config_file": str(cfg_path), "config_sha256": sha256_file(cfg_path),
        "judge_pin_verification": pin,
        "judge_independence": indep,
        "judge_metadata": jmeta,
        "validation_set": str(vs_path), "validation_set_sha256": sha256_file(vs_path),
        "gold_answer_key": str(gold_path), "gold_answer_key_sha256": sha256_file(gold_path),
        "label_provenance": cal["label_provenance"],
        "gold_check": gold_block,
        "primary_kappa_excluding_gold": primary,
        "reference_kappa_including_gold": reference_incl_gold,
        "rows": cal["rows"],
    }

    stamp = t0.strftime("%Y%m%d_%H%M%S")
    out = resolve(args.out) if args.out else HV / f"human_asr_kappa_report_{stamp}.json"
    if out.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{out} exists; refusing to overwrite a scored report. Pass "
                              f"--allow_overwrite if this is deliberate.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n===== HUMAN ASR KAPPA -- {trust_stamp} =====")
    for jn, blk in primary["per_judge_pooled"].items():
        ci = blk["cohens_kappa_ci95"]
        k = "DEGENERATE" if blk["cohens_kappa_degenerate"] else f"{blk['cohens_kappa']:.3f}"
        civ = f"[{ci['lo']}, {ci['hi']}]" if ci.get("lo") is not None else "[n/a]"
        c = blk["confusion"]
        print(f"  {jn:<24} n={blk['n']:<4} agree={blk['raw_agreement']:.3f}  kappa={k:<10} "
              f"95% CI {civ:<18} tp{c['tp']} fp{c['fp']} fn{c['fn']} tn{c['tn']}")
    print("  -- per category --")
    for cat, blk in primary["combined_per_category"].items():
        ci = blk["cohens_kappa_ci95"]
        k = "DEGENERATE" if blk["cohens_kappa_degenerate"] else f"{blk['cohens_kappa']:.3f}"
        civ = f"[{ci['lo']}, {ci['hi']}]" if ci.get("lo") is not None else "[n/a]"
        print(f"  {cat:<16} n={blk['n']:<4} agree={blk['raw_agreement']:.3f}  kappa={k:<10} "
              f"95% CI {civ}")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
