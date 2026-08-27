"""
Project 71 -- tables_final: assemble final Table 1 / Table 2 inputs + TOST equivalence CI.

Feeds: final Table 1, final Table 2, the headline sentence, the null-as-a-bound paragraph
(TOST/equivalence reading), and the sign-convention statement (results/tables_final/).

READ-ONLY over all scored files, summaries and stats reports. It never modifies them.

Policy (task rules, 2026-08-28):
  1. Prefer numbers already in results/stats_report_realsuite_handlabelled.json.
  2. Before computing anything new, REPRODUCE the report's published numbers from the
     hand-labelled scored files with the identical procedure (same code imported from
     scripts/stats.py, same seed, same n_boot). Any mismatch -> hard exit, no output.
  3. New computations only:
       a. per-arm primary-ASR 95% bootstrap CI over the 180 prompts (all five arms);
       b. per-arm crisis failure-rate 95% bootstrap CI over the 60 prompts (b3, t);
       c. per-category paired T-B3 difference with 95% bootstrap CI (new for the three
          non-crisis categories; crisis is verified against the report);
       d. TOST/equivalence: 90% paired bootstrap CI (two one-sided tests at alpha=0.05)
          on the same per-prompt differences the primary test uses, n_boot=20000, seed 0;
          plus the smallest delta at which TOST would conclude |effect| < delta
          (EXPLORATORY, NOT PRE-REGISTERED).
  All bootstraps use stats.bootstrap_paired (np.random.default_rng(seed), joint prompt
  resampling) and stats.pct_ci percentile intervals. Deterministic given --seed.

Usage:
    python scripts\\make_tables_final.py --seed 0
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from stats import (load_scored, outcome_vectors, bootstrap_paired, pct_ci,
                   paired_permutation, seed_mean, PRIMARY_ATTACK_CATEGORIES,
                   CRISIS_CATEGORY, OVER_REFUSAL_CATEGORY)  # noqa: E402

REPORT = REPO / "results" / "stats_report_realsuite_handlabelled.json"
SCORED = {
    # canonical per-arm files per each dir's README.md (2026-08-27)
    "b0": REPO / "results" / "b0_seed42" / "scored_realsuite.jsonl",
    "b1": REPO / "results" / "b1_seed42" / "scored_realsuite.jsonl",
    "b2": REPO / "results" / "b2_seed42" / "scored.jsonl",
    "b3": REPO / "results" / "b3_seed42" / "scored.jsonl",
    "t":  REPO / "results" / "t_seed42" / "scored.jsonl",
}

FAILURES = []


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}: got {got!r} want {want!r}")
    if not ok:
        FAILURES.append((name, got, want))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0,
                    help="bootstrap seed (0 = the seed used by the pinned stats report)")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--n_boot_tost", type=int, default=20000)
    ap.add_argument("--out", default=str(REPO / "results" / "tables_final" / "tables_final.json"))
    args = ap.parse_args()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    data = {}
    for arm, path in SCORED.items():
        header, rows = load_scored(path)
        assert header["arm"].lower() == arm, f"{path}: header arm {header['arm']!r} != {arm!r}"
        data[arm] = rows
        print(f"loaded {arm}: {len(rows)} rows from {path.name}")

    # fixed prompt-id order, identical to stats.py (sorted), verified identical across arms
    ref = data["t"]
    atk_ids = sorted(i for i, r in ref.items() if r["category"] in PRIMARY_ATTACK_CATEGORIES)
    crisis_ids = sorted(i for i, r in ref.items() if r["category"] == CRISIS_CATEGORY)
    ben_ids = sorted(i for i, r in ref.items() if r["category"] == OVER_REFUSAL_CATEGORY)
    cat_ids = {c: sorted(i for i, r in ref.items() if r["category"] == c)
               for c in PRIMARY_ATTACK_CATEGORIES + [CRISIS_CATEGORY]}
    assert len(atk_ids) == 180 and len(crisis_ids) == 60 and len(ben_ids) == 60

    vec = {arm: {"atk": outcome_vectors(rows, atk_ids, "unsafe").astype(float),
                 "crisis": outcome_vectors(rows, crisis_ids, "unsafe").astype(float)}
           for arm, rows in data.items()}

    # ---------------- verification pass: reproduce the pinned report exactly ------------
    print("\nVERIFICATION against", REPORT.name)
    rp = report["primary_test"]
    b3a, ta = vec["b3"]["atk"], vec["t"]["atk"]
    diff = ta - b3a
    check("b3 primary ASR", float(b3a.mean()), rp["primary"]["baseline_asr"])
    check("t primary ASR", float(ta.mean()), rp["primary"]["treatment_asr"])
    check("mean difference", float(diff.mean()), rp["primary"]["mean_difference"])
    check("n nonzero diffs", int((diff != 0).sum()), rp["primary"]["n_nonzero_differences"])
    perm = paired_permutation(diff, n_perm=rp["primary"]["n_permutations"],
                              seed=rp["primary"]["permutation_seed"])
    check("permutation p", perm["p_value"], rp["primary"]["p_value"])
    boot = bootstrap_paired({"t": ta, "b": b3a}, report["n_boot"], report["bootstrap_seed"],
                            lambda d: d["t"].mean() - d["b"].mean())
    ci95 = pct_ci(boot)
    want95 = rp["asr_difference_treatment_minus_baseline"]["ci95_bootstrap_over_prompts"]
    check("95% CI lo (diff)", ci95[0], want95[0])
    check("95% CI hi (diff)", ci95[1], want95[1])
    b3c, tc = vec["b3"]["crisis"], vec["t"]["crisis"]
    cp = rp["crisis_co_primary"]
    check("b3 crisis rate", float(b3c.mean()), cp["baseline_failure_rate"])
    check("t crisis rate", float(tc.mean()), cp["treatment_failure_rate"])
    cboot = bootstrap_paired({"t": tc, "b": b3c}, report["n_boot"], report["bootstrap_seed"],
                             lambda d: d["t"].mean() - d["b"].mean())
    cci = pct_ci(cboot)
    check("crisis 95% CI lo", cci[0], cp["ci95_bootstrap_over_prompts"][0])
    check("crisis 95% CI hi", cci[1], cp["ci95_bootstrap_over_prompts"][1])
    # per-category rates vs report table2
    for arm in ("b3", "t"):
        for c in PRIMARY_ATTACK_CATEGORIES + [CRISIS_CATEGORY]:
            got = float(outcome_vectors(data[arm], cat_ids[c], "unsafe").mean())
            check(f"{arm} {c} rate", got, report["table2"][arm][c]["mean"])
    # b0/b1/b2 primary ASR vs their own summaries
    for arm, want in (("b0", 0.2611111111111111), ("b1", 0.5111111111111111),
                      ("b2", 0.46111111111111114)):
        check(f"{arm} primary ASR (summary)", float(vec[arm]["atk"].mean()), want)
    # hand-labelled over-refusal (b2/b3/t): 0/60 refused in the scored files
    for arm in ("b2", "b3", "t"):
        n_ref = int(outcome_vectors(data[arm], ben_ids, "refused").sum())
        check(f"{arm} over-refusal n_refused (hand labels)", n_ref, 0)

    if FAILURES:
        print(f"\nSTOP: {len(FAILURES)} recomputed number(s) disagree with the pinned "
              f"report/summaries. NO OUTPUT WRITTEN. Discrepancies:")
        for name, got, want in FAILURES:
            print(f"  {name}: recomputed {got!r} vs recorded {want!r}")
        sys.exit(1)
    print("verification: ALL EXACT -- proceeding to new computations\n")

    # ---------------- new computations --------------------------------------------------
    per_arm_asr_ci = {}
    for arm in SCORED:
        b = bootstrap_paired({"x": vec[arm]["atk"]}, args.n_boot, args.seed,
                             lambda d: d["x"].mean())
        per_arm_asr_ci[arm] = {"asr": float(vec[arm]["atk"].mean()),
                               "ci95_bootstrap_over_prompts": list(pct_ci(b)),
                               "n_prompts": 180, "n_boot": args.n_boot, "seed": args.seed}
    per_arm_crisis_ci = {}
    for arm in ("b3", "t"):
        b = bootstrap_paired({"x": vec[arm]["crisis"]}, args.n_boot, args.seed,
                             lambda d: d["x"].mean())
        per_arm_crisis_ci[arm] = {"rate": float(vec[arm]["crisis"].mean()),
                                  "ci95_bootstrap_over_prompts": list(pct_ci(b)),
                                  "n_prompts": 60, "n_boot": args.n_boot, "seed": args.seed}
    per_cat_diff = {}
    for c in PRIMARY_ATTACK_CATEGORIES + [CRISIS_CATEGORY]:
        tv = outcome_vectors(data["t"], cat_ids[c], "unsafe").astype(float)
        bv = outcome_vectors(data["b3"], cat_ids[c], "unsafe").astype(float)
        b = bootstrap_paired({"t": tv, "b": bv}, args.n_boot, args.seed,
                             lambda d: d["t"].mean() - d["b"].mean())
        per_cat_diff[c] = {"b3": float(bv.mean()), "t": float(tv.mean()),
                           "diff_t_minus_b3": float(tv.mean() - bv.mean()),
                           "ci95_bootstrap_over_prompts": list(pct_ci(b)),
                           "n_prompts": len(cat_ids[c]), "n_boot": args.n_boot,
                           "seed": args.seed}

    # TOST / equivalence: 90% CI = two one-sided tests at alpha=0.05
    tost_boot = bootstrap_paired({"t": ta, "b": b3a}, args.n_boot_tost, args.seed,
                                 lambda d: d["t"].mean() - d["b"].mean())
    ci90 = pct_ci(tost_boot, alpha=0.10)
    margin_pts = 10.0
    lo_pts, hi_pts = ci90[0] * 100, ci90[1] * 100
    equiv10 = (lo_pts > -margin_pts) and (hi_pts < margin_pts)
    smallest_delta_pts = max(abs(lo_pts), abs(hi_pts))
    tost = {
        "method": ("TOST via 90% percentile bootstrap CI on the paired per-prompt "
                   "T-B3 differences (two one-sided tests at alpha=0.05); same 180 paired "
                   "prompts and same joint-resampling bootstrap as the primary analysis"),
        "n_pairs": 180, "n_boot": args.n_boot_tost, "seed": args.seed,
        "point_estimate_pts": float(diff.mean() * 100),
        "ci90_bootstrap_over_prompts_pts": [lo_pts, hi_pts],
        "equivalence_margin_pts": margin_pts,
        "equivalence_within_pm10_concluded": bool(equiv10),
        "verdict": ("EQUIVALENCE NOT CONCLUDED: the 90% CI is not contained in "
                    "(-10, +10)" if not equiv10 else
                    "equivalence within +/-10 points concluded at alpha=0.05"),
        "smallest_delta_pts_exploratory": float(smallest_delta_pts),
        "smallest_delta_note": ("EXPLORATORY, NOT PRE-REGISTERED: the smallest margin "
                                "delta at which TOST would conclude |effect| < delta is "
                                "max(|CI90 lo|, |CI90 hi|)."),
        "true_scale_caveat": ("Amendment 22.3: the pre-registered 10-point threshold is on "
                              "the TRUE scale (Revision 6); this raw-observed TOST is "
                              "reported against the same 10-point number for transparency "
                              "but the two scales are NOT directly comparable. See the "
                              "recall-sensitivity sweep in notebook/power_analysis.md."),
    }

    # ---------------- assembly (copied from pinned reports/summaries, no computation) ----
    SUMMARY = {
        "b0": REPO / "results" / "b0_seed42" / "summary_realsuite.json",
        "b1": REPO / "results" / "b1_seed42" / "summary_realsuite.json",
        "b2": REPO / "results" / "b2_seed42" / "summary.json",
        "b3": REPO / "results" / "b3_seed42" / "summary.json",
        "t":  REPO / "results" / "t_seed42" / "summary.json",
    }
    table1 = {}
    for arm, sp in SUMMARY.items():
        sm = json.loads(sp.read_text(encoding="utf-8"))["metrics"]
        table1[arm] = {
            "single_run_context": arm in ("b0", "b1"),
            "asr_primary_pct": sm["asr_primary_non_crisis"]["value"] * 100,
            "asr_ci95_over_prompts_pct": [x * 100 for x in
                                          per_arm_asr_ci[arm]["ci95_bootstrap_over_prompts"]],
            "over_refusal_pct": sm["over_refusal"]["value"] * 100,
            "over_refusal_source": sm["over_refusal"].get("source"),
            "helpfulness_mean_reward": sm["helpfulness"]["value"],
        }
        check(f"table1 {arm} over_refusal source",
              sm["over_refusal"].get("source"),
              "hand_label" if arm in ("b2", "b3", "t") else "rubric_judge")
    table2 = {c: per_cat_diff[c] for c in per_cat_diff}
    table2_decomposition = report["table2_decomposition"]

    # post-hoc panel (Amendment 23), assembled from the three stats reports and verified
    # against the values recorded in the 2026-08-27 lab-notebook entry
    NOTEBOOK_PANEL = {  # seed: (b3_asr, t_asr, d_primary, p, crisis_b3, crisis_t, d_crisis, p)
        1: (45.00, 39.44, -5.56, 0.227, 36.67, 30.00, -6.67, 0.484),
        2: (40.00, 36.67, -3.33, 0.460, 25.00, 40.00, 15.00, 0.048),
        3: (40.00, 39.44, -0.56, 1.000, 35.00, 35.00, 0.00, 1.000),
    }
    panel = {}
    for s, rep_path in ((1, REPORT),
                        (2, REPO / "results" / "stats_report_ts2_posthoc.json"),
                        (3, REPO / "results" / "stats_report_ts3_posthoc.json")):
        r = json.loads(rep_path.read_text(encoding="utf-8"))["primary_test"]
        row = {
            "b3_asr_pct": r["primary"]["baseline_asr"] * 100,
            "t_asr_pct": r["primary"]["treatment_asr"] * 100,
            "d_primary_pts": r["primary"]["mean_difference"] * 100,
            "p_primary": r["primary"]["p_value"],
            "crisis_b3_pct": r["crisis_co_primary"]["baseline_failure_rate"] * 100,
            "crisis_t_pct": r["crisis_co_primary"]["treatment_failure_rate"] * 100,
            "d_crisis_pts": r["crisis_co_primary"]["difference_treatment_minus_baseline"] * 100,
            "p_crisis": r["crisis_co_primary"]["test"]["p_value"],
            "crisis_ci95_pts": [x * 100 for x in
                                r["crisis_co_primary"]["ci95_bootstrap_over_prompts"]],
            "status": "pre-registered primary" if s == 1 else "POST-HOC (Amendment 23)",
            "source": str(rep_path),
        }
        nb = NOTEBOOK_PANEL[s]
        got = (round(row["b3_asr_pct"], 2), round(row["t_asr_pct"], 2),
               round(row["d_primary_pts"], 2), round(row["p_primary"], 3),
               round(row["crisis_b3_pct"], 2), round(row["crisis_t_pct"], 2),
               round(row["d_crisis_pts"], 2), round(row["p_crisis"], 3))
        check(f"panel seed {s} vs 2026-08-27 notebook entry", got, nb)
        panel[f"seed{s}"] = row
    if FAILURES:
        print("\nSTOP: assembly cross-checks failed; NO OUTPUT WRITTEN.")
        for name, got, want in FAILURES:
            print(f"  {name}: {got!r} vs {want!r}")
        sys.exit(1)

    out = {
        "record_type": "tables_final",
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "script": "scripts/make_tables_final.py",
        "seed": args.seed,
        "sources": {
            "primary_report": str(REPORT),
            "scored_files": {a: str(p) for a, p in SCORED.items()},
            "posthoc_reports": [str(REPO / "results" / f"stats_report_ts{s}_posthoc.json")
                                for s in (2, 3)],
        },
        "verification": "ALL EXACT (see stdout log): recomputed primary/crisis/per-category"
                        " numbers reproduce the pinned hand-labelled report bit-for-bit;"
                        " post-hoc panel matches the 2026-08-27 lab-notebook entry",
        "table1": table1,
        "table1_notes": {
            "asr_endpoint": report["table1"]["b3"]["asr_endpoint"],
            "crisis_not_pooled": "crisis_adjacent (n=60) is a co-primary reported in "
                                 "Table 2, never pooled into Table 1 (Amendment 13).",
            "ci_scope": "CIs are 95% percentile bootstrap over the 180-prompt set "
                        "(n_boot=10000, seed 0), NOT across training seeds; single "
                        "training seed per arm (Amendment 22.1).",
            "over_refusal_convention": (
                "b2/b3/t: HAND LABELS (Revision 4 primary instrument). b2 uses b3's "
                "label file -- benign responses byte-identical, filter fired 0/60 benign "
                "(provenance dagger). b0/b1: rubric-judge CROSS-CHECK ONLY (kappa~0.075, "
                "precision 0.20, failed validation twice) -- flagged, never comparable "
                "to the hand-labelled rows. Convention chosen: show the flagged judge "
                "number rather than a blank, because b0/b1 are single-run context rows "
                "and an explicit flagged value with its kappa is more honest than an "
                "empty cell that invites silent assumption of comparability."),
        },
        "table2": table2,
        "table2_decomposition": table2_decomposition,
        "headline_sentence": report["headline_sentence"],
        "headline_blockers": report.get("headline_blockers"),
        "posthoc_panel_amendment23": panel,
        "posthoc_panel_notes": (
            "Seed 1 is the pre-registered primary analysis; seeds 2-3 are POST-HOC "
            "(Amendment 23) and carry no hand over-refusal labels. Helpfulness cost "
            "stable across seeds (T -1.6 to -2.1 vs B2/B3). The crisis co-primary is "
            "NOT robust to training seed: it flips sign across seeds, and seed 2 is "
            "nominally significant AGAINST T (+15.00 pts, CI [+1.67, +28.33], p=0.048; "
            "post-hoc, n=60). No seed approaches the pre-registered 10-point TRUE-scale "
            "threshold."),
        "per_arm_primary_asr_ci_NEW": per_arm_asr_ci,
        "per_arm_crisis_ci_NEW": per_arm_crisis_ci,
        "per_category_paired_diff_NEW": per_cat_diff,
        "tost_equivalence_NEW": tost,
        "sign_convention": (
            "All differences in this project are T minus B3. A NEGATIVE ASR difference "
            "means T (trained-in safety) has the LOWER attack-success rate. At -5.56 "
            "points, T is the arm that is ahead and B3 is the baseline being beaten. "
            "This convention was previously implicit; it is recorded explicitly as of "
            "2026-08-28."),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # tables_final.md is assembled from this JSON + the pinned reports; JSON is the
    # machine-readable record of every newly computed number.
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps({"per_arm_primary_asr_ci_NEW": per_arm_asr_ci,
                      "per_arm_crisis_ci_NEW": per_arm_crisis_ci,
                      "per_category_paired_diff_NEW": per_cat_diff,
                      "tost_equivalence_NEW": tost}, indent=2))


if __name__ == "__main__":
    main()
