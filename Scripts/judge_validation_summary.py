"""
Project 71 -- roll the judge-validation reports up into the numbers that go in Methods.

Feeds: the Cohen's kappa figures reported in Methods, which are what make every ASR and
over-refusal number in Tables 1-2 credible rather than assumed. Produces NO arm metric.

Two things it does that cannot be read off the individual reports:

1. UNION DECOMPOSITION. For each validation set it reports, from the same items, what each
   ASR judge scored ALONE and what the union scored, plus how many true attacks the second
   judge RECOVERED from the first and how many false positives it ADDED. "Does the union
   recover the misses or just add false positives" is the question the union exists to
   answer, and it has to be answered with counts, not with a kappa alone.

2. BASE-RATE CORRECTION for the refusal-enriched set. That set was drawn stratified on the
   judge's own verdict, so its raw agreement is NOT an estimate of population agreement --
   the judge-positive stratum is massively over-sampled. Each stratum's observed rates are
   reweighted by its population share (recorded at sampling time in *_strata.json) to give
   an estimated population confusion matrix and kappa. Reporting the enriched kappa as if it
   were a population kappa would overstate the judge.

Usage:
    python Scripts\\judge_validation_summary.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HV = REPO / "results" / "human_validation"


def kappa_from_counts(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    pe = ((tp + fp) / n) * ((tp + fn) / n) + ((fn + tn) / n) * ((fp + tn) / n)
    return None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def union_decomposition(report_path: Path, members=("safety", "behavioural")):
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [r for r in rep["rows"] if all(m in r["per_judge"] for m in members)]
    if not rows:
        return None
    a, b = members
    truth = {r["id"]: r["human_label"] == "unsafe" for r in rows}
    va = {r["id"]: r["per_judge"][a]["verdict"] == "unsafe" for r in rows}
    vb = {r["id"]: r["per_judge"][b]["verdict"] == "unsafe" for r in rows}
    ids = list(truth)
    n_true = sum(truth.values())
    out = {
        "set": report_path.name, "n_items": len(ids), "n_true_attacks": n_true,
        "judges": list(members),
    }
    for name, v in ((a, va), (b, vb)):
        tp = sum(1 for i in ids if truth[i] and v[i]); fp = sum(1 for i in ids if not truth[i] and v[i])
        fn = sum(1 for i in ids if truth[i] and not v[i]); tn = sum(1 for i in ids if not truth[i] and not v[i])
        out[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "kappa": round(kappa_from_counts(tp, fp, fn, tn), 4),
                     "recall": round(tp / n_true, 4) if n_true else None,
                     "precision": round(tp / (tp + fp), 4) if (tp + fp) else None}
    tp = sum(1 for i in ids if truth[i] and (va[i] or vb[i]))
    fp = sum(1 for i in ids if not truth[i] and (va[i] or vb[i]))
    fn = sum(1 for i in ids if truth[i] and not (va[i] or vb[i]))
    tn = sum(1 for i in ids if not truth[i] and not (va[i] or vb[i]))
    out["union"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "kappa": round(kappa_from_counts(tp, fp, fn, tn), 4),
                    "recall": round(tp / n_true, 4) if n_true else None,
                    "precision": round(tp / (tp + fp), 4) if (tp + fp) else None}
    # what the SECOND judge contributes on top of the first, and vice versa
    out["contribution"] = {
        f"{b}_recovers_true_attacks_{a}_missed": sum(1 for i in ids if truth[i] and not va[i] and vb[i]),
        f"{b}_adds_false_positives_{a}_avoided": sum(1 for i in ids if not truth[i] and not va[i] and vb[i]),
        f"{a}_recovers_true_attacks_{b}_missed": sum(1 for i in ids if truth[i] and not vb[i] and va[i]),
        f"{a}_adds_false_positives_{b}_avoided": sum(1 for i in ids if not truth[i] and not vb[i] and va[i]),
        "missed_by_both": sum(1 for i in ids if truth[i] and not va[i] and not vb[i]),
        "false_positive_from_both": sum(1 for i in ids if not truth[i] and va[i] and vb[i]),
    }
    return out


def refusal_population_estimate(report_path: Path, set_path: Path, strata_paths):
    """Reweight the enriched sample back to the natural base rate."""
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    items = {json.loads(l)["id"]: json.loads(l)
             for l in set_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    pop = {}
    for p in strata_paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"strata file missing: {p}")
        for k, v in json.loads(Path(p).read_text(encoding="utf-8"))["strata"].items():
            pop[k] = v
    n_pop = sum(v["n_population"] for v in pop.values())
    n_pop_pos = sum(v["n_judge_positive"] for v in pop.values())
    n_pop_neg = n_pop - n_pop_pos

    # Sampled counts per stratum. The stratum is the judge's verdict AT SAMPLING TIME, which
    # is the quantity the population share was measured on.
    obs = {"judge_positive": {"refusal": 0, "complied": 0},
           "judge_negative": {"refusal": 0, "complied": 0}}
    for r in rep["rows"]:
        st = items[r["id"]]["sampling_stratum"]
        obs[st][r["human_label"]] += 1

    npos, nneg = sum(obs["judge_positive"].values()), sum(obs["judge_negative"].values())
    # Within the judge-positive stratum the judge is positive by definition, so human
    # 'refusal' -> TP and human 'complied' -> FP; mirrored in the negative stratum.
    tp = n_pop_pos * obs["judge_positive"]["refusal"] / npos
    fp = n_pop_pos * obs["judge_positive"]["complied"] / npos
    fn = n_pop_neg * obs["judge_negative"]["refusal"] / nneg
    tn = n_pop_neg * obs["judge_negative"]["complied"] / nneg
    return {
        "note": ("estimated POPULATION confusion for the over-refusal judge, obtained by "
                 "reweighting each verdict-stratum of the enriched sample by its population "
                 "share. Fractional counts are expected values, not observations."),
        "population_n": n_pop, "population_judge_positive": n_pop_pos,
        "sampled_positive_stratum": npos, "sampled_negative_stratum": nneg,
        "observed_in_positive_stratum": obs["judge_positive"],
        "observed_in_negative_stratum": obs["judge_negative"],
        "estimated_confusion": {k: round(v, 2) for k, v in
                                (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn))},
        "estimated_precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "estimated_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "judge_over_refusal_rate": round((tp + fp) / n_pop, 4),
        "estimated_true_over_refusal_rate": round((tp + fn) / n_pop, 4),
        "estimated_population_kappa": round(kappa_from_counts(tp, fp, fn, tn), 4),
    }


def main():
    out = {"generated": "scripts/judge_validation_summary.py",
           "warning": ("NOT a paper number while configs/judges.yaml pin_status is not PINNED "
                       "and judge_independence.status is not OK."),
           "union_decomposition": [], "refusal_population_estimate": None}

    for name in ("judge_validation_report_v2_union.json",
                 "judge_validation_report_heldout2.json",
                 "judge_validation_report_heldout3.json"):
        p = HV / name
        if p.is_file():
            d = union_decomposition(p)
            if d:
                out["union_decomposition"].append(d)

    rp = HV / "judge_validation_report_refusal_enriched.json"
    if rp.is_file():
        out["refusal_population_estimate"] = refusal_population_estimate(
            rp, HV / "refusal_enriched_validation_set.jsonl",
            [HV / "_refusal_b1_strata.json", HV / "_refusal_b0_strata.json"])

    dest = HV / "judge_validation_summary.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("===== ASR JUDGE UNION DECOMPOSITION =====")
    for d in out["union_decomposition"]:
        print(f"\n{d['set']}   n={d['n_items']}  true attacks={d['n_true_attacks']}")
        for k in ("safety", "behavioural", "union"):
            v = d[k]
            print(f"  {k:<12} kappa={v['kappa']:>7}  recall={v['recall']:>6}  "
                  f"precision={str(v['precision']):>6}  tp{v['tp']} fp{v['fp']} fn{v['fn']} tn{v['tn']}")
        for k, v in d["contribution"].items():
            print(f"      {k}: {v}")
    if out["refusal_population_estimate"]:
        e = out["refusal_population_estimate"]
        print("\n===== OVER-REFUSAL JUDGE, BASE-RATE CORRECTED =====")
        print(f"  population n={e['population_n']}, judge flags {e['population_judge_positive']}")
        print(f"  estimated confusion (expected counts): {e['estimated_confusion']}")
        print(f"  estimated precision {e['estimated_precision']}, recall {e['estimated_recall']}")
        print(f"  judge over-refusal rate {e['judge_over_refusal_rate']} vs estimated true "
              f"{e['estimated_true_over_refusal_rate']}")
        print(f"  estimated POPULATION kappa {e['estimated_population_kappa']}")
    print(f"\n[written] {dest}")


if __name__ == "__main__":
    main()
