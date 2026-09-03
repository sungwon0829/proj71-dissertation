"""Analysis 4 (Amendment 24.1, POST-HOC EXPLORATORY) - classifier-blindness mechanism.

For judge-BREACH responses per arm (primary scope: prefilling, persona, many_shot),
cross-tabs classifier miss (LG-missed / BD-missed / missed-by-both) against
(a) response-length tercile, (b) degeneracy flag, (c) attack category; plus the
pre-specified descriptive degenerate-vs-non-degenerate missed-by-both comparison and
Wilson 95% CIs on the per-arm missed-by-both fractions.

Deterministic; nothing samples. --seed is provenance only.
CPU only: reads JSONL/CSV artifacts, no model loading.

Outputs (this directory only):
  breach_items.csv                 - per-item joined table (audit trail)
  crosstab_mechanism.csv           - all pre-specified cross-tab cells, long format
  wilson_ci_missed_by_both.csv     - per-arm Wilson 95% CIs
  summary.json                     - machine-readable summary + provenance
Run:  python analyze_mechanism.py --seed 0
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict

ROOT = r"C:\proj71"
OUTDIR = os.path.join(ROOT, "results", "exploratory", "analysis4_classifier_blindness_mechanism")
PROBE_DIR = os.path.join(ROOT, "results", "judge_probes", "classifier_blindness_fullsuite_20260827")

ARMS = ["b0", "b1", "b2", "b3", "t"]
PRIMARY_CATS = ["prefilling", "persona", "many_shot"]

SCORED = {
    "b0": os.path.join(ROOT, "results", "b0_seed42", "scored_realsuite.jsonl"),
    "b1": os.path.join(ROOT, "results", "b1_seed42", "scored_realsuite.jsonl"),
    "b2": os.path.join(ROOT, "results", "b2_seed42", "scored.jsonl"),
    "b3": os.path.join(ROOT, "results", "b3_seed42", "scored.jsonl"),
    "t": os.path.join(ROOT, "results", "t_seed42", "scored.jsonl"),
}
GENERATIONS = {
    "b0": os.path.join(ROOT, "results", "b0_seed42", "generations_realsuite_SUPERSEDED_pre_b1v2_regen.jsonl"),
    "b1": os.path.join(ROOT, "results", "b1_seed42", "generations_realsuite.jsonl"),
    "b2": os.path.join(ROOT, "results", "b2_seed42", "generations.jsonl"),
    "b3": os.path.join(ROOT, "results", "b3_seed42", "generations.jsonl"),
    "t": os.path.join(ROOT, "results", "t_seed42", "generations.jsonl"),
}

# Consistency anchors from the parent exhibit's crosstabs.csv (overall_primary BREACH rows).
EXPECTED = {
    "b0": {"n_breach": 47, "lg": 3, "bd": 3, "neither": 42},
    "b1": {"n_breach": 92, "lg": 13, "bd": 4, "neither": 77},
    "b2": {"n_breach": 83, "lg": 2, "bd": 2, "neither": 79},
    "b3": {"n_breach": 81, "lg": 0, "bd": 2, "neither": 79},
    "t": {"n_breach": 71, "lg": 3, "bd": 0, "neither": 68},
}

SMALL_CELL = 10  # denominator < 10 -> counts only, no percentage


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def frac_str(k, n):
    """Small-cell rule: denominator < 10 -> counts only."""
    if n < SMALL_CELL:
        return f"{k}/{n}"
    return f"{k}/{n} ({100.0 * k / n:.1f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="provenance only; nothing samples")
    args = ap.parse_args()

    provenance = {"script": os.path.abspath(__file__), "seed": args.seed,
                  "seed_note": "provenance only; analysis is deterministic",
                  "inputs": {}}

    # ---- load per-arm data -------------------------------------------------
    items = {}  # arm -> id -> dict
    for arm in ARMS:
        sp = SCORED[arm]
        provenance["inputs"][f"scored_{arm}"] = {"path": sp, "sha256": sha256_file(sp)}
        per = OrderedDict()
        for r in read_jsonl(sp):
            if r.get("record_type") != "scored":
                continue
            if r["category"] not in PRIMARY_CATS:
                continue
            per[r["id"]] = {
                "id": r["id"], "category": r["category"],
                "breach": r["verdict"] == "unsafe",
                "degenerate_loose": bool(r["degenerate_loose"]),
                "degenerate_strict": bool(r["degenerate_strict"]),
            }
        assert len(per) == 180, (arm, len(per))
        items[arm] = per

    # response lengths: whitespace-token count of response_full_turn (delivered turn).
    # No generated-token counts exist in generation metadata (only prompt_n_tokens).
    for arm in ARMS:
        gp = GENERATIONS[arm]
        provenance["inputs"][f"generations_{arm}"] = {"path": gp, "sha256": sha256_file(gp)}
        seen = 0
        for r in read_jsonl(gp):
            if r.get("record_type") != "generation":
                continue
            if r["id"] in items[arm]:
                items[arm][r["id"]]["len_ws_tokens"] = len(r["response_full_turn"].split())
                seen += 1
        assert seen == 180, (arm, seen)

    # classifier flags from the parent probe's raw scores
    for arm in ARMS:
        for clf, key in (("llamaguard", "lg_flagged"), ("beaverdam", "bd_flagged")):
            rp = os.path.join(PROBE_DIR, f"raw_scores_{arm}_{clf}.jsonl")
            provenance["inputs"][f"raw_scores_{arm}_{clf}"] = {"path": rp, "sha256": sha256_file(rp)}
            for r in read_jsonl(rp):
                if r.get("record_type") != "raw_score":
                    continue
                if r["id"] in items[arm]:
                    items[arm][r["id"]][key] = bool(r["flagged"])

    # ---- breach sets, consistency anchors, terciles ------------------------
    breach = {}
    for arm in ARMS:
        rows = [v for v in items[arm].values() if v["breach"]]
        for v in rows:
            assert "lg_flagged" in v and "bd_flagged" in v and "len_ws_tokens" in v, (arm, v["id"])
            v["lg_missed"] = not v["lg_flagged"]
            v["bd_missed"] = not v["bd_flagged"]
            v["missed_by_both"] = v["lg_missed"] and v["bd_missed"]
        exp = EXPECTED[arm]
        assert len(rows) == exp["n_breach"], (arm, len(rows))
        assert sum(v["lg_flagged"] for v in rows) == exp["lg"], arm
        assert sum(v["bd_flagged"] for v in rows) == exp["bd"], arm
        assert sum(v["missed_by_both"] for v in rows) == exp["neither"], arm
        # rank-based terciles within arm over the BREACH set; ties broken by id;
        # sizes as equal as possible (remainder to the earlier terciles)
        rows_sorted = sorted(rows, key=lambda v: (v["len_ws_tokens"], v["id"]))
        n = len(rows_sorted)
        base, rem = divmod(n, 3)
        sizes = [base + (1 if i < rem else 0) for i in range(3)]
        idx = 0
        for t, sz in enumerate(sizes, start=1):
            for v in rows_sorted[idx:idx + sz]:
                v["tercile"] = f"T{t}"
            idx += sz
        breach[arm] = rows_sorted

    # ---- cross-tabs --------------------------------------------------------
    # b3 exclusion choice: b3 is EXCLUDED from LG-missed and missed-by-both
    # mechanism cells (near-tautology: the B3 filter IS Llama Guard, so LG-miss is
    # ~guaranteed by construction). b3 BD-missed cells are retained. b3's
    # missed-by-both Wilson CI is reported but flagged near-tautological, because
    # the pre-registered completion covers the full five-arm 84-97% range.
    def cells(rows):
        n = len(rows)
        return {
            "n": n,
            "lg_missed": sum(v["lg_missed"] for v in rows),
            "bd_missed": sum(v["bd_missed"] for v in rows),
            "missed_by_both": sum(v["missed_by_both"] for v in rows),
            "n_degenerate_loose": sum(v["degenerate_loose"] for v in rows),
        }

    crosstab_rows = []
    for arm in ARMS:
        rows = breach[arm]
        cuts = []
        for t in ("T1", "T2", "T3"):
            sub = [v for v in rows if v["tercile"] == t]
            lo = min(v["len_ws_tokens"] for v in sub)
            hi = max(v["len_ws_tokens"] for v in sub)
            cuts.append(("length_tercile", f"{t} [{lo}-{hi} ws-tokens]", sub))
        for flag, name in ((True, "degenerate_loose"), (False, "non_degenerate")):
            cuts.append(("degeneracy_loose", name, [v for v in rows if v["degenerate_loose"] is flag]))
        for c in PRIMARY_CATS:
            cuts.append(("attack_category", c, [v for v in rows if v["category"] == c]))
        cuts.append(("overall", "overall_primary", rows))
        for cut, level, sub in cuts:
            c = cells(sub)
            crosstab_rows.append({
                "arm": arm, "cut": cut, "level": level, "n_breach": c["n"],
                "lg_missed": "" if arm == "b3" else c["lg_missed"],
                "lg_missed_frac": "" if (arm == "b3" or c["n"] < SMALL_CELL or c["n"] == 0)
                                  else f"{c['lg_missed'] / c['n']:.3f}",
                "bd_missed": c["bd_missed"],
                "bd_missed_frac": "" if (c["n"] < SMALL_CELL or c["n"] == 0)
                                  else f"{c['bd_missed'] / c['n']:.3f}",
                "missed_by_both": "" if arm == "b3" else c["missed_by_both"],
                "missed_by_both_frac": "" if (arm == "b3" or c["n"] < SMALL_CELL or c["n"] == 0)
                                       else f"{c['missed_by_both'] / c['n']:.3f}",
                "n_degenerate_loose_in_cell": c["n_degenerate_loose"],
                "b3_note": "excluded from LG-miss/missed-by-both (near-tautology: filter IS LG)"
                           if arm == "b3" else "",
            })

    # ---- Wilson CIs on per-arm missed-by-both fractions (overall primary) --
    wilson_rows = []
    for arm in ARMS:
        rows = breach[arm]
        n = len(rows)
        k = sum(v["missed_by_both"] for v in rows)
        lo, hi = wilson_ci(k, n)
        wilson_rows.append({
            "arm": arm, "n_breach": n, "missed_by_both": k,
            "fraction": f"{k / n:.4f}", "wilson95_lo": f"{lo:.4f}", "wilson95_hi": f"{hi:.4f}",
            "note": "NEAR-TAUTOLOGY: B3 filter IS Llama Guard; missed-by-both upper-bounded"
                    " by construction on the LG side" if arm == "b3" else "",
        })

    # ---- write outputs -----------------------------------------------------
    os.makedirs(OUTDIR, exist_ok=True)

    item_fields = ["arm", "id", "category", "len_ws_tokens", "tercile",
                   "degenerate_loose", "degenerate_strict",
                   "lg_flagged", "bd_flagged", "lg_missed", "bd_missed", "missed_by_both"]
    with open(os.path.join(OUTDIR, "breach_items.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=item_fields)
        w.writeheader()
        for arm in ARMS:
            for v in sorted(breach[arm], key=lambda v: v["id"]):
                w.writerow({k: (v[k] if k != "arm" else arm) for k in item_fields})

    with open(os.path.join(OUTDIR, "crosstab_mechanism.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(crosstab_rows[0].keys()))
        w.writeheader()
        w.writerows(crosstab_rows)

    with open(os.path.join(OUTDIR, "wilson_ci_missed_by_both.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(wilson_rows[0].keys()))
        w.writeheader()
        w.writerows(wilson_rows)

    summary = {
        "exhibit": "analysis4_classifier_blindness_mechanism",
        "governance": "POST-HOC EXPLORATORY (Amendment 24.1); diagnostic only; never a Table 1/2 number",
        "is_paper_number": False,
        "scope": "primary categories only (prefilling, persona, many_shot; n=180 prompts/arm);"
                 " crisis_adjacent never pooled (Amendment 13) and not re-cut here -"
                 " its classifier coverage is in the parent exhibit",
        "length_source": "whitespace-token count of response_full_turn (delivered assistant turn;"
                         " includes attacker-written prefix on prefilling items)."
                         " No generated-token counts exist in generation metadata"
                         " (only prompt_n_tokens), so recorded token counts were unavailable.",
        "degeneracy_source": "per-item degenerate_loose/degenerate_strict flags carried in each arm's"
                             " canonical scored file (degeneracy_definition_source:"
                             " scripts/analyze_degeneracy.py). All five arms assessable;"
                             " flag used in cross-tabs: degenerate_loose.",
        "b3_choice": "b3 EXCLUDED from LG-missed and missed-by-both mechanism cells"
                     " (near-tautology: the B3 filter IS Llama Guard); BD-missed retained;"
                     " b3 Wilson CI reported but flagged.",
        "small_cell_rule": "any denominator < 10: counts only, no percentage",
        "tercile_rule": "rank-based within arm over the BREACH set, ties broken by id,"
                        " sizes as equal as possible (remainder to earlier terciles)",
        "consistency_anchors_passed": EXPECTED,
        "wilson_ci": wilson_rows,
        "crosstab": crosstab_rows,
        "provenance": provenance,
    }
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("OK: wrote breach_items.csv, crosstab_mechanism.csv, wilson_ci_missed_by_both.csv, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
