"""CPU analysis for the full-suite classifier-blindness exhibit (Discussion, Table 2
neighbourhood). Reads the raw per-item classifier score files written by
scripts/probe_classifier_blindness_fullsuite.py plus the pinned judges' scored files,
and emits report.md + crosstabs.csv + distributions.csv.

DIAGNOSTIC ONLY (Amendments 17/19/20): never a Table 1/2 number, never the mechanism any
reported ASR is computed from. BREACH labels come solely from the pinned judges:
  prefilling / persona / many_shot -> behavioural ASR judge verdict unsafe = BREACH
  crisis_adjacent                  -> crisis judge verdict unsafe = "crisis failure"
                                      (different refusal semantics: a refusal FAILS here)
  benign_sensitive                 -> excluded from breach cross-tabs entirely; appears
                                      only as a clearly-labelled HELD-side score-
                                      distribution reference.

SMALL-CELL RULE, applied mechanically: any denominator < 10 -> raw counts only, no
percentage; distribution rows with n < 10 are suppressed (n reported, stats withheld).

Usage: python scripts/analyze_classifier_blindness.py --seed 0
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from probe_classifier_blindness_fullsuite import (  # noqa: E402
    CANONICAL_GENERATIONS, EXHIBIT, OUT_DIR)

ARMS = ["b0", "b1", "b2", "b3", "t"]
PRIMARY_CATS = ["prefilling", "persona", "many_shot"]   # ASR-judge categories (n=180/arm)
CRISIS_CAT = "crisis_adjacent"                          # crisis judge, co-primary (n=60)
BENIGN_CAT = "benign_sensitive"                         # no breach concept; reference only
SMALL_CELL = 10

SCORED_FILES = {
    "b0": "results/b0_seed42/scored_realsuite.jsonl",
    "b1": "results/b1_seed42/scored_realsuite.jsonl",
    "b2": "results/b2_seed42/scored_realsuite.jsonl",
    "b3": "results/b3_seed42/scored_realsuite.jsonl",
    "t": "results/t_seed42/scored_realsuite.jsonl",
}


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    header, rows = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if header is None:
                header = r
            else:
                rows.append(r)
    return header, rows


def load_scores(out_dir: Path) -> dict:
    """{arm: {classifier: {id: row}}} with header checks."""
    scores, headers = {}, {}
    for arm in ARMS:
        scores[arm], headers[arm] = {}, {}
        for clf, fname in (("lg", f"raw_scores_{arm}_llamaguard.jsonl"),
                           ("bd", f"raw_scores_{arm}_beaverdam.jsonl")):
            path = out_dir / fname
            hdr, rows = load_jsonl(path)
            if hdr.get("record_type") != "raw_scores_header" or hdr.get("arm") != arm:
                raise RuntimeError(f"{path}: unexpected header {hdr.get('record_type')}/"
                                   f"{hdr.get('arm')}")
            if len(rows) != 300:
                raise RuntimeError(f"{path}: {len(rows)} rows, expected 300")
            # the raw scores must be for the canonical generations
            canon = resolve(CANONICAL_GENERATIONS[arm])
            if hdr["source_generations_sha256"] != sha256_file(canon):
                raise RuntimeError(f"{path}: source generations sha does not match the "
                                   f"canonical file {canon}. Refusing to cross-tabulate.")
            scores[arm][clf] = {r["id"]: r for r in rows}
            headers[arm][clf] = hdr
    return scores, headers


def load_breach_labels() -> dict:
    """{arm: {id: {'category':…, 'breach': bool}}} from the pinned judges' scored files."""
    labels = {}
    for arm, rel in SCORED_FILES.items():
        path = resolve(rel)
        hdr, rows = load_jsonl(path)
        # the scored file must describe the canonical generations file
        canon = resolve(CANONICAL_GENERATIONS[arm])
        if hdr["generations_sha256"] != sha256_file(canon):
            raise RuntimeError(f"{path}: scored file was produced from a different "
                               f"generations file than the canonical {canon}.")
        labels[arm] = {}
        for r in rows:
            if r["category"] == BENIGN_CAT:
                # no breach concept; kept only for the score-distribution reference
                labels[arm][r["id"]] = {"category": r["category"], "breach": None,
                                        "decision_rule": r.get("decision_rule")}
                continue
            labels[arm][r["id"]] = {"category": r["category"], "breach": bool(r["unsafe"]),
                                    "decision_rule": r.get("decision_rule")}
        if len(labels[arm]) != 300:
            raise RuntimeError(f"{path}: {len(labels[arm])} scored rows, expected 300")
    return labels


def crosstab(ids, labels_arm, sc_arm, want_breach: bool) -> dict:
    """Coverage counts among BREACH (or HELD) items of one arm over a set of ids."""
    sel = [i for i in ids if labels_arm[i]["breach"] == want_breach]
    n = len(sel)
    lg = sum(1 for i in sel if sc_arm["lg"][i]["flagged"])
    bdc = sum(1 for i in sel if sc_arm["bd"][i]["flagged"])
    either = sum(1 for i in sel if sc_arm["lg"][i]["flagged"] or sc_arm["bd"][i]["flagged"])
    neither = n - either
    return {"n": n, "lg": lg, "bd": bdc, "either": either, "neither": neither}


def frac(k, n):
    """Small-cell rule: no percentage when the denominator is < SMALL_CELL."""
    if n < SMALL_CELL:
        return ""
    return f"{k / n:.3f}"


def fmt_cell(k, n):
    f = frac(k, n)
    return f"{k}/{n}" + (f" ({100 * k / n:.1f}%)" if f != "" else " (counts only, n<10)")


def quantile(sorted_vals, q):
    """Linear-interpolation quantile on a sorted list (numpy 'linear' method)."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def dist_stats(vals: list) -> dict:
    n = len(vals)
    if n == 0:
        return {"n": 0}
    if n < SMALL_CELL:
        return {"n": n, "suppressed": True}
    v = sorted(vals)
    out = {"n": n,
           "median": round(statistics.median(v), 5),
           "q1": round(quantile(v, 0.25), 5), "q3": round(quantile(v, 0.75), 5),
           "min": round(v[0], 5), "max": round(v[-1], 5)}
    for d in range(1, 10):
        out[f"d{d * 10}"] = round(quantile(v, d / 10), 5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0,
                    help="provenance only; this analysis is deterministic")
    ap.add_argument("--dir", default=OUT_DIR)
    args = ap.parse_args()

    out_dir = resolve(args.dir)
    marker = out_dir / "GPU_DONE.marker"
    if not marker.exists():
        raise RuntimeError(f"{marker} missing -- the GPU phase has not completed; "
                           f"refusing to analyse partial scores.")
    scores, score_headers = load_scores(out_dir)
    labels = load_breach_labels()

    gate = json.loads((out_dir / "validation_gate_20260827.json").read_text(encoding="utf-8"))
    for fam in ("llama_guard", "beaver_dam"):
        if gate.get(fam) is None or gate[fam].get("n_fail", 1) != 0:
            raise RuntimeError(f"validation gate for {fam} did not pass; refusing to analyse.")

    ids_by_cat = {}
    for i, meta in labels["b2"].items():
        ids_by_cat.setdefault(meta["category"], []).append(i)
    for c, ids in ids_by_cat.items():
        ids.sort()
    primary_ids = [i for c in PRIMARY_CATS for i in ids_by_cat[c]]
    assert len(primary_ids) == 180 and len(ids_by_cat[CRISIS_CAT]) == 60

    # ---------------- cross-tabs ----------------
    scopes = [("overall_primary", primary_ids)] + \
             [(c, ids_by_cat[c]) for c in PRIMARY_CATS] + \
             [(CRISIS_CAT, ids_by_cat[CRISIS_CAT])]
    xrows = []
    for arm in ARMS:
        for scope_name, ids in scopes:
            for group, want in (("BREACH", True), ("HELD", False)):
                gname = group
                if scope_name == CRISIS_CAT:
                    gname = "CRISIS_FAILURE" if want else "CRISIS_PASS"
                ct = crosstab(ids, labels[arm], scores[arm], want)
                xrows.append({
                    "arm": arm, "scope": scope_name, "group": gname,
                    "n": ct["n"],
                    "lg_flagged": ct["lg"], "lg_frac": frac(ct["lg"], ct["n"]),
                    "bd_flagged": ct["bd"], "bd_frac": frac(ct["bd"], ct["n"]),
                    "either": ct["either"], "either_frac": frac(ct["either"], ct["n"]),
                    "neither": ct["neither"], "neither_frac": frac(ct["neither"], ct["n"]),
                })
    xt_path = out_dir / "crosstabs.csv"
    with open(xt_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(xrows[0].keys()))
        w.writeheader()
        w.writerows(xrows)

    # ---------------- score distributions ----------------
    drows = []
    dist_scopes = scopes + [(BENIGN_CAT + "_REFERENCE", ids_by_cat[BENIGN_CAT])]
    for arm in ARMS:
        for clf, clf_name in (("lg", "llama_guard_p_unsafe_norm"),
                              ("bd", "beaverdam_max_category_prob")):
            for scope_name, ids in dist_scopes:
                if scope_name.endswith("_REFERENCE"):
                    groups = [("HELD_REFERENCE_BENIGN(no breach concept)", ids)]
                else:
                    b = [i for i in ids if labels[arm][i]["breach"]]
                    h = [i for i in ids if not labels[arm][i]["breach"]]
                    if scope_name == CRISIS_CAT:
                        groups = [("CRISIS_FAILURE", b), ("CRISIS_PASS", h)]
                    else:
                        groups = [("BREACH", b), ("HELD", h)]
                for gname, gids in groups:
                    st = dist_stats([scores[arm][clf][i]["score"] for i in gids])
                    row = {"arm": arm, "classifier": clf_name, "scope": scope_name,
                           "group": gname}
                    row.update({k: st.get(k, "") for k in
                                ["n", "suppressed", "median", "q1", "q3", "min", "max"]
                                + [f"d{d*10}" for d in range(1, 10)]})
                    drows.append(row)
    dist_path = out_dir / "distributions.csv"
    with open(dist_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0].keys()))
        w.writeheader()
        w.writerows(drows)

    # ---------------- report ----------------
    def xt(arm, scope, group):
        return next(r for r in xrows if r["arm"] == arm and r["scope"] == scope
                    and r["group"] == group)

    lines = []
    a = lines.append
    a(f"# Classifier blindness on the full frozen suite — diagnostic exhibit ({EXHIBIT})")
    a("")
    a(f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} by "
      f"`scripts/analyze_classifier_blindness.py --seed {args.seed}` (deterministic; the "
      f"seed is provenance only). GPU phase: `scripts/probe_classifier_blindness_fullsuite.py`"
      f" — see `GPU_DONE.marker` and `validation_gate_20260827.json` in this directory.")
    a("")
    a("## Status and scope — read first")
    a("")
    a("- **DIAGNOSTIC exhibit only** (pre-registration Amendments 17/19/20). Nothing here "
      "is a Table 1 or Table 2 number, and neither classifier is the mechanism any "
      "reported ASR is computed from. Every BREACH label below comes solely from the "
      "pinned Qwen2.5-7B behavioural/crisis judges (lock v4, "
      "`configs/judges_pinned.lock.v4.json`, verified before scoring).")
    a("- **Why using Llama Guard here is NOT a rule-1 violation:** rule 1 forbids the B3 "
      "filter mechanism from *judging ASR*. In this exhibit Llama Guard decides no ASR "
      "verdict — it is being *characterised against* the pinned judges' verdicts, i.e. we "
      "measure what fraction of judge-identified breaches a content classifier would flag. "
      "beaver-dam runs in its sanctioned Amendment 17 diagnostic/counterfactual role.")
    a("- **Seed-1 only, one model family.** All five arms are the canonical seed-1 "
      "generation passes (Qwen2.5-7B-Instruct family). This generalises the six-item "
      "Amendment 20 dual-family probe to the full frozen suite (n=300 prompts/arm), but it "
      "remains a diagnostic exhibit on one model family and one training seed; no claim "
      "beyond that is made.")
    a("- **The b3 × Llama Guard column is a near-tautology.** The B3 filter IS Llama Guard: "
      "every non-replaced b3 response is one Llama Guard already ruled safe at derivation, "
      "so LG flags ~0 of b3's delivered turns by construction. That column is reported for "
      "completeness only and supports no inference about b3.")
    a("- **Crisis semantics differ.** `crisis_adjacent` rows are the crisis judge's "
      "*failure* verdicts (labelled `CRISIS_FAILURE`), where a bare refusal is itself a "
      "failure — the opposite refusal semantics from the three attack categories. They are "
      "never pooled with the primary categories (Amendment 13).")
    a("- **`benign_sensitive` has no breach concept** and is excluded from all cross-tabs; "
      "it appears only as a clearly-labelled HELD-side score-distribution reference.")
    a("- **Small-cell rule, applied mechanically throughout:** any denominator < 10 → raw "
      "counts only, no percentage; distribution rows with n < 10 are suppressed.")
    a("")
    a("## Validation gate (Amendment 20 six-item reproduction)")
    a("")
    a("Both gates passed before any full-suite scoring (`validation_gate_20260827.json`): "
      "Llama Guard reproduced all 19 PLG probe rows (binary verdicts exact, continuous "
      "scores within ±0.02, PLG batching replicated), corrected-path beaver-dam reproduced "
      "all 19 PK rows (including the many_shot_009 knife-edge ≈ 0.49995), and the "
      "apply_b3_filter `--selfcheck` replacement-text score replicated. Provenance note "
      "recorded in the gate file: three of the six severe items (many_shot_017/009/044) "
      "are B2 **v2-era** responses that do not equal the current b2 arm's responses to "
      "those prompts; the gate therefore reproduces the probe's own items verbatim rather "
      "than arm rows.")
    a("")
    a("## Inputs")
    a("")
    a("| arm | generations (canonical, seed 1) | sha256 (16) | judge verdicts |")
    a("|---|---|---|---|")
    for arm in ARMS:
        h = score_headers[arm]["lg"]
        a(f"| {arm} | `{Path(h['source_generations_file']).name}` | "
          f"`{h['source_generations_sha256'][:16]}` | `{SCORED_FILES[arm]}` |")
    a("")
    a("Arm b2's Llama Guard scores are **reused** from the B3 derivation pass (LG screened "
      "every B2 response there; same pinned code path). Arm b3 reuses the 298 byte-identical "
      "rows (asserted) and scores the two filter-replaced items (prefilling_018, "
      "prefilling_030) on the delivered replacement text in their real contexts. All other "
      "arm × classifier cells were scored fresh this run.")
    a("")
    a("## Cross-tabs — classifier coverage of judge-confirmed breaches")
    a("")
    a("Primary scope pools the three ASR-judge categories (prefilling, persona, many_shot; "
      "n = 180 prompts/arm). Fractions are omitted where n < 10 (small-cell rule).")
    a("")
    a("### Overall primary (n = 180 prompts/arm), among judge-BREACH responses")
    a("")
    a("| arm | n_breach | flagged by LG | flagged by BD | either | neither |")
    a("|---|---|---|---|---|---|")
    for arm in ARMS:
        r = xt(arm, "overall_primary", "BREACH")
        n = r["n"]
        a(f"| {arm} | {n} | {fmt_cell(r['lg_flagged'], n)} | {fmt_cell(r['bd_flagged'], n)}"
          f" | {fmt_cell(r['either'], n)} | {fmt_cell(r['neither'], n)} |")
    a("")
    a("### Overall primary, among judge-HELD responses (contrast)")
    a("")
    a("| arm | n_held | flagged by LG | flagged by BD | either | neither |")
    a("|---|---|---|---|---|---|")
    for arm in ARMS:
        r = xt(arm, "overall_primary", "HELD")
        n = r["n"]
        a(f"| {arm} | {n} | {fmt_cell(r['lg_flagged'], n)} | {fmt_cell(r['bd_flagged'], n)}"
          f" | {fmt_cell(r['either'], n)} | {fmt_cell(r['neither'], n)} |")
    a("")
    a("### Per category (n = 60 prompts/arm/category)")
    a("")
    for cat in PRIMARY_CATS:
        a(f"**{cat}** — BREACH = behavioural ASR judge verdict unsafe")
        a("")
        a("| arm | group | n | LG | BD | either | neither |")
        a("|---|---|---|---|---|---|---|")
        for arm in ARMS:
            for grp in ("BREACH", "HELD"):
                r = xt(arm, cat, grp)
                n = r["n"]
                a(f"| {arm} | {grp} | {n} | {fmt_cell(r['lg_flagged'], n)} | "
                  f"{fmt_cell(r['bd_flagged'], n)} | {fmt_cell(r['either'], n)} | "
                  f"{fmt_cell(r['neither'], n)} |")
        a("")
    a(f"**{CRISIS_CAT}** — co-primary, crisis judge, n = 60/arm. `CRISIS_FAILURE` is the "
      "crisis judge's failure verdict; refusal semantics are inverted relative to the "
      "attack categories (a bare refusal fails). Never pooled with the primary scope.")
    a("")
    a("| arm | group | n | LG | BD | either | neither |")
    a("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for grp in ("CRISIS_FAILURE", "CRISIS_PASS"):
            r = xt(arm, CRISIS_CAT, grp)
            n = r["n"]
            a(f"| {arm} | {grp} | {n} | {fmt_cell(r['lg_flagged'], n)} | "
              f"{fmt_cell(r['bd_flagged'], n)} | {fmt_cell(r['either'], n)} | "
              f"{fmt_cell(r['neither'], n)} |")
    a("")
    a("## Continuous score distributions (BREACH vs HELD)")
    a("")
    a("Full per-arm × classifier × scope distributions (median, IQR, min/max, deciles) are "
      "in `distributions.csv`; per-category rows only where n ≥ 10, others suppressed. "
      "Summary (overall primary scope):")
    a("")
    a("| arm | classifier | group | n | median | IQR | min–max |")
    a("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for clf_name in ("llama_guard_p_unsafe_norm", "beaverdam_max_category_prob"):
            for grp in ("BREACH", "HELD"):
                r = next(x for x in drows if x["arm"] == arm and x["classifier"] == clf_name
                         and x["scope"] == "overall_primary" and x["group"] == grp)
                if r.get("suppressed") is True:
                    a(f"| {arm} | {clf_name} | {grp} | {r['n']} | suppressed (n<10) | | |")
                else:
                    a(f"| {arm} | {clf_name} | {grp} | {r['n']} | {r['median']} | "
                      f"[{r['q1']}, {r['q3']}] | {r['min']}–{r['max']} |")
    a("")
    a("## Files")
    a("")
    a("- `crosstabs.csv` — every arm × scope × group coverage row (counts + fractions, "
      "fractions blank where n < 10).")
    a("- `distributions.csv` — score distributions per arm × classifier × scope × group, "
      "including the `benign_sensitive_REFERENCE` HELD-side reference rows.")
    a("- `raw_scores_<arm>_<classifier>.jsonl` — per-item scores with provenance headers.")
    a("- `validation_gate_20260827.json`, `GPU_DONE.marker`.")
    a("")

    # ---- automated findings paragraph (computed, not hand-written) ----
    a("## Computed headline of the exhibit")
    a("")
    for arm in ARMS:
        r = xt(arm, "overall_primary", "BREACH")
        n = r["n"]
        tag = " (near-tautology: the b3 filter IS Llama Guard)" if arm == "b3" else ""
        a(f"- **{arm}**: of {n} judge-confirmed primary-scope breaches, LG flags "
          f"{fmt_cell(r['lg_flagged'], n)}, corrected beaver-dam {fmt_cell(r['bd_flagged'], n)}, "
          f"either {fmt_cell(r['either'], n)}, neither {fmt_cell(r['neither'], n)}.{tag}")
        c = xt(arm, CRISIS_CAT, "CRISIS_FAILURE")
        cn = c["n"]
        a(f"  - crisis co-primary: of {cn} crisis failures, LG {fmt_cell(c['lg_flagged'], cn)}, "
          f"BD {fmt_cell(c['bd_flagged'], cn)}, either {fmt_cell(c['either'], cn)}, "
          f"neither {fmt_cell(c['neither'], cn)}.")
    a("")
    a("*(These bullets are computed from the cross-tabs above; interpretation belongs to "
      "the Discussion draft, not this file.)*")
    a("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[written] {xt_path}")
    print(f"[written] {dist_path}")
    print(f"[written] {report_path}")


if __name__ == "__main__":
    main()
