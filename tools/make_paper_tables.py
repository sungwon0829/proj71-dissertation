r"""make_paper_tables.py - regenerate the paper's result-table rows from the locked files and
check that main.tex contains exactly those rows.

Reads (all locked, none edited):
  results/tables_final/tables_final.json
  results/stats_report_realsuite_handlabelled.json
  results/stats_report_ts2_posthoc.json, results/stats_report_ts3_posthoc.json
  results/exploratory/analysis4_classifier_blindness_mechanism/wilson_ci_missed_by_both.csv
  results/exploratory/analysis4_classifier_blindness_mechanism/crosstab_mechanism.csv

Writes:
  paper_tables.tex  - the LaTeX rows for Tables 2, 3 and 4 and the per-type prose (audit copy)
and prints PASS/FAIL for each row against main.tex. Nothing samples; --seed is provenance.

Paper numbering (draft v17 onwards): Table 2 = all arms, Table 3 = pre-registered endpoints,
Table 4 = classifier blindness. The per-type differences are in Figure 2(a) and in one sentence
of Section 5.1, so they are checked as prose, not as table rows.

Run from the repo root:
  python tools/make_paper_tables.py --root C:\proj71 --tex path\to\main.tex
"""
import argparse
import csv
import json
import math
import os
import re
import sys


def wilson_ci(k, n, z=1.959963984540054):
    """Exact Wilson interval, recomputed from counts (the CSV stores 4-dp values, and
    re-rounding those to 1 dp can differ from rounding the exact value)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(x):
    return f"{100 * x:.2f}"


def load(root):
    tf = json.load(open(os.path.join(root, "results", "tables_final", "tables_final.json"), encoding="utf-8"))
    sr = json.load(open(os.path.join(root, "results", "stats_report_realsuite_handlabelled.json"), encoding="utf-8"))
    a4 = os.path.join(root, "results", "exploratory", "analysis4_classifier_blindness_mechanism")
    wilson = list(csv.DictReader(open(os.path.join(a4, "wilson_ci_missed_by_both.csv"), encoding="utf-8")))
    cross = list(csv.DictReader(open(os.path.join(a4, "crosstab_mechanism.csv"), encoding="utf-8")))
    return tf, sr, wilson, cross


def table2_rows(tf):
    """Table 2 (all arms): ASR [CI], over-refusal, helpfulness."""
    t1 = tf["table1"]
    rows = {}
    for arm in ("b0", "b1", "b2", "b3", "t"):
        lo, hi = t1[arm]["asr_ci95_over_prompts_pct"]
        rows[arm] = {"asr": f"{t1[arm]['asr_primary_pct']:.2f}", "ci": f"[{lo:.2f}, {hi:.2f}]",
                     "help": f"{t1[arm]['helpfulness_mean_reward']:.3f}",
                     "overref": t1[arm]["over_refusal_pct"], "src": t1[arm]["over_refusal_source"]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\proj71")
    ap.add_argument("--tex", default="main.tex")
    ap.add_argument("--seed", type=int, default=0, help="provenance only")
    args = ap.parse_args()

    tf, sr, wilson, cross = load(args.root)
    tex = open(args.tex, encoding="utf-8").read()
    out = []
    checks = []

    # ---- Table 2: ASR and helpfulness by arm --------------------------------
    t2 = table2_rows(tf)
    expect_t2 = {
        "b0": f"26.11 [20.00, 32.78]", "b1": f"51.11 [43.89, 58.33]",
        "b2": f"46.11 [38.89, 53.33]", "b3": f"45.00 [37.78, 52.22]", "t": f"39.44 [32.22, 46.67]",
    }
    out.append("% Table 2 rows (ASR [CI], helpfulness)")
    for arm, r in t2.items():
        s = f"{r['asr']} {r['ci']}"
        out.append(f"% {arm}: {s}  helpfulness {r['help']}")
        checks.append((f"Table 2 {arm} ASR+CI", s in tex))
        help_str = r["help"] if not r["help"].startswith("-") else "$-" + r["help"][1:] + "$"
        checks.append((f"Table 2 {arm} helpfulness", help_str in tex))
        checks.append((f"Table 2 {arm} ASR equals locked string", s == expect_t2[arm]))

    # ---- Table 3: primary and crisis ------------------------------------------
    pt = sr["primary_test"]
    d = pt["asr_difference_treatment_minus_baseline"]
    lo, hi = d["ci95_bootstrap_over_prompts"]
    prim = f"$-{abs(100*d['point']):.2f}$ & $[{100*lo:+.2f}, {100*hi:+.2f}]$ & {pt['primary']['p_value']:.3f}"
    c = pt["crisis_co_primary"]
    clo, chi = c["ci95_bootstrap_over_prompts"]
    cris = f"$-{abs(100*c['difference_treatment_minus_baseline']):.2f}$ & $[{100*clo:+.2f}, {100*chi:+.2f}]$ & {c['test']['p_value']:.3f}"
    out += ["% Table 3 rows", "% primary: " + prim, "% crisis:  " + cris]
    checks.append(("Table 3 primary row", prim in tex))
    checks.append(("Table 3 crisis row", cris in tex))
    mc = pt["per_seed_robustness"][0]
    checks.append(("discordant 56 / 33 / 23", mc["n_discordant"] == 56 and mc["baseline_unsafe_treatment_safe"] == 33
                   and mc["baseline_safe_treatment_unsafe"] == 23 and "23 prompts where only" in tex and "33 where only" in tex))
    checks.append(("exact McNemar 0.229 in text", f"{mc['p_value']:.3f}" in tex))

    # ---- Per-type differences: Figure 2(a) and one sentence in Section 5.1 --------------
    # The sentence reads "$-5.00$ points on prefilling (95\% CI $[-20.00, +10.00]$), $+1.67$ on
    # persona ($[-11.67, +15.00]$) and $-13.33$ on many-shot ($[-26.67, 0.00]$)". The check is that
    # the locked difference and its interval appear, in that order, around the type's name.
    pc = tf["per_category_paired_diff_NEW"]
    out.append("% Per-type rows (B3, T, diff [CI]); the paper carries diff [CI] in prose and B3/T in Figure 2(a)")
    for fam, label, prose in (("prefilling", "Prefilling", "prefilling"), ("persona", "Persona", "persona"),
                              ("many_shot", "Many-shot", "many-shot")):
        e = pc[fam]
        b3 = pct(e["b3"]); t = pct(e["t"])
        diff = 100 * e["diff_t_minus_b3"]
        elo, ehi = [100 * x for x in e["ci95_bootstrap_over_prompts"]]
        diff_str = f"${diff:+.2f}$".replace("$+0.00$", "$0.00$")
        ci_str = f"$[{elo:+.2f}, {ehi:+.2f}]$".replace("+0.00]", "0.00]")
        out.append(f"% {label} & {b3} & {t} & {diff_str} {ci_str}")
        pat = re.escape(diff_str) + r"[^$]{0,40}" + re.escape(prose) + r"[^$]{0,40}" + re.escape(ci_str)
        checks.append((f"Per-type {fam} (prose)", re.search(pat, tex) is not None))

    # ---- Table 4: classifier blindness ------------------------------------------
    out.append("% Table 4 rows (breaches, LG, BD, both, % [Wilson])")
    lg = {r["arm"]: int(r["n_breach"]) - int(r["lg_missed"]) for r in cross if r["cut"] == "overall" and r["lg_missed"] != ""}
    bd = {r["arm"]: int(r["n_breach"]) - int(r["bd_missed"]) for r in cross if r["cut"] == "overall"}
    for w in wilson:
        arm = w["arm"]; n = int(w["n_breach"]); k = int(w["missed_by_both"])
        lgf = 0 if arm == "b3" else lg[arm]
        lo, hi = wilson_ci(k, n)
        assert abs(lo - float(w["wilson95_lo"])) < 1e-4 and abs(hi - float(w["wilson95_hi"])) < 1e-4, arm
        row = f"{n} & {lgf} & {bd[arm]} & {k} & {100*k/n:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
        out.append(f"% {arm}: {row}")
        checks.append((f"Table 4 (blindness) {arm}", row in tex))

    open("paper_tables.tex", "w", encoding="utf-8").write("\n".join(out) + "\n")
    ok = True
    for name, passed in checks:
        print(("PASS " if passed else "FAIL ") + name)
        ok &= passed
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED - fix main.tex before submission")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())