r"""make_figures.py - build figures/family_seed.pdf from the locked results files.

Left panel: attack-success rate by family, B3 against T, with the paired-difference 95%
bootstrap interval printed under each family (tables_final.json, per_category_paired_diff_NEW).
Right panel: T minus B3 by training seed for the primary and crisis endpoints, with the 95%
bootstrap intervals from the three stats reports (seed 1 pre-registered; 2 and 3 post hoc).

Deterministic, CPU only, reads locked files, writes one PDF. No number is computed here that is
not already in the locked files. Run from the repo root:
  python tools/make_figures.py --root C:\proj71 --out paper\figures\family_seed.pdf
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\proj71")
    ap.add_argument("--out", default=os.path.join("figures", "family_seed.pdf"))
    args = ap.parse_args()

    tf = json.load(open(os.path.join(args.root, "results", "tables_final", "tables_final.json"), encoding="utf-8"))
    reports = {
        1: json.load(open(os.path.join(args.root, "results", "stats_report_realsuite_handlabelled.json"), encoding="utf-8")),
        2: json.load(open(os.path.join(args.root, "results", "stats_report_ts2_posthoc.json"), encoding="utf-8")),
        3: json.load(open(os.path.join(args.root, "results", "stats_report_ts3_posthoc.json"), encoding="utf-8")),
    }

    # ---- left panel data: per family ---------------------------------------
    pc = tf["per_category_paired_diff_NEW"]
    fams = [("prefilling", "Prefilling"), ("persona", "Persona"), ("many_shot", "Many-shot")]
    c1 = reports[1]["primary_test"]["crisis_co_primary"]
    b3 = [100 * pc[f]["b3"] for f, _ in fams] + [100 * c1["baseline_failure_rate"]]
    t = [100 * pc[f]["t"] for f, _ in fams] + [100 * c1["treatment_failure_rate"]]
    diffs = [100 * pc[f]["diff_t_minus_b3"] for f, _ in fams] + [100 * c1["difference_treatment_minus_baseline"]]
    cis = [[100 * x for x in pc[f]["ci95_bootstrap_over_prompts"]] for f, _ in fams] + [[100 * x for x in c1["ci95_bootstrap_over_prompts"]]]
    labels = [l for _, l in fams] + ["Crisis\n(co-primary)"]

    # ---- right panel data: seed panel ----------------------------------------
    seeds = [1, 2, 3]
    prim = []; prim_ci = []; cri = []; cri_ci = []
    for s in seeds:
        pt = reports[s]["primary_test"]
        d = pt["asr_difference_treatment_minus_baseline"]
        prim.append(100 * d["point"]); prim_ci.append([100 * x for x in d["ci95_bootstrap_over_prompts"]])
        c = pt["crisis_co_primary"]
        cri.append(100 * c["difference_treatment_minus_baseline"]); cri_ci.append([100 * x for x in c["ci95_bootstrap_over_prompts"]])

    # ---- draw -------------------------------------------------------------------
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})

    x = range(len(labels)); w = 0.36
    ax1.bar([i - w / 2 for i in x], b3, w, color="0.75", edgecolor="black", label="B3 (filter arm)")
    ax1.bar([i + w / 2 for i in x], t, w, color="0.25", edgecolor="black", label="T (safety-mixed)")
    for i in x:
        fmt = lambda v: "0.0" if abs(v) < 0.05 else f"{v:+.1f}"
        ax1.text(i, max(b3[i], t[i]) + 2.5, f"{fmt(diffs[i])}\n[{fmt(cis[i][0])}, {fmt(cis[i][1])}]",
                 ha="center", va="bottom", fontsize=6.5)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Attack-success or failure rate (%)")
    ax1.set_ylim(0, 85); ax1.legend(frameon=False, fontsize=7, loc="upper right")
    ax1.set_title("(a) By family, seed 1, 60 prompts each", fontsize=8)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    off = 0.15
    for k, s in enumerate(seeds):
        ax2.errorbar(s - off, prim[k], yerr=[[prim[k] - prim_ci[k][0]], [prim_ci[k][1] - prim[k]]],
                     fmt="o", color="black", capsize=3, label="Primary (n=180)" if k == 0 else None)
        ax2.errorbar(s + off, cri[k], yerr=[[cri[k] - cri_ci[k][0]], [cri_ci[k][1] - cri[k]]],
                     fmt="s", color="0.55", capsize=3, label="Crisis (n=60)" if k == 0 else None)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.axhline(-10, color="black", lw=0.6, ls="--")
    ax2.text(3.42, -10, "threshold", fontsize=6.5, va="bottom", ha="right")
    ax2.set_xticks(seeds); ax2.set_xticklabels(["seed 1\n(pre-registered)", "seed 2\n(post hoc)", "seed 3\n(post hoc)"], fontsize=7)
    ax2.set_ylabel("T minus B3 (points)")
    ax2.set_title("(b) By training seed, 95% bootstrap CIs", fontsize=8)
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)
    print("left panel:", list(zip(labels, [round(v, 2) for v in b3], [round(v, 2) for v in t], [round(v, 2) for v in diffs])))
    print("right panel primary:", [round(v, 2) for v in prim], "crisis:", [round(v, 2) for v in cri])
    return 0


if __name__ == "__main__":
    sys.exit(main())
