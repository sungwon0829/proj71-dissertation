r"""recall_sensitivity.py - implied true-scale effect under the stats report's own approximation
(observed effect ~= judge recall x true effect, equal recall across arms assumed).

Reads the locked observed difference and its 95% CI from
results/stats_report_realsuite_handlabelled.json and prints the LaTeX rows for the appendix
table. Pure arithmetic on locked numbers; nothing is estimated.

Run from the repo root:  python tools/recall_sensitivity.py --root C:\proj71
"""
import argparse, json, os, sys

RECALLS = [(0.55, "inter-model check, n=36"), (0.65, ""), (0.79, "human validation, n=54"), (0.90, ""), (1.00, "no attenuation")]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=r"C:\proj71"); a = ap.parse_args()
    sr = json.load(open(os.path.join(a.root, "results", "stats_report_realsuite_handlabelled.json"), encoding="utf-8"))
    d = sr["primary_test"]["asr_difference_treatment_minus_baseline"]
    obs = 100 * d["point"]; lo, hi = [100 * x for x in d["ci95_bootstrap_over_prompts"]]
    print(f"% observed {obs:.2f} [{lo:+.2f}, {hi:+.2f}]; implied true = observed / recall")
    for r, note in RECALLS:
        print(f"{r:.2f} & {note} & ${obs/r:.2f}$ & $[{lo/r:+.2f}, {hi/r:+.2f}]$ \\\\")
    return 0

if __name__ == "__main__":
    sys.exit(main())
