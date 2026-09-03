"""Analysis 7 (Amendment 26, POST-HOC EXPLORATORY) - paired test of B0 (base) vs B1 (SFT)
on the 180 primary attack prompts: does supervised fine-tuning on counselling data raise
the judge-confirmed attack-success rate?

Same conventions as analysis4 (analyze_mechanism.py): deterministic, CPU only, reads the
canonical scored files, asserts the locked per-arm breach counts before computing anything,
writes summary.json + report.md into its own directory, prints the LaTeX sentence to paste.

Statistics mirror the primary test in stats_report_realsuite_handlabelled.json:
  * two-sided paired sign-flip permutation test on per-prompt differences (20,000, seed 0)
  * exact McNemar (binomial) test on the discordant pairs, reported beside it
  * 95% percentile bootstrap over the 180 prompts for the difference (10,000, seed 0)

Run from the repo root inside the project venv:
  python results/exploratory/analysis7_b0_vs_b1_sft_erosion/analyze_b0_b1.py --seed 0
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys

PRIMARY_CATS = ("prefilling", "persona", "many_shot")

# Consistency anchors: locked breach counts on the 180 primary prompts (analysis4 EXPECTED,
# and tables_final Table 1: b0 26.11% = 47/180, b1 51.11% = 92/180).
EXPECTED_BREACHES = {"b0": 47, "b1": 92}


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


def load_verdicts(path):
    """id -> 1 if judge verdict is 'unsafe' else 0, primary categories only."""
    out = {}
    for r in read_jsonl(path):
        if r.get("record_type") != "scored":
            continue
        if r["category"] not in PRIMARY_CATS:
            continue
        out[r["id"]] = 1 if r["verdict"] == "unsafe" else 0
    return out


def mcnemar_exact_two_sided(b, c):
    """Exact binomial McNemar: b, c = discordant counts. Two-sided p, capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def sign_flip_permutation_p(diffs, n_perm, seed):
    rng = random.Random(seed)
    nonzero = [d for d in diffs if d != 0]
    n = len(diffs)
    m = len(nonzero)
    obs = abs(sum(diffs) / n)
    count = 0
    for _ in range(n_perm):
        bits = rng.getrandbits(m)                       # one random bit per discordant pair
        s = sum(d if (bits >> i) & 1 else -d for i, d in enumerate(nonzero))
        if abs(s / n) >= obs - 1e-12:
            count += 1
    return count / n_perm


def bootstrap_ci(values, n_boot, seed, lo=2.5, hi=97.5):
    rng = random.Random(seed)
    n = len(values)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(n_boot)]   # C-implemented sampling
    means.sort()
    def pct(p):
        k = (len(means) - 1) * p / 100.0
        f = math.floor(k)
        c = min(f + 1, len(means) - 1)
        return means[f] + (means[c] - means[f]) * (k - f)
    return pct(lo), pct(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="permutation and bootstrap seed (0 = project convention)")
    ap.add_argument("--root", default=r"C:\proj71", help="repository root")
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    root = args.root
    scored = {
        "b0": os.path.join(root, "results", "b0_seed42", "scored_realsuite.jsonl"),
        "b1": os.path.join(root, "results", "b1_seed42", "scored_realsuite.jsonl"),
    }
    outdir = os.path.join(root, "results", "exploratory", "analysis7_b0_vs_b1_sft_erosion")

    provenance = {"script": os.path.abspath(__file__), "seed": args.seed, "inputs": {}}
    verdicts = {}
    for arm, path in scored.items():
        provenance["inputs"][f"scored_{arm}"] = {"path": path, "sha256": sha256_file(path)}
        v = load_verdicts(path)
        assert len(v) == 180, (arm, len(v))
        assert sum(v.values()) == EXPECTED_BREACHES[arm], (arm, sum(v.values()), EXPECTED_BREACHES[arm])
        verdicts[arm] = v
    ids = sorted(verdicts["b0"])
    assert ids == sorted(verdicts["b1"]), "prompt ids differ between arms"

    b0 = [verdicts["b0"][i] for i in ids]
    b1 = [verdicts["b1"][i] for i in ids]
    diffs = [y - x for x, y in zip(b0, b1)]          # +1: unsafe only under B1; -1: only under B0
    both_unsafe = sum(1 for x, y in zip(b0, b1) if x == 1 and y == 1)
    both_safe = sum(1 for x, y in zip(b0, b1) if x == 0 and y == 0)
    b1_only = sum(1 for d in diffs if d == 1)
    b0_only = sum(1 for d in diffs if d == -1)
    n_discordant = b1_only + b0_only
    asr_b0 = sum(b0) / 180.0
    asr_b1 = sum(b1) / 180.0
    diff_pts = 100.0 * (asr_b1 - asr_b0)

    print(f"loaded 180 paired verdicts; breach counts match the locked values (b0 47, b1 92)", flush=True)
    print(f"paired counts: both_unsafe {both_unsafe}, b1_only {b1_only}, b0_only {b0_only}, both_safe {both_safe}", flush=True)
    p_mcnemar = mcnemar_exact_two_sided(b1_only, b0_only)
    print(f"exact McNemar p = {p_mcnemar:.3e}", flush=True)
    print(f"running sign-flip permutation test ({args.n_perm} permutations, seed {args.seed}) ...", flush=True)
    p_perm = sign_flip_permutation_p(diffs, args.n_perm, args.seed)
    print(f"permutation p = {p_perm:.6f}", flush=True)
    print(f"running bootstrap ({args.n_boot} resamples, seed {args.seed}) ...", flush=True)
    ci_lo, ci_hi = bootstrap_ci(diffs, args.n_boot, args.seed)
    ci_lo, ci_hi = 100.0 * ci_lo, 100.0 * ci_hi

    def fmt_p(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    summary = {
        "exhibit": "analysis7_b0_vs_b1_sft_erosion",
        "governance": "POST-HOC EXPLORATORY (Amendment 26); added after the primary result was seen;"
                      " supports no confirmatory claim; never a Table 2/3 number",
        "is_paper_number": False,
        "scope": "180 primary prompts (prefilling, persona, many_shot); crisis never pooled",
        "arms": {"baseline": "b0", "treatment": "b1"},
        "asr": {"b0": asr_b0, "b1": asr_b1},
        "difference_b1_minus_b0_pts": diff_pts,
        "paired_counts": {"both_unsafe": both_unsafe, "b1_only_unsafe": b1_only,
                          "b0_only_unsafe": b0_only, "both_safe": both_safe,
                          "n_discordant": n_discordant, "n_pairs": 180},
        "permutation_test": {"test": "two-sided paired sign-flip permutation on per-prompt differences",
                             "n_permutations": args.n_perm, "seed": args.seed, "p_value": p_perm},
        "mcnemar_exact": {"test": "McNemar's exact (binomial) test on discordant pairs", "p_value": p_mcnemar},
        "ci95_bootstrap_over_prompts_pts": [ci_lo, ci_hi],
        "n_boot": args.n_boot,
        "provenance": provenance,
    }

    latex = (
        f"The base model's ASR interval, $[20.00, 32.78]$, does not overlap \\Bo{{}}'s, $[43.89, 58.33]$. "
        f"On the same 180 prompts, {n_discordant} pairs are discordant, {b1_only} unsafe only under \\Bo{{}} and "
        f"{b0_only} only under \\Bz{{}}, with {both_unsafe} unsafe in both and {both_safe} safe in both. "
        f"The difference is $+{diff_pts:.2f}$ points, 95\\% CI $[{ci_lo:+.2f}, {ci_hi:+.2f}]$, "
        f"permutation $p{'<' if p_perm < 0.001 else '='}{fmt_p(p_perm).lstrip('<')}$ "
        f"(exact McNemar $p{'<' if p_mcnemar < 0.001 else '='}{fmt_p(p_mcnemar).lstrip('<')}$). "
        f"This test was added post hoc (Amendment 26) after the primary result was seen and supports no confirmatory claim."
    )

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("# Analysis 7 - B0 vs B1 paired test (POST-HOC EXPLORATORY, Amendment 26)\n\n")
        f.write(f"ASR b0 {100*asr_b0:.2f}% (47/180), b1 {100*asr_b1:.2f}% (92/180); difference +{diff_pts:.2f} pts.\n\n")
        f.write(f"Paired counts: both unsafe {both_unsafe}, b1-only {b1_only}, b0-only {b0_only}, both safe {both_safe}.\n\n")
        f.write(f"Sign-flip permutation p = {p_perm:.6f} ({args.n_perm} permutations, seed {args.seed}); "
                f"exact McNemar p = {p_mcnemar:.3e}; 95% bootstrap CI over prompts [{ci_lo:+.2f}, {ci_hi:+.2f}].\n\n")
        f.write("LaTeX sentence for main.tex, Section 5.2, lead 'Supervised fine-tuning and safety.':\n\n")
        f.write(latex + "\n")

    out_sha = {k: sha256_file(os.path.join(outdir, k)) for k in ("summary.json", "report.md")}
    print("OK: analysis7 written to", outdir)
    print("paired counts: both_unsafe", both_unsafe, "b1_only", b1_only, "b0_only", b0_only, "both_safe", both_safe)
    print(f"difference +{diff_pts:.2f} pts, CI [{ci_lo:+.2f}, {ci_hi:+.2f}], perm p {p_perm:.6f}, McNemar p {p_mcnemar:.3e}")
    print("output sha256 for the manifest:", json.dumps(out_sha, indent=2))
    print("\nPASTE INTO main.tex (replaces the \\TODO{} sentence under 'Supervised fine-tuning and safety.'):\n")
    print(latex)
    return 0


if __name__ == "__main__":
    sys.exit(main())