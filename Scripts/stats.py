"""
Project 71 -- PASS 3 of 3: statistics, Tables 1 and 2, and the headline sentence.

Feeds: Table 1 (ASR / over-refusal / helpfulness per arm), Table 2 (per-category ASR,
B3 vs T), and the headline sentence the harness must emit automatically.

Reads ONLY scored files produced by eval_score.py. It never touches raw generations and
never re-runs a judge. Every number it prints is traceable to a scored.jsonl on disk.

STATISTICS (CLAUDE.md safeguards v2 rule 5)
  PRIMARY TEST -- McNemar's exact test on PAIRED binary safe/unsafe outcomes across the
  prompt set. Both arms see identical prompts, so the test runs over hundreds of paired
  outcomes rather than over 3 seeds. Because each prompt is evaluated once per seed, the
  per-prompt outcome is reduced across seeds by MAJORITY VOTE before testing, which keeps
  one independent observation per prompt. Per-seed McNemar tests are also reported as a
  robustness check. Pooling all seed-prompt pairs into one test would trebly count each
  prompt and is deliberately NOT done.
  Exact (binomial) McNemar is used throughout, not the chi-square approximation, because
  the discordant counts can be small.

  BOOTSTRAP CIs over the prompt set -- prompts are resampled with replacement and the SAME
  resampled prompt ids are applied to every arm, so the pairing that makes the comparison
  valid is preserved in every replicate.

  SEED VARIANCE is reported separately as a robustness check, never as the primary n.
  With 3 seeds the across-seed interval uses the t distribution (df=2), not 1.96.

NEVER FABRICATE. If any quantity needed for the headline sentence is missing -- an arm
absent, a seed missing, helpfulness unscored -- this script prints an explicit placeholder
and states what is missing. It will not print a half-filled sentence.

Usage:
    python Scripts\\stats.py --results_dir results --scored_name scored_realsuite.jsonl
    python Scripts\\stats.py --results_dir results --treatment t --baseline b3
"""

import argparse
import datetime
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTACK_CATEGORIES = ["prefilling", "persona", "many_shot", "crisis_adjacent"]
OVER_REFUSAL_CATEGORY = "benign_sensitive"
MISSING = "<NOT COMPUTED -- see reason>"


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_scored(path: Path):
    """Return (header, {id: row}). Fails loudly rather than skipping."""
    header, rows = None, {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("record_type")
            if t == "scored_header":
                header = r
            elif t == "scored":
                if r["id"] in rows:
                    raise ValueError(f"{path}:{lineno}: duplicate id {r['id']!r}")
                rows[r["id"]] = r
            else:
                raise ValueError(f"{path}:{lineno}: unexpected record_type {t!r}")
    if header is None:
        raise ValueError(f"{path}: no scored_header record")
    if not rows:
        raise ValueError(f"{path}: no scored records")
    return header, rows


def discover(results_dir: Path, scored_name: str):
    """results/<arm>_seed<N>/<scored_name> -> {arm: {seed: (header, rows)}}"""
    out = defaultdict(dict)
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"(.+)_seed(\d+)", d.name)
        if not m:
            continue
        path = d / scored_name
        if not path.is_file():
            continue
        arm, seed = m.group(1).lower(), int(m.group(2))
        header, rows = load_scored(path)
        if header["arm"].lower() != arm:
            raise ValueError(f"{path}: header arm {header['arm']!r} != directory arm {arm!r}")
        if int(header["seed"]) != seed:
            raise ValueError(f"{path}: header seed {header['seed']} != directory seed {seed}")
        out[arm][seed] = {"header": header, "rows": rows, "path": path}
    return dict(out)


def check_identical_treatment(data):
    """Every arm must have been evaluated on the same suite with the same judges.

    Any asymmetry here invalidates the comparison, so this is a hard failure, not a warning.
    """
    ref = None
    problems = []
    for arm, seeds in sorted(data.items()):
        for seed, d in sorted(seeds.items()):
            h = d["header"]
            key = {
                "suite_sha256": h["suite_sha256"],
                "judge_prompts": {k: v["prompt_sha256"] for k, v in h["judges"].items()},
                "judge_models": {k: f"{v['actual_model']}@{v['actual_revision']}"
                                 for k, v in h["judges"].items()},
                "judge_target_field": h["judge_target_field"],
            }
            if ref is None:
                ref, ref_name = key, f"{arm}_seed{seed}"
            elif key != ref:
                problems.append(f"{arm}_seed{seed} differs from {ref_name}: {key} != {ref}")
    if problems:
        raise RuntimeError(
            "IDENTICAL TREATMENT VIOLATED -- arms were not evaluated under the same "
            "conditions, so they are not comparable:\n  " + "\n  ".join(problems))
    return ref


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def outcome_vectors(rows, prompt_ids, field):
    """Binary outcome per prompt id, in a fixed id order. Fails loudly on a missing id."""
    v = []
    for pid in prompt_ids:
        if pid not in rows:
            raise KeyError(f"prompt id {pid!r} missing from a scored file -- refusing to "
                           "compare arms over different prompt sets")
        r = rows[pid]
        if field not in r:
            raise KeyError(f"prompt id {pid!r} has no {field!r} field")
        v.append(bool(r[field]))
    return np.array(v, dtype=bool)


def arm_prompt_ids(data, arm, category_filter):
    seeds = sorted(data[arm])
    idsets = []
    for s in seeds:
        idsets.append({pid for pid, r in data[arm][s]["rows"].items()
                       if category_filter(r["category"])})
    common = set.intersection(*idsets) if idsets else set()
    for i, s in enumerate(seeds):
        if idsets[i] != common:
            raise ValueError(f"{arm} seed {s} has a different prompt set from its other seeds")
    return sorted(common)


def majority_vote(mats):
    """mats: (n_seeds, n_prompts) bool -> (n_prompts,) bool majority."""
    return mats.sum(axis=0) * 2 > mats.shape[0]


def mcnemar_exact(a: np.ndarray, b: np.ndarray):
    """Exact (binomial) McNemar for paired binary outcomes.

    a, b are aligned boolean vectors (True = unsafe). Returns discordant counts and the
    two-sided exact p-value under H0: P(b_only) == P(a_only) == 1/2.
    """
    if a.shape != b.shape:
        raise ValueError("McNemar requires aligned vectors of equal length")
    b_only = int(np.sum(~a & b))   # baseline safe, treatment unsafe
    a_only = int(np.sum(a & ~b))   # baseline unsafe, treatment safe
    n_disc = a_only + b_only
    if n_disc == 0:
        return {"n_pairs": int(a.size), "n_discordant": 0,
                "baseline_unsafe_treatment_safe": a_only,
                "baseline_safe_treatment_unsafe": b_only,
                "p_value": 1.0, "test": "McNemar's exact (binomial) test",
                "note": "no discordant pairs; p defined as 1.0"}
    if HAVE_SCIPY:
        p = float(scipy_stats.binomtest(b_only, n_disc, 0.5).pvalue)
    else:
        # exact two-sided binomial without scipy
        k, n = min(a_only, b_only), n_disc
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"n_pairs": int(a.size), "n_discordant": n_disc,
            "baseline_unsafe_treatment_safe": a_only,
            "baseline_safe_treatment_unsafe": b_only,
            "p_value": p, "test": "McNemar's exact (binomial) test",
            "scipy": HAVE_SCIPY}


def bootstrap_paired(vec_by_arm, n_boot, seed, statistic):
    """Percentile bootstrap over PROMPTS, with the same resampled ids applied to every arm.

    vec_by_arm: {arm: (n_prompts,) bool}. Resampling prompt indices jointly is what keeps
    the paired structure intact; resampling each arm independently would destroy it and
    inflate the CI on the difference.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(vec_by_arm.values())))
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out.append(statistic({a: v[idx] for a, v in vec_by_arm.items()}))
    return np.array(out, dtype=float)


def pct_ci(samples, alpha=0.05):
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def across_seed_ci(values):
    """Mean +/- 95% CI across seeds using the t distribution (df = n-1). n=3 -> t=4.303."""
    v = np.asarray(values, dtype=float)
    n = v.size
    mean = float(v.mean())
    if n < 2:
        return {"mean": mean, "n_seeds": n, "ci95": None, "sd": None,
                "note": "single run -- no across-seed interval (report in italics)"}
    sd = float(v.std(ddof=1))
    sem = sd / math.sqrt(n)
    tcrit = float(scipy_stats.t.ppf(0.975, n - 1)) if HAVE_SCIPY else \
        {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.776)
    return {"mean": mean, "n_seeds": n, "sd": sd, "sem": sem,
            "ci95": [mean - tcrit * sem, mean + tcrit * sem], "t_crit": tcrit,
            "note": "across-seed variability (robustness check, NOT the primary n)"}


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Project 71 pass 3: statistics and headline sentence")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--scored_name", default="scored_realsuite.jsonl",
                    help="Filename of the scored file inside each results/<arm>_seed<N>/ directory")
    ap.add_argument("--treatment", default="t")
    ap.add_argument("--baseline", default="b3")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_dir = resolve(args.results_dir)
    data = discover(results_dir, args.scored_name)
    if not data:
        raise SystemExit(f"No scored files named {args.scored_name!r} found under {results_dir}")

    treatment_conditions = check_identical_treatment(data)

    print(f"[stats] results dir: {results_dir}")
    print(f"[stats] scored file: {args.scored_name}")
    for arm in sorted(data):
        print(f"[stats] arm {arm}: seeds {sorted(data[arm])}")
    print(f"[stats] identical-treatment check PASSED "
          f"(suite {treatment_conditions['suite_sha256'][:12]}, judges "
          f"{ {k: v[:8] for k, v in treatment_conditions['judge_prompts'].items()} })")
    if not HAVE_SCIPY:
        print("[stats] WARNING: scipy not installed; using exact binomial fallback and "
              "tabulated t critical values.")

    report = {
        "record_type": "stats_report",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "results_dir": str(results_dir), "scored_name": args.scored_name,
        "treatment_arm": args.treatment, "baseline_arm": args.baseline,
        "identical_treatment": treatment_conditions,
        "n_boot": args.n_boot, "bootstrap_seed": args.seed,
        "arms": {}, "table1": {}, "table2": {}, "primary_test": None,
        "headline_sentence": None, "headline_blockers": [],
    }

    # ---------------- per-arm metrics (Table 1) and per-category ASR (Table 2) ----------
    is_attack = lambda c: c in ATTACK_CATEGORIES
    is_benign = lambda c: c == OVER_REFUSAL_CATEGORY

    for arm in sorted(data):
        seeds = sorted(data[arm])
        atk_ids = arm_prompt_ids(data, arm, is_attack)
        ben_ids = arm_prompt_ids(data, arm, is_benign)
        asr_by_seed, orr_by_seed, help_by_seed = [], [], []
        percat = defaultdict(list)
        for s in seeds:
            rows = data[arm][s]["rows"]
            asr_by_seed.append(float(outcome_vectors(rows, atk_ids, "unsafe").mean()))
            orr_by_seed.append(float(outcome_vectors(rows, ben_ids, "refused").mean()))
            for c in ATTACK_CATEGORIES:
                cids = [i for i in atk_ids if rows[i]["category"] == c]
                if cids:
                    percat[c].append(float(outcome_vectors(rows, cids, "unsafe").mean()))
            hs = [r["helpfulness"] for r in rows.values() if "helpfulness" in r]
            help_by_seed.append(float(np.mean(hs)) if hs else None)

        helpful_ok = all(h is not None for h in help_by_seed)
        report["arms"][arm] = {
            "seeds": seeds, "n_attack_prompts": len(atk_ids), "n_benign_prompts": len(ben_ids),
            "asr_by_seed": asr_by_seed, "over_refusal_by_seed": orr_by_seed,
            "helpfulness_by_seed": help_by_seed if helpful_ok else None,
            "helpfulness_available": helpful_ok,
        }
        report["table1"][arm] = {
            "asr": across_seed_ci(asr_by_seed),
            "over_refusal": across_seed_ci(orr_by_seed),
            "helpfulness": across_seed_ci(help_by_seed) if helpful_ok else
                {"mean": None, "n_seeds": len(seeds), "ci95": None,
                 "note": "NOT SCORED -- reward model unavailable at scoring time"},
            "single_run": len(seeds) == 1,
        }
        report["table2"][arm] = {c: across_seed_ci(v) for c, v in percat.items()}

    # ---------------- primary test: treatment vs baseline -------------------------------
    T, B = args.treatment.lower(), args.baseline.lower()
    blockers = []
    if T not in data:
        blockers.append(f"treatment arm {T!r} has no scored results")
    if B not in data:
        blockers.append(f"baseline arm {B!r} has no scored results")

    if not blockers:
        atk_ids = arm_prompt_ids(data, T, is_attack)
        if atk_ids != arm_prompt_ids(data, B, is_attack):
            raise RuntimeError(f"{T} and {B} were scored on different attack prompt sets; "
                               "the paired test is invalid.")
        ben_ids = arm_prompt_ids(data, T, is_benign)

        t_seeds, b_seeds = sorted(data[T]), sorted(data[B])
        t_mat = np.array([outcome_vectors(data[T][s]["rows"], atk_ids, "unsafe") for s in t_seeds])
        b_mat = np.array([outcome_vectors(data[B][s]["rows"], atk_ids, "unsafe") for s in b_seeds])
        t_maj, b_maj = majority_vote(t_mat), majority_vote(b_mat)

        primary = mcnemar_exact(b_maj, t_maj)
        primary["reduction_across_seeds"] = "per-prompt majority vote across seeds"
        primary["baseline_asr"] = float(b_maj.mean())
        primary["treatment_asr"] = float(t_maj.mean())

        per_seed = []
        for s in sorted(set(t_seeds) & set(b_seeds)):
            r = mcnemar_exact(outcome_vectors(data[B][s]["rows"], atk_ids, "unsafe"),
                              outcome_vectors(data[T][s]["rows"], atk_ids, "unsafe"))
            r["seed"] = s
            per_seed.append(r)

        # bootstrap over prompts for the ASR difference (paired resampling)
        boot_asr = bootstrap_paired({"t": t_maj, "b": b_maj}, args.n_boot, args.seed,
                                    lambda d: d["t"].mean() - d["b"].mean())
        asr_diff_ci = pct_ci(boot_asr)

        t_ben = majority_vote(np.array([outcome_vectors(data[T][s]["rows"], ben_ids, "refused")
                                        for s in t_seeds]))
        b_ben = majority_vote(np.array([outcome_vectors(data[B][s]["rows"], ben_ids, "refused")
                                        for s in b_seeds]))
        boot_orr = bootstrap_paired({"t": t_ben, "b": b_ben}, args.n_boot, args.seed,
                                    lambda d: d["t"].mean() - d["b"].mean())
        orr_diff_ci = pct_ci(boot_orr)
        orr_mcnemar = mcnemar_exact(b_ben, t_ben)

        report["primary_test"] = {
            "contrast": f"{T} vs {B}",
            "primary": primary, "per_seed_robustness": per_seed,
            "asr_difference_treatment_minus_baseline": {
                "point": float(t_maj.mean() - b_maj.mean()),
                "ci95_bootstrap_over_prompts": asr_diff_ci, "n_boot": args.n_boot},
            "over_refusal_difference_treatment_minus_baseline": {
                "point": float(t_ben.mean() - b_ben.mean()),
                "ci95_bootstrap_over_prompts": orr_diff_ci, "n_boot": args.n_boot,
                "mcnemar": orr_mcnemar},
            "n_attack_prompts": len(atk_ids), "n_benign_prompts": len(ben_ids),
            "n_seeds_treatment": len(t_seeds), "n_seeds_baseline": len(b_seeds),
        }

        # ---------------- headline sentence --------------------------------------------
        n_seeds = len(t_seeds)
        if n_seeds != len(b_seeds):
            blockers.append(f"{T} has {len(t_seeds)} seeds but {B} has {len(b_seeds)}")
        if n_seeds < 2:
            blockers.append(f"only {n_seeds} seed(s) available; CLAUDE.md requires >= 2 on core arms")
        for name, val in (("baseline ASR", primary["baseline_asr"]),
                          ("treatment ASR", primary["treatment_asr"]),
                          ("over-refusal CI", orr_diff_ci[1])):
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                blockers.append(f"{name} is not a finite number")

        if not blockers:
            X = primary["baseline_asr"] * 100
            Y = primary["treatment_asr"] * 100
            # "at most Z points" is an UPPER BOUND, so take the more conservative of the two
            # available estimates of the over-refusal increase:
            #   (a) bootstrap upper bound on the per-prompt majority-vote difference, and
            #   (b) upper bound of the across-seed mean difference.
            # These can differ substantially: majority voting across seeds cancels
            # independent per-seed noise, so (a) alone can understate an increase that the
            # per-seed means clearly show. Reporting the smaller of the two would overstate
            # the treatment's safety on the counter-metric.
            t_orr = report["table1"][T]["over_refusal"]
            b_orr = report["table1"][B]["over_refusal"]
            seed_diff = t_orr["mean"] - b_orr["mean"]
            seed_diff_hi = seed_diff
            if t_orr.get("ci95") and b_orr.get("ci95"):
                seed_diff_hi = t_orr["ci95"][1] - b_orr["ci95"][0]
            Z_boot = orr_diff_ci[1] * 100
            Z_seed = seed_diff_hi * 100
            Z = max(0.0, Z_boot, Z_seed)
            report["primary_test"]["over_refusal_upper_bounds"] = {
                "bootstrap_majority_vote_upper_pts": Z_boot,
                "across_seed_mean_difference_pts": seed_diff * 100,
                "across_seed_conservative_upper_pts": Z_seed,
                "reported_Z_pts": Z,
                "rule": "Z = max(0, bootstrap upper bound, across-seed conservative upper bound)",
            }
            report["headline_sentence"] = (
                f"{T.upper()} reduces ASR from {X:.1f}% to {Y:.1f}% relative to {B.upper()} "
                f"(n={n_seeds} seeds, {len(atk_ids)} paired prompts, 95% CI, "
                f"{primary['test']}, p={primary['p_value']:.4g}) while increasing "
                f"over-refusal by at most {Z:.1f} points.")
    report["headline_blockers"] = blockers

    # ---------------- print ---------------------------------------------------------------
    print("\n===== TABLE 1 " + "=" * 62)
    print(f"{'arm':<6}{'ASR':>26}{'over-refusal':>26}{'helpfulness':>22}")
    for arm in sorted(report["table1"]):
        t1 = report["table1"][arm]
        def fmt(d, scale=100, unit="%"):
            if d.get("mean") is None:
                return "NOT SCORED"
            if d.get("ci95") is None:
                return f"{d['mean'] * scale:.2f}{unit} (single run)"
            lo, hi = d["ci95"]
            return f"{d['mean'] * scale:.2f}{unit} [{lo * scale:.2f}, {hi * scale:.2f}]"
        h = t1["helpfulness"]
        hs = "NOT SCORED" if h.get("mean") is None else (
            f"{h['mean']:.3f}" if h.get("ci95") is None
            else f"{h['mean']:.3f} [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}]")
        print(f"{arm:<6}{fmt(t1['asr']):>26}{fmt(t1['over_refusal']):>26}{hs:>22}")
    print("  (B0/B1 are single runs -- italicise in the paper; intervals are across-seed, "
          "t-based, and are a robustness check, not the primary n.)")

    print("\n===== TABLE 2: per-category ASR " + "=" * 44)
    header = f"{'arm':<6}" + "".join(f"{c:>20}" for c in ATTACK_CATEGORIES)
    print(header)
    for arm in sorted(report["table2"]):
        cells = []
        for c in ATTACK_CATEGORIES:
            d = report["table2"][arm].get(c)
            cells.append("--" if d is None else f"{d['mean'] * 100:.2f}%")
        print(f"{arm:<6}" + "".join(f"{x:>20}" for x in cells))

    print("\n===== PRIMARY TEST " + "=" * 57)
    if report["primary_test"] is None:
        print(f"  NOT COMPUTED: {'; '.join(blockers)}")
    else:
        p = report["primary_test"]["primary"]
        d = report["primary_test"]["asr_difference_treatment_minus_baseline"]
        print(f"  {report['primary_test']['contrast']}: {p['test']}")
        print(f"  n = {p['n_pairs']} paired prompts, {p['n_discordant']} discordant "
              f"(baseline-unsafe/treatment-safe = {p['baseline_unsafe_treatment_safe']}, "
              f"baseline-safe/treatment-unsafe = {p['baseline_safe_treatment_unsafe']})")
        print(f"  ASR {p['baseline_asr'] * 100:.2f}% -> {p['treatment_asr'] * 100:.2f}%   "
              f"difference {d['point'] * 100:+.2f} pts, 95% bootstrap CI "
              f"[{d['ci95_bootstrap_over_prompts'][0] * 100:+.2f}, "
              f"{d['ci95_bootstrap_over_prompts'][1] * 100:+.2f}]")
        print(f"  p = {p['p_value']:.4g}")
        for r in report["primary_test"]["per_seed_robustness"]:
            print(f"    [robustness] seed {r['seed']}: p = {r['p_value']:.4g} "
                  f"({r['n_discordant']} discordant)")

    print("\n===== HEADLINE SENTENCE " + "=" * 52)
    if report["headline_sentence"]:
        print("  " + report["headline_sentence"])
    else:
        print("  *** NOT EMITTED -- the harness refuses to print a partially filled sentence. ***")
        for b in blockers:
            print(f"    missing: {b}")

    out = resolve(args.out) if args.out else results_dir / "stats_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
