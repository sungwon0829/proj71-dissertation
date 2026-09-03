"""
Analysis 3 (Amendment 24.1) -- Retrospective minimum detectable effect. POST-HOC EXPLORATORY.

WHICH PAPER NUMBER: none (post-hoc exploratory, Amendment 24.1; never Table 1/2, never the
headline). Reports the MDE of McNemar's exact test at 80% power given the OBSERVED
discordant structure of the seed-1 T-vs-B3 primary comparison (n=180), beside the
pre-registered 10-point threshold, plus power at the observed effect.

Method
------
1. Read canonical seed-1 scored files (results/b3_seed42/scored.jsonl,
   results/t_seed42/scored.jsonl), restrict to the primary attack categories
   {prefilling, persona, many_shot} (n=180; crisis is a separate endpoint, never pooled),
   pair by prompt id, and VERIFY the observed structure: B3 45.00%, T 39.44%,
   56 discordant pairs. Fail loudly on any mismatch.
2. McNemar's exact test, conditional on the number of discordant pairs n_d:
   under H0 the count of B3-only-unsafe pairs C ~ Binomial(n_d, 1/2); two-sided
   p-value = min(1, 2*min(P(X<=c), P(X>=c))) (equivalently 2*P(X<=min(b,c)) capped at 1).
   The alpha=.05 rejection region over c in {0..n_d} is enumerated exactly.
3. Power, exact enumeration (primary method): for a raw marginal-rate difference of
   delta percentage points with the discordant rate HELD AT THE OBSERVED VALUE
   n_d/n = 56/180, the discordant split is p = 1/2 + (delta*n)/(2*n_d); power =
   sum over c in the rejection region of Binomial(n_d, p).pmf(c). Fully deterministic.
4. MDE = smallest delta with power >= 0.80, found by bisection (deterministic).
   Note the constraint delta <= n_d/n = 31.1 pts: with the discordant rate fixed, no
   larger raw difference is arithmetically possible.
5. Cross-check, unconditional simulation (--seed, default 0): each of the 180 pairs is
   discordant with probability 56/180; a discordant pair is B3-only-unsafe with
   probability p; the exact test is applied to each realised (b, c). Confirms the
   conditional enumeration is not an artefact of conditioning on n_d = 56.

Scale note: all effects are on the RAW OBSERVED scale (Amendment 22.3 -- attenuation
correction NOT APPLIED), the same scale as the primary reported effect (-5.56 pts).

Usage: python compute_mde.py [--seed 0] [--n-sim 100000]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

REPO = Path(r"C:\proj71")
OUTDIR = REPO / "results" / "exploratory" / "analysis3_retrospective_mde"
PRIMARY_CATEGORIES = {"prefilling", "persona", "many_shot"}
ALPHA = 0.05
TARGET_POWER = 0.80
PREREG_THRESHOLD_PTS = 10.0

# Observed structure this analysis is conditioned on (verified against the files, not trusted).
EXPECTED_N = 180
EXPECTED_B3_UNSAFE = 81   # 45.00%
EXPECTED_T_UNSAFE = 71    # 39.44%
EXPECTED_DISCORDANT = 56


def load_primary_verdicts(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") != "scored":
                continue
            if rec.get("category") not in PRIMARY_CATEGORIES:
                continue
            v = rec.get("verdict")
            if v not in ("safe", "unsafe"):
                raise ValueError(f"{path}: id={rec.get('id')} has verdict {v!r}")
            out[rec["id"]] = (v == "unsafe")
    return out


def mcnemar_exact_p(c: int, n_d: int) -> float:
    """Two-sided exact McNemar p conditional on n_d discordant pairs."""
    lo = binom.cdf(c, n_d, 0.5)
    hi = binom.sf(c - 1, n_d, 0.5)
    return min(1.0, 2.0 * min(lo, hi))


def rejection_region(n_d: int, alpha: float) -> np.ndarray:
    return np.array([c for c in range(n_d + 1) if mcnemar_exact_p(c, n_d) <= alpha])


def power_conditional(p: float, n_d: int, region: np.ndarray) -> float:
    return float(binom.pmf(region, n_d, p).sum())


def delta_to_p(delta_pts: float, n: int, n_d: int) -> float:
    """Raw marginal difference (pts) -> discordant split, discordant rate held at n_d/n."""
    return 0.5 + (delta_pts / 100.0) * n / (2.0 * n_d)


def mde_bisect(n: int, n_d: int, region: np.ndarray,
               target: float, tol: float = 1e-4) -> float:
    lo, hi = 0.0, 100.0 * n_d / n  # arithmetic maximum given fixed discordant rate
    if power_conditional(delta_to_p(hi, n, n_d), n_d, region) < target:
        raise RuntimeError("target power unreachable even at the arithmetic maximum delta")
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if power_conditional(delta_to_p(mid, n, n_d), n_d, region) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def power_unconditional_sim(delta_pts: float, n: int, pd: float,
                            n_sim: int, seed: int) -> float:
    """Cross-check: n_d ~ Bin(n, pd); C|n_d ~ Bin(n_d, p); exact test on each draw."""
    rng = np.random.default_rng(seed)
    p = delta_to_p(delta_pts, n, n * pd)  # same split as conditional at expected n_d
    nds = rng.binomial(n, pd, size=n_sim)
    cs = rng.binomial(nds, p)
    # p-values vectorised via per-n_d critical sets would be faster; direct is fine at 1e5.
    rejects = np.fromiter(
        (mcnemar_exact_p(int(c), int(nd)) <= ALPHA if nd > 0 else False
         for c, nd in zip(cs, nds)),
        dtype=bool, count=n_sim)
    return float(rejects.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sim", type=int, default=100_000)
    args = ap.parse_args()

    b3 = load_primary_verdicts(REPO / "results" / "b3_seed42" / "scored.jsonl")
    t = load_primary_verdicts(REPO / "results" / "t_seed42" / "scored.jsonl")

    if set(b3) != set(t):
        raise RuntimeError("id sets differ between b3 and t scored files")
    ids = sorted(b3)
    n = len(ids)

    both_unsafe = sum(1 for i in ids if b3[i] and t[i])
    both_safe = sum(1 for i in ids if not b3[i] and not t[i])
    b = sum(1 for i in ids if t[i] and not b3[i])   # T-only-unsafe
    c = sum(1 for i in ids if b3[i] and not t[i])   # B3-only-unsafe
    n_d = b + c
    b3_rate = (both_unsafe + c) / n
    t_rate = (both_unsafe + b) / n

    # Verify the observed structure this analysis was specified against.
    assert n == EXPECTED_N, (n, EXPECTED_N)
    assert both_unsafe + c == EXPECTED_B3_UNSAFE, (both_unsafe + c, EXPECTED_B3_UNSAFE)
    assert both_unsafe + b == EXPECTED_T_UNSAFE, (both_unsafe + b, EXPECTED_T_UNSAFE)
    assert n_d == EXPECTED_DISCORDANT, (n_d, EXPECTED_DISCORDANT)

    observed_delta_pts = 100.0 * (t_rate - b3_rate)          # -5.56
    observed_p = mcnemar_exact_p(c, n_d)

    region = rejection_region(n_d, ALPHA)
    # Rejection region in terms of c (B3-only-unsafe count): {0..lo} U {hi..n_d}
    reg_lo = int(region[region < n_d / 2].max()) if (region < n_d / 2).any() else None
    reg_hi = int(region[region > n_d / 2].min()) if (region > n_d / 2).any() else None
    attained_size = power_conditional(0.5, n_d, region)

    mde_pts = mde_bisect(n, n_d, region, TARGET_POWER)
    p_at_mde = delta_to_p(mde_pts, n, n_d)

    power_at_observed = power_conditional(delta_to_p(abs(observed_delta_pts), n, n_d),
                                          n_d, region)
    power_at_prereg = power_conditional(delta_to_p(PREREG_THRESHOLD_PTS, n, n_d),
                                        n_d, region)

    sim_at_mde = power_unconditional_sim(mde_pts, n, n_d / n, args.n_sim, args.seed)
    sim_at_prereg = power_unconditional_sim(PREREG_THRESHOLD_PTS, n, n_d / n,
                                            args.n_sim, args.seed + 1)
    sim_at_observed = power_unconditional_sim(abs(observed_delta_pts), n, n_d / n,
                                              args.n_sim, args.seed + 2)

    results = {
        "analysis": "Amendment 24.1 analysis 3 -- retrospective MDE (POST-HOC EXPLORATORY)",
        "status": "post-hoc exploratory (Amendment 24.1); never confirmatory; "
                  "never Table 1/2; never the headline sentence",
        "date": "2026-08-28",
        "inputs": {
            "b3": str(REPO / "results" / "b3_seed42" / "scored.jsonl"),
            "t": str(REPO / "results" / "t_seed42" / "scored.jsonl"),
            "endpoint": "primary (prefilling, persona, many_shot); crisis excluded, "
                        "never pooled (Amendment 13)",
        },
        "observed_structure_verified": {
            "n_pairs": n,
            "b3_unsafe": both_unsafe + c, "b3_asr_pct": round(100 * b3_rate, 2),
            "t_unsafe": both_unsafe + b, "t_asr_pct": round(100 * t_rate, 2),
            "both_unsafe": both_unsafe, "both_safe": both_safe,
            "t_only_unsafe_b": b, "b3_only_unsafe_c": c,
            "n_discordant": n_d,
            "discordant_rate": round(n_d / n, 4),
            "observed_delta_pts": round(observed_delta_pts, 2),
            "observed_exact_mcnemar_p": round(observed_p, 4),
        },
        "test": {
            "name": "McNemar's exact test (two-sided), conditional binomial over "
                    "discordant pairs; p = min(1, 2*min(P(X<=c), P(X>=c))), "
                    "X ~ Binomial(n_d, 1/2)",
            "alpha": ALPHA,
            "rejection_region_c": {"c_le": reg_lo, "c_ge": reg_hi},
            "attained_size": round(attained_size, 4),
        },
        "method": {
            "power": "exact enumeration of Binomial(n_d=56, p) over the rejection region; "
                     "discordant rate held at observed 56/180; raw marginal difference "
                     "delta maps to split p = 1/2 + delta*n/(2*n_d)",
            "mde": "bisection to smallest delta with power >= 0.80 (tol 1e-4 pts); "
                   "deterministic (no RNG in the primary method)",
            "cross_check": f"unconditional simulation, n_d ~ Bin(180, 56/180), "
                           f"n_sim={args.n_sim}, seed={args.seed}",
            "scale": "raw observed scale (Amendment 22.3: attenuation correction NOT APPLIED)",
            "arithmetic_ceiling_pts": round(100 * n_d / n, 2),
        },
        "results": {
            "mde_pts_80pct_power": round(mde_pts, 2),
            "discordant_split_at_mde": round(p_at_mde, 4),
            "power_at_prereg_10pts": round(power_at_prereg, 4),
            "power_at_observed_5.56pts": round(power_at_observed, 4),
            "prereg_threshold_pts": PREREG_THRESHOLD_PTS,
            "design_detects_10pt_at_80pct_power": bool(power_at_prereg >= TARGET_POWER),
        },
        "cross_check_unconditional_sim": {
            "seed": args.seed, "n_sim": args.n_sim,
            "power_at_mde": round(sim_at_mde, 4),
            "power_at_prereg_10pts": round(sim_at_prereg, 4),
            "power_at_observed_5.56pts": round(sim_at_observed, 4),
        },
        "small_cell_rule": "any denominator < 10 gets counts only, no percentage; "
                           "all denominators here (56, 180) are >= 10",
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
