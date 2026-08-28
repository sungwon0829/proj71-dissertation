"""Amendment 24.2 dose-response exhibit: safety-pair ratio vs ASR, EXPLORATORY.

Pre-specified output (preregistration_amendments.md 24.2, written before any result
existed): ASR primary and crisis co-primary per achieved ratio (0%, 50%, 100%, and the
high arm at its ACHIEVED ratio -- the pool contingency fired at launch: the filtered
safety pool is 4,924 rows, all of which T already uses, so T_r200's achieved ratio is
100% and it is an expected duplicate of T), with prompt-set bootstrap CIs, ordered by
ratio, plus an explicit statement of whether the ordering is monotone. Reported
whatever the pattern is. Never Table 1/2, never the headline.

Method notes:
  - b2/t points and CIs are REUSED VERBATIM from results/tables_final/tables_final.json
    (per_arm_primary_asr_ci_NEW / per_arm_crisis_ci_NEW) where they exist; anything not
    precomputed there (b2 crisis CI, both t_r050/t_r200 CIs) is computed HERE with the
    identical method: stats.bootstrap_paired + stats.pct_ci, n_boot=10000, seed 0.
  - The t_r200 replication claim is asserted, not assumed: its adapter bytes must hash
    identically to T's, its data manifest must differ from T's only in labels, and its
    per-item outcome vectors must equal T's exactly. Any deviation is a hard failure.

Outputs: results/exploratory/ratio_ablation/{dose_response.json, report.md}
Run:  C:\\proj71\\env\\Scripts\\python.exe scripts\\make_ratio_ablation_exhibit.py
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from stats import (load_scored, outcome_vectors, bootstrap_paired, pct_ci,
                   PRIMARY_ATTACK_CATEGORIES, CRISIS_CATEGORY)  # noqa: E402

FROZEN_SUITE_SHA = "e14c3a24184d01cbf31bbcfa42be03104ae07b0bea1132bc5b08a177645b6689"
T_SAFETY_POOL = 4924  # full filtered PKU pool; T's own n_safety_sampled

SCORED = {
    "b2": REPO / "results" / "b2_seed42" / "scored.jsonl",
    "t": REPO / "results" / "t_seed42" / "scored.jsonl",
    "t_r050": REPO / "results" / "t_r050_seed42" / "scored_realsuite.jsonl",
    "t_r200": REPO / "results" / "t_r200_seed42" / "scored_realsuite.jsonl",
}
TRAIN_DIRS = {
    "t": REPO / "results" / "T_dpo_seed1",
    "t_r050": REPO / "results" / "T_r050_dpo_seed1",
    "t_r200": REPO / "results" / "T_r200_dpo_seed1",
}
LABEL_ONLY_MANIFEST_FIELDS = {"arm", "cli_invocation", "config"}

FAILURES = []


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}: got {got!r} want {want!r}")
    if not ok:
        FAILURES.append((name, got, want))
    return ok


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (matches tables_final)")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    tf = json.loads((REPO / "results" / "tables_final" / "tables_final.json")
                    .read_text(encoding="utf-8"))

    data, headers = {}, {}
    for arm, path in SCORED.items():
        if not path.is_file():
            print(f"FATAL: {path} missing -- scoring not finished?", file=sys.stderr)
            sys.exit(1)
        header, rows = load_scored(path)
        assert header["arm"].lower() == arm, f"{path}: header arm {header['arm']!r} != {arm!r}"
        assert header["suite_sha256"] == FROZEN_SUITE_SHA, f"{path}: suite sha mismatch"
        data[arm], headers[arm] = rows, header
        print(f"loaded {arm}: {len(rows)} rows from {path}")

    ref = data["t"]
    atk_ids = sorted(i for i, r in ref.items() if r["category"] in PRIMARY_ATTACK_CATEGORIES)
    crisis_ids = sorted(i for i, r in ref.items() if r["category"] == CRISIS_CATEGORY)
    assert len(atk_ids) == 180 and len(crisis_ids) == 60
    vec = {arm: {"atk": outcome_vectors(rows, atk_ids, "unsafe").astype(float),
                 "crisis": outcome_vectors(rows, crisis_ids, "unsafe").astype(float)}
           for arm, rows in data.items()}

    # ---- verification pass (fail loud, nothing written on failure) ---------------
    print("\nVERIFICATION")
    tf_asr = tf["per_arm_primary_asr_ci_NEW"]
    tf_crisis = tf["per_arm_crisis_ci_NEW"]
    check("b2 primary ASR == tables_final", float(vec["b2"]["atk"].mean()), tf_asr["b2"]["asr"])
    check("t primary ASR == tables_final", float(vec["t"]["atk"].mean()), tf_asr["t"]["asr"])
    check("t crisis rate == tables_final", float(vec["t"]["crisis"].mean()), tf_crisis["t"]["rate"])

    # t_r200 replication: adapter bytes, manifest labels-only diff, outcome vectors
    a_t = sha256_file(TRAIN_DIRS["t"] / "adapter_model.safetensors")
    a_r200 = sha256_file(TRAIN_DIRS["t_r200"] / "adapter_model.safetensors")
    check("t_r200 adapter bit-identical to T", a_r200, a_t)
    man_t = json.loads((TRAIN_DIRS["t"] / "dpo_data_manifest.json").read_text(encoding="utf-8"))
    man_r = json.loads((TRAIN_DIRS["t_r200"] / "dpo_data_manifest.json").read_text(encoding="utf-8"))
    diff_fields = sorted(k for k in set(man_t) | set(man_r) if man_t.get(k) != man_r.get(k))
    check("t_r200 manifest differs from T only in labels", diff_fields,
          sorted(LABEL_ONLY_MANIFEST_FIELDS & set(diff_fields))
          if set(diff_fields) <= LABEL_ONLY_MANIFEST_FIELDS else diff_fields + ["<UNEXPECTED>"])
    check("t_r200 primary outcome vector == t (exact, 180 items)",
          int((vec["t_r200"]["atk"] != vec["t"]["atk"]).sum()), 0)
    check("t_r200 crisis outcome vector == t (exact, 60 items)",
          int((vec["t_r200"]["crisis"] != vec["t"]["crisis"]).sum()), 0)

    ratios = {}
    for arm, d in TRAIN_DIRS.items():
        man = json.loads((d / "dpo_data_manifest.json").read_text(encoding="utf-8"))
        assert man["n_consumed"] == 19924, f"{arm}: matched volume violated"
        ratios[arm] = man["n_safety_sampled"] / T_SAFETY_POOL
    check("t_r050 achieved ratio", ratios["t_r050"], 2462 / 4924)
    check("t_r200 achieved ratio (pool contingency)", ratios["t_r200"], 1.0)

    if FAILURES:
        print(f"\nSTOP: {len(FAILURES)} verification failure(s). NO OUTPUT WRITTEN.")
        sys.exit(1)
    print("verification: ALL EXACT\n")

    # ---- dose-response points, ordered by achieved ratio -------------------------
    def ci_for(arm, kind):
        """Reuse tables_final CIs verbatim where they exist; compute otherwise."""
        pre = tf_asr if kind == "atk" else tf_crisis
        if arm in pre:
            src = pre[arm]
            return (src["asr"] if kind == "atk" else src["rate"],
                    src["ci95_bootstrap_over_prompts"], "tables_final_verbatim")
        v = vec[arm][kind]
        b = bootstrap_paired({"x": v}, args.n_boot, args.seed, lambda d: d["x"].mean())
        return float(v.mean()), list(pct_ci(b)), "computed_here_same_method"

    points = []
    for arm, ratio, role in [("b2", 0.0, "0% safety pairs (helpfulness-only DPO)"),
                             ("t_r050", ratios["t_r050"], "50% of T's safety pairs"),
                             ("t", 1.0, "100% -- the pre-registered T arm"),
                             ("t_r200", ratios["t_r200"],
                              "requested 200%; ACHIEVED 100% (pool contingency) -- "
                              "exact replication of T")]:
        asr, asr_ci, asr_src = ci_for(arm, "atk")
        cr, cr_ci, cr_src = ci_for(arm, "crisis")
        points.append({
            "arm": arm, "achieved_ratio": ratio, "role": role,
            "n_safety_pairs": int(round(ratio * T_SAFETY_POOL)),
            "asr_primary": {"value": asr, "ci95_bootstrap_over_prompts": asr_ci,
                            "n_prompts": 180, "source": asr_src},
            "crisis_co_primary": {"value": cr, "ci95_bootstrap_over_prompts": cr_ci,
                                  "n_prompts": 60, "source": cr_src},
        })

    # monotonicity over the DISTINCT achieved ratios (0, 0.5, 1.0); t_r200 is a
    # replication of the 1.0 point, not a fourth ratio
    def pattern(values):
        d = np.diff(values)
        if np.all(d < 0):
            return "strictly monotone decreasing"
        if np.all(d <= 0):
            return "monotone non-increasing (with ties)"
        if np.all(d > 0):
            return "strictly monotone increasing"
        if np.all(d >= 0):
            return "monotone non-decreasing (with ties)"
        return "NON-MONOTONE"

    distinct = [p for p in points if p["arm"] in ("b2", "t_r050", "t")]
    asr_vals = [p["asr_primary"]["value"] for p in distinct]
    crisis_vals = [p["crisis_co_primary"]["value"] for p in distinct]
    monotonicity = {
        "ratios": [p["achieved_ratio"] for p in distinct],
        "asr_primary_values": asr_vals,
        "asr_primary_pattern": pattern(asr_vals),
        "crisis_values": crisis_vals,
        "crisis_pattern": pattern(crisis_vals),
        "note": ("Assessed over the three distinct achieved ratios (0, 0.5, 1.0); "
                 "t_r200 replicates the 1.0 point exactly (bit-identical adapter) and "
                 "is not an independent ratio. Three points cannot establish a "
                 "dose-response curve; the pattern statement is descriptive "
                 "(Amendment 24.2), with no trend test and no reframing of the "
                 "primary null."),
    }

    out = {
        "record_type": "ratio_ablation_dose_response",
        "amendment": "24.2 (post-hoc exploratory, sighted; pre-specified output)",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/make_ratio_ablation_exhibit.py",
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed,
                      "method": "stats.bootstrap_paired + stats.pct_ci, identical to tables_final"},
        "suite_sha256": FROZEN_SUITE_SHA,
        "replication_check_t_r200": {
            "adapter_sha256": a_r200, "adapter_matches_t": True,
            "manifest_fields_differing_from_t": diff_fields,
            "outcome_vectors_identical_to_t": True,
            "training_determinism_note": (
                "Same config totals + same seed reproduced T's adapter BIT-FOR-BIT "
                "(sha256 equal): LoRA DPO training on this stack (Windows Server 2025, "
                "Blackwell, PyTorch 2.11 sdpa) is bitwise deterministic given identical "
                "data and seed. Incidental but paper-worthy reproducibility evidence."),
        },
        "points_ordered_by_ratio": points,
        "monotonicity": monotonicity,
        "scored_files": {arm: str(SCORED[arm].relative_to(REPO)).replace("\\", "/")
                         for arm in SCORED},
        "scored_sha256": {arm: sha256_file(SCORED[arm]) for arm in SCORED},
    }

    outdir = REPO / "results" / "exploratory" / "ratio_ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "dose_response.json").write_text(json.dumps(out, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")

    md = ["# Safety-pair ratio ablation -- dose-response (Amendment 24.2, EXPLORATORY)",
          "",
          "Post-hoc exploratory, sighted; output pre-specified in the amendment before any",
          "result existed. Never Table 1/2, never the headline. Seed 1 only.",
          "",
          "| Achieved ratio | Arm | n safety pairs | ASR primary (n=180) | 95% CI | Crisis rate (n=60) | 95% CI |",
          "|---|---|---|---|---|---|---|"]
    for p in points:
        a, c = p["asr_primary"], p["crisis_co_primary"]
        md.append(f"| {p['achieved_ratio']:.0%} | {p['arm']} | {p['n_safety_pairs']} | "
                  f"{a['value']:.2%} | [{a['ci95_bootstrap_over_prompts'][0]:.2%}, "
                  f"{a['ci95_bootstrap_over_prompts'][1]:.2%}] | {c['value']:.2%} | "
                  f"[{c['ci95_bootstrap_over_prompts'][0]:.2%}, "
                  f"{c['ci95_bootstrap_over_prompts'][1]:.2%}] |")
    md += ["",
           f"**Monotonicity (pre-committed reading).** ASR primary over ratios 0 / 0.5 / 1.0: "
           f"{' -> '.join(f'{v:.2%}' for v in asr_vals)} -- **{monotonicity['asr_primary_pattern']}**. "
           f"Crisis co-primary: {' -> '.join(f'{v:.2%}' for v in crisis_vals)} -- "
           f"**{monotonicity['crisis_pattern']}**.",
           "",
           monotonicity["note"],
           "",
           "**T_r200 replication check.** " +
           out["replication_check_t_r200"]["training_determinism_note"],
           "",
           "CIs: 95% percentile bootstrap over the prompt set (n_boot=10000, seed 0), the",
           "same method and parameters as tables_final; b2/t values reused verbatim from",
           "tables_final where precomputed, remainder computed here identically."]
    (outdir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"[written] {outdir / 'dose_response.json'}")
    print(f"[written] {outdir / 'report.md'}")
    print("\n" + "\n".join(md[4:]))


if __name__ == "__main__":
    main()
