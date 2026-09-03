"""Results manifest: every paper exhibit -> artifact -> config -> seed -> hash -> commit.

Produces the pre-lock closure-3 deliverable (owner directive 2026-08-28): a
machine-readable map from every table cell / statistic / figure the paper reports to
the artifact it comes from, the config and seed that produced that artifact, the
SHA-256 fixing its bytes, and the git commit identifying the code+config tree it was
produced and verified under -- plus a human-readable one-page version for the appendix.

Everything is read LIVE from the artifacts at generation time: values in the manifest
are pulled from the same files a reviewer would open, so the manifest cannot drift
from the artifacts without the drift being visible on regeneration. Missing files or
fields are hard failures, never silently skipped (CLAUDE.md conventions).

Outputs (committed -- results/ itself is never committed, which is exactly why the
manifest lives in notebook/):
  notebook/results_manifest.json
  notebook/results_manifest.md

Run:  C:\\proj71\\env\\Scripts\\python.exe scripts\\make_results_manifest.py
Deterministic: pure read of existing artifacts; no RNG, no --seed required.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FROZEN_SUITE_SHA = "e14c3a24184d01cbf31bbcfa42be03104ae07b0bea1132bc5b08a177645b6689"


def die(msg: str):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")


def must(p) -> Path:
    p = Path(p)
    p = p if p.is_absolute() else REPO / p
    if not p.is_file():
        die(f"required artifact missing: {p}")
    return p


def jload(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsonl_header(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.loads(f.readline())


def git(*args) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def main():
    git_head = git("rev-parse", "HEAD")
    git_branch = git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l.strip() and not l.endswith("claude.md")]

    sources: dict[str, dict] = {}

    def add_source(sid: str, path, role: str, extra: dict | None = None) -> str:
        p = must(path)
        if sid in sources:
            return sid
        sources[sid] = {"path": rel(p), "sha256": sha256_file(p), "role": role,
                        **(extra or {})}
        return sid

    # ---- pins -----------------------------------------------------------------
    suite = must("data/redteam/redteam_suite.jsonl")
    suite_sha = sha256_file(suite)
    if suite_sha != FROZEN_SUITE_SHA:
        die(f"frozen suite hash mismatch: {suite_sha} != {FROZEN_SUITE_SHA}")
    # the current lock is resolved through judges.yaml's pin pointer, exactly as
    # verify_judge_pin() resolves it, so a legitimate re-pin auto-tracks here; the
    # two-file tandem (pointer + sha) is still cross-checked
    yaml_text = must("configs/judges.yaml").read_text(encoding="utf-8")
    m_file = re.search(r"pin_lock_file:\s*(\S+)", yaml_text)
    m_sha = re.search(r'pin_lock_sha256:\s*"([0-9a-f]{64})"', yaml_text)
    if not (m_file and m_sha):
        die("judges.yaml pin pointer (pin_lock_file / pin_lock_sha256) not found")
    lock = must(m_file.group(1))
    lock_sha = sha256_file(lock)
    if lock_sha != m_sha.group(1):
        die(f"lock {lock} hashes to {lock_sha} but judges.yaml records {m_sha.group(1)}")
    pins = {
        "frozen_suite": {"path": rel(suite), "sha256": suite_sha},
        "judges_lock_current": {"path": rel(lock), "sha256": lock_sha},
        "judges_yaml": {"path": "configs/judges.yaml",
                        "sha256": sha256_file(must("configs/judges.yaml")),
                        "note": "recorded provenance only; the ENFORCED pin is the lock file"},
        "preregistration": {"path": "notebook/preregistration.md",
                            "sha256": sha256_file(must("notebook/preregistration.md"))},
        "preregistration_amendments": {"path": "notebook/preregistration_amendments.md",
                                       "sha256": sha256_file(must("notebook/preregistration_amendments.md"))},
    }

    # ---- per-arm provenance chains -------------------------------------------
    # scored file -> generations file -> (adapter, configs, seeds).  All pulled from
    # the artifact headers eval_generate/eval_score wrote at run time.
    def arm_chain(arm: str, scored_path: str, train_config: str | None,
                  train_seed: int | None, status: str) -> str:
        sp = must(scored_path)
        sh = jsonl_header(sp)
        if sh.get("arm") != arm:
            die(f"{sp}: header arm {sh.get('arm')!r} != expected {arm!r}")
        if sh.get("suite_sha256") != FROZEN_SUITE_SHA:
            die(f"{sp}: suite sha {sh.get('suite_sha256')} != frozen pin")
        gen = must(sh["generations_file"])
        gen_sha = sha256_file(gen)
        if gen_sha != sh["generations_sha256"]:
            die(f"{gen}: bytes {gen_sha} != scored-header record {sh['generations_sha256']}")
        gh = jsonl_header(gen)
        extra = {
            "arm": arm, "status": status,
            "generation_seed": sh.get("seed"),
            "generations_file": rel(gen), "generations_sha256": gen_sha,
            "generation_config": gh.get("config_file"),
            "generation_config_sha256": gh.get("config_sha256"),
            "base_model": {"name": gh["model"].get("name_or_path"),
                           "revision": gh["model"].get("revision")},
            "adapter": gh.get("adapter"),
            "judges_config_sha256": sh.get("judges_config_sha256"),
            "is_paper_number": sh.get("is_paper_number"),
        }
        if train_config is not None:
            tc = must(train_config)
            extra["train_config"] = rel(tc)
            extra["train_config_sha256"] = sha256_file(tc)
            extra["train_seed"] = train_seed
        return add_source(f"scored_{arm}", sp, "scored generations (judge verdicts)", extra)

    arm_specs = {
        "b0": ("results/b0_seed42/scored_realsuite.jsonl", None, None, "context_single_run"),
        "b1": ("results/b1_seed42/scored_realsuite.jsonl", "configs/sft_lora.yaml", 42, "context_single_run"),
        "b2": ("results/b2_seed42/scored.jsonl", "configs/dpo_b2.yaml", 1, "headline_adjacent"),
        "b3": ("results/b3_seed42/scored.jsonl", "configs/dpo_b2.yaml", 1, "headline_baseline"),
        "t":  ("results/t_seed42/scored.jsonl", "configs/dpo_t.yaml", 1, "headline_treatment"),
        "t_ctrl": ("results/t_ctrl_seed42/scored_realsuite.jsonl", "configs/dpo_t_ctrl.yaml", 1,
                   "weak_control_single_run"),
    }
    for arm, (sp, tc, ts, status) in arm_specs.items():
        arm_chain(arm, sp, tc, ts, status)

    # hand-label instruments behind the Table 1 over-refusal cells (Revision 4)
    add_source("labels_b3", "results/human_validation/over_refusal_labels_b3_from_merged.json",
               "hand labels: over-refusal, arm b3 (Revision 4 primary instrument)")
    add_source("labels_t", "results/human_validation/over_refusal_labels_t_from_merged.json",
               "hand labels: over-refusal, arm t (Revision 4 primary instrument)")

    # ---- exhibits -------------------------------------------------------------
    exhibits: list[dict] = []

    def ex(eid: str, kind: str, status: str, value, source_ids: list[str],
           artifact: Path, notes: str | None = None):
        exhibits.append({
            "id": eid, "kind": kind, "status": status, "value": value,
            "artifact": rel(artifact), "artifact_sha256": sha256_file(artifact),
            "sources": source_ids, **({"notes": notes} if notes else {}),
        })

    tf_path = must("results/tables_final/tables_final.json")
    tf = jload(tf_path)
    add_source("tables_final", tf_path, "final Tables 1-2 artifact (make_tables_final.py, "
               "reproduces the primary stats report bit-exact before deriving)")
    rep_path = must("results/stats_report_realsuite_handlabelled.json")
    rep = jload(rep_path)
    add_source("primary_report", rep_path,
               "authoritative primary stats report (stats.py, hand-labelled over-refusal)")

    # Table 1 cells
    for arm in ["b0", "b1", "b2", "b3", "t"]:
        row = tf["table1"][arm]
        st = ("context" if row["single_run_context"] else
              "primary" if arm in ("b3", "t") else "context")
        ex(f"table1.{arm}.asr_primary_pct", "table_cell", st,
           {"pct": row["asr_primary_pct"], "ci95_pct": row["asr_ci95_over_prompts_pct"]},
           ["tables_final", f"scored_{arm}"], tf_path)
        ex(f"table1.{arm}.over_refusal_pct", "table_cell", st,
           {"pct": row["over_refusal_pct"], "instrument": row["over_refusal_source"]},
           ["tables_final", f"scored_{arm}"] +
           (["labels_b3"] if arm in ("b2", "b3") else ["labels_t"] if arm == "t" else []),
           tf_path,
           notes=None if row["over_refusal_source"] == "hand_label" else
           "rubric-judge value (kappa 0.074); dagger in the paper, never pooled with hand labels")
        ex(f"table1.{arm}.helpfulness_mean_reward", "table_cell", st,
           row["helpfulness_mean_reward"], ["tables_final", f"scored_{arm}"], tf_path)

    # T_ctrl row (single run, italics, no CI -- Amendment 9)
    tctrl_sum = must("results/t_ctrl_seed42/summary_realsuite.json")
    ts = jload(tctrl_sum)
    add_source("summary_t_ctrl", tctrl_sum, "t_ctrl summary (single-run weak control)")
    m = ts["metrics"]
    ex("table1.t_ctrl.row", "table_cell", "weak_control",
       {"asr_primary_pct": 100 * m["asr_primary_non_crisis"]["value"],
        "crisis_co_primary_pct": 100 * m["crisis_co_primary"]["value"],
        "over_refusal_pct": 100 * m["over_refusal"]["value"],
        "over_refusal_instrument": m["over_refusal"]["source"],
        "helpfulness_mean_reward": m["helpfulness"]["value"]},
       ["summary_t_ctrl", "scored_t_ctrl"], tctrl_sum,
       notes="Revision 5 weak control: single run, italics, no CI, no significance test "
             "(Amendment 9); never a headline arm. Over-refusal is the rubric judge "
             "(kappa 0.074) -- dagger, never pooled with hand-labelled cells")

    # Table 2 cells + decomposition
    for cat in ["prefilling", "persona", "many_shot", "crisis_adjacent"]:
        c = tf["table2"][cat]
        ex(f"table2.{cat}", "table_cell",
           "co_primary" if cat == "crisis_adjacent" else "primary",
           {"b3": c["b3"], "t": c["t"], "diff_t_minus_b3": c["diff_t_minus_b3"],
            "ci95": c["ci95_bootstrap_over_prompts"], "n_prompts": c["n_prompts"],
            "n_boot": c["n_boot"], "bootstrap_seed": c["seed"]},
           ["tables_final", "scored_b3", "scored_t"], tf_path)
    ex("table2.decomposition.b3", "table_cell", "primary",
       tf["table2_decomposition"]["b3"]["per_category"],
       ["tables_final", "scored_b3"], tf_path,
       notes="filter-attributable vs model-generated split; the filter fired 2/300")
    ex("table2.decomposition.t", "table_cell", "primary",
       tf["table2_decomposition"]["t"]["per_category"],
       ["tables_final", "scored_t"], tf_path,
       notes="T has no guardrail filter, so every failure is model-generated and the "
             "decomposition is trivial; reported for symmetry with B3")

    # Primary + co-primary statistics (hard key access: a missing key is a loud failure)
    pt = rep["primary_test"]
    prim = pt["primary"]
    diff = pt["asr_difference_treatment_minus_baseline"]
    ex("stats.primary", "statistic", "primary",
       {"test": prim["test"], "n_pairs": prim["n_pairs"],
        "n_nonzero_differences": prim["n_nonzero_differences"],
        "baseline_asr": prim["baseline_asr"], "treatment_asr": prim["treatment_asr"],
        "mean_difference": prim["mean_difference"], "p_value": prim["p_value"],
        "n_permutations": prim["n_permutations"],
        "permutation_seed": prim["permutation_seed"],
        "diff_point": diff["point"], "diff_ci95": diff["ci95_bootstrap_over_prompts"],
        "n_boot": diff["n_boot"],
        "attenuation_correction": pt["attenuation_correction"]["correction_status"]},
       ["primary_report", "scored_b3", "scored_t"], rep_path,
       notes="two-sided paired sign-flip permutation test, n=180 (Amendment 13); "
             "raw observed scale -- attenuation correction pending, never compare the "
             "raw diff against the 10-pt TRUE-scale threshold directly")
    ex("stats.co_primary_crisis", "statistic", "co_primary", pt["crisis_co_primary"],
       ["primary_report", "scored_b3", "scored_t"], rep_path,
       notes="crisis_adjacent n=60, separate judge, refusal-is-failure semantics; "
             "never pooled (Amendment 13)")
    ex("stats.over_refusal_bound", "statistic", "primary",
       {"difference": pt["over_refusal_difference_treatment_minus_baseline"]["point"],
        "ci95": pt["over_refusal_difference_treatment_minus_baseline"][
            "ci95_bootstrap_over_prompts"],
        "criterion": pt["over_refusal_criterion"]["criterion"],
        "criterion_met": pt["over_refusal_criterion"]["criterion_met"],
        "reporting_stance": pt["over_refusal_criterion"]["reporting_stance"]},
       ["primary_report", "labels_b3", "labels_t"], rep_path,
       notes="Revision 7: descriptive bound, NOT a passed test -- the design cannot "
             "resolve a 5-point over-refusal difference")
    ex("stats.tost_bound", "statistic", "primary", tf["tost_equivalence_NEW"],
       ["tables_final"], tf_path,
       notes="null-as-bound reading: TOST/equivalence interval on the raw scale")
    ex("stats.sign_convention", "statistic", "primary", tf["sign_convention"],
       ["tables_final"], tf_path,
       notes="explicit record of which arm is ahead at the reported difference")

    # headline sentence
    ex("stats.headline_sentence", "statistic", "primary",
       {"sentence": tf["headline_sentence_full"],
        "qualifier": tf["headline_sentence_qualifier"]},
       ["tables_final", "primary_report"], tf_path)

    # Amendment 23 robustness panel (sighted post-hoc seeds; never the primary n)
    ex("stats.robustness.panel", "statistic", "robustness",
       tf["posthoc_panel_amendment23"], ["tables_final", "posthoc_s2", "posthoc_s3"],
       tf_path, notes="assembled Amendment 23 panel as reported; raw per-seed reports "
                      "are the stats.robustness.seedN exhibits")
    for s in (2, 3):
        rp = must(f"results/stats_report_ts{s}_posthoc.json")
        r = jload(rp)
        add_source(f"posthoc_s{s}", rp, f"Amendment 23 post-hoc robustness report, training seed {s}")
        sp = r["primary_test"]["primary"]
        sd = r["primary_test"]["asr_difference_treatment_minus_baseline"]
        ex(f"stats.robustness.seed{s}", "statistic", "robustness",
           {"baseline_asr": sp["baseline_asr"], "treatment_asr": sp["treatment_asr"],
            "mean_difference": sp["mean_difference"], "p_value": sp["p_value"],
            "n_pairs": sp["n_pairs"],
            "n_nonzero_differences": sp["n_nonzero_differences"],
            "diff_ci95": sd["ci95_bootstrap_over_prompts"],
            "crisis_co_primary": {
                "difference": r["primary_test"]["crisis_co_primary"][
                    "difference_treatment_minus_baseline"],
                "p_value": r["primary_test"]["crisis_co_primary"]["test"]["p_value"]}},
           [f"posthoc_s{s}"], rp,
           notes="sighted post-hoc robustness (Amendment 23); primary analysis remains "
                 "training seed 1 exactly as pre-registered")

    # Figure: safety-helpfulness frontier (pin the WHOLE directory, including the
    # producing script and its report, same as the other exploratory analyses)
    frontier_ids = []
    for f in ["frontier.png", "frontier.pdf", "frontier_data.json",
              "plot_frontier.py", "report.md"]:
        sid = f"frontier_{f.replace('.', '_')}"
        add_source(sid, f"results/exploratory/frontier_figure/{f}",
                   "frontier figure artifact (exploratory analysis 6, Amendment 24)")
        frontier_ids.append(sid)
    ex("figure.frontier", "figure", "exploratory",
       "safety-helpfulness frontier, all six arms",
       frontier_ids + ["tables_final"],
       must("results/exploratory/frontier_figure/frontier_data.json"),
       notes="plotted values reused verbatim from tables_final.json / stats reports, "
             "never recomputed; Amendment 24 analysis 6")

    # Exploratory analyses 1-5 (Amendment 24): dir-level entries over their key artifacts
    expl = {
        "analysis1_discordance": "results/exploratory/discordant_pairs_t_vs_b3",
        "analysis2_variance": "results/exploratory/variance_decomposition",
        "analysis3_mde": "results/exploratory/analysis3_retrospective_mde",
        "analysis4_blindness_mechanism": "results/exploratory/analysis4_classifier_blindness_mechanism",
        "analysis5_direction": "results/exploratory/analysis5_t_vs_tctrl_direction",
        "ratio_ablation": "results/exploratory/ratio_ablation",
        "analysis7_b0_vs_b1": "results/exploratory/analysis7_b0_vs_b1_sft_erosion",
    }
    for name, d in expl.items():
        dp = REPO / d
        if not dp.is_dir():
            die(f"exploratory dir missing: {dp}")
        files = sorted(p for p in dp.iterdir() if p.is_file())
        if not files:
            die(f"exploratory dir empty: {dp}")
        ex(f"exploratory.{name}", "exploratory_table", "exploratory",
           {rel(p): sha256_file(p) for p in files},
           ["scored_b0", "scored_b1"] if name == "analysis7_b0_vs_b1"
           else ["primary_report"], files[0],
           notes="post-hoc exploratory (Amendment 24, or Amendment 26 for analysis7), "
                 "decided after seeing the primary result; reported as such, never "
                 "confirmatory")

    manifest = {
        "record_type": "results_manifest",
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/make_results_manifest.py",
        "git_head": git_head,
        "git_branch": git_branch,
        "working_tree_dirty": dirty,
        "policy_note": (
            "results/ artifacts are deliberately not version-controlled (repository "
            "rule); their integrity is fixed by the sha256 values recorded here. git_head "
            "identifies the exact code+config tree the artifacts were produced and "
            "verified under. Every value in `exhibits` was read live from the named "
            "artifact when this manifest was generated; regenerate and diff to audit."),
        "pins": pins,
        "sources": sources,
        "exhibits": exhibits,
    }

    out_json = REPO / "notebook" / "results_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"[written] {out_json}  ({len(sources)} sources, {len(exhibits)} exhibits)")

    # ---- human-readable one-pager --------------------------------------------
    def short(sha): return sha[:12]

    lines = [
        "# Results manifest (one-page appendix version)",
        "",
        f"Generated {manifest['generated']} at git `{short(git_head)}` (branch {git_branch}). "
        f"Machine-readable version: `notebook/results_manifest.json`.",
        "",
        "Every number and figure in the paper traces to a results artifact, the config and",
        "seed that produced it, and a SHA-256 fixing its bytes. Generations, scored outputs",
        "and adapters are not version-controlled; this manifest is the integrity record.",
        "",
        "## Pinned instruments",
        "",
        "| Pin | Path | SHA-256 (12) |",
        "|---|---|---|",
    ]
    for k, v in pins.items():
        lines.append(f"| {k} | `{v['path']}` | `{short(v['sha256'])}` |")
    lines += [
        "",
        "## Arm provenance (scored artifact -> generations -> config, seed)",
        "",
        "| Arm | Scored artifact (sha12) | Generations (sha12) | Train config (sha12) | Train seed | Gen seed | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ["b0", "b1", "b2", "b3", "t", "t_ctrl"]:
        s = sources[f"scored_{arm}"]
        tc = (f"`{s['train_config']}` ({short(s['train_config_sha256'])})"
              if "train_config" in s else "— (base model)")
        lines.append(
            f"| {arm} | `{s['path']}` ({short(s['sha256'])}) | "
            f"`{s['generations_file']}` ({short(s['generations_sha256'])}) | {tc} | "
            f"{s.get('train_seed', '—')} | {s['generation_seed']} | {s['status']} |")
    lines += [
        "",
        "## Exhibits",
        "",
        "| Exhibit | Status | Artifact (sha12) | Sources |",
        "|---|---|---|---|",
    ]
    for e in exhibits:
        lines.append(f"| {e['id']} | {e['status']} | `{e['artifact']}` "
                     f"({short(e['artifact_sha256'])}) | {', '.join(e['sources'])} |")
    lines += [
        "",
        "**Sign convention.** " + tf["sign_convention"],
        "",
        "Notes. b2/b3/t over-refusal cells rest on the Revision 4 hand-label instrument",
        "(`labels_b3`, `labels_t`; b2 shares b3's label file -- the filter replaced no",
        "benign response); b0/b1/t_ctrl over-refusal are rubric-judge values (kappa",
        "0.074) and carry daggers in the paper. T_ctrl is a single-run weak control:",
        "no CI, no significance test (Amendment 9). Robustness seeds 2-3 are the",
        "sighted Amendment 23 panel; the pre-registered primary analysis is training",
        "seed 1. Exploratory exhibits are Amendment 24 and Amendment 26 post-hoc analyses,",
        "never confirmatory. The primary ASR difference is on the raw observed scale (judge",
        "recall 0.55): it is not directly comparable to the 10-point TRUE-scale",
        "pre-registered threshold.",
    ]
    out_md = REPO / "notebook" / "results_manifest.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[written] {out_md}")


if __name__ == "__main__":
    main()
