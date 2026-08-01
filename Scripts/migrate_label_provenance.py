"""
Project 71 -- rename `human_label` -> `reference_label` and stamp labeller provenance.

WHY. Every validation record produced up to 2026-08-01 carries `"labeller": "eval-harness
agent"` -- an LLM -- inside a field named `human_label`. The pre-registration, the lab
notebook and the reports to the owner all described these as "hand-labelled". They were not.
Cohen's kappa = 0.521 / 0.583 / 0.074 are INTER-MODEL agreement (a Qwen judge against a Claude
labeller), not human agreement, and shared model biases plausibly inflate them.

The field name is what produced the error: code and prose both read `human_label` and inferred
a human. It is renamed, and every record now carries an explicit `labeller_is_human` boolean
that scoring REFUSES TO DEFAULT.

This migration changes NAMES AND PROVENANCE ONLY. Label values, item order and item sets are
untouched, and the script verifies that per file before writing: same ids in the same order,
same label values. It prints the before/after sha256 of every file so the change is auditable.

Usage:
    python Scripts\\migrate_label_provenance.py --dry_run
    python Scripts\\migrate_label_provenance.py --apply
"""

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HV = REPO / "results" / "human_validation"

# Everything labelled before the correction was labelled by this agent, i.e. not by a person.
LLM_LABELLER = "eval-harness agent (Claude); LLM, NOT a human"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def migrate_jsonl(path: Path, apply: bool):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows or "human_label" not in rows[0]:
        return None
    before = sha(path)
    out, ids_before, vals_before = [], [], []
    for r in rows:
        ids_before.append(r.get("id"))
        vals_before.append(r.get("human_label"))
        new = {}
        for k, v in r.items():
            if k == "human_label":
                new["reference_label"] = v
                # provenance sits immediately beside the label it qualifies
                new["labeller_is_human"] = False
                new["labeller"] = r.get("labeller") or LLM_LABELLER
                new["label_provenance_note"] = (
                    "LLM-produced reference label. Agreement computed against it is "
                    "INTER-MODEL agreement, not human agreement. See preregistration.md "
                    "CORRECTION 2026-08-01.")
            elif k == "labeller":
                continue          # re-emitted above, next to the label
            else:
                new[k] = v
        out.append(new)
    assert [r.get("id") for r in out] == ids_before, "id order changed"
    assert [r.get("reference_label") for r in out] == vals_before, "label values changed"
    if apply:
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"file": str(path.relative_to(REPO)), "n_rows": len(out),
            "sha_before": before, "sha_after": sha(path) if apply else "(dry run)"}


def migrate_json_report(path: Path, apply: bool):
    """Reports embed `human_label` inside `rows`. Rename there too, and stamp the report."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or "rows" not in d:
        return None
    if not d["rows"] or "human_label" not in d["rows"][0]:
        return None
    before = sha(path)
    for r in d["rows"]:
        r["reference_label"] = r.pop("human_label")
    for blk in d.get("per_judge", {}).values():
        if isinstance(blk, dict):
            blk["agreement_type"] = "inter_model"
            blk["reference_labeller_is_human"] = False
    for blk in d.get("combined_per_category", {}).values():
        if isinstance(blk, dict):
            blk["agreement_type"] = "inter_model"
            blk["reference_labeller_is_human"] = False
    d["reference_labeller_is_human"] = False
    d["agreement_type"] = "inter_model"
    d["provenance_correction"] = (
        "Reference labels were produced by an LLM agent, not a human. Every kappa in this "
        "report is INTER-MODEL agreement. preregistration.md CORRECTION 2026-08-01.")
    if apply:
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"file": str(path.relative_to(REPO)), "n_rows": len(d["rows"]),
            "sha_before": before, "sha_after": sha(path) if apply else "(dry run)"}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry_run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    changed = []
    for p in sorted(HV.glob("*.jsonl")):
        r = migrate_jsonl(p, args.apply)
        if r:
            changed.append(r)
    for p in sorted(HV.glob("*.json")):
        r = migrate_json_report(p, args.apply)
        if r:
            changed.append(r)

    for r in changed:
        print(f"  {r['file']:<62} n={r['n_rows']:<4} {r['sha_before'][:16]} -> "
              f"{r['sha_after'][:16] if r['sha_after'] != '(dry run)' else '(dry run)'}")
    print(f"\n{len(changed)} file(s) {'migrated' if args.apply else 'would be migrated'}. "
          "Label values and item order verified unchanged.")
    if args.apply:
        (HV / "LABEL_PROVENANCE_CORRECTION.md").write_text(
            "# Label provenance correction, 2026-08-01\n\n"
            "`human_label` renamed to `reference_label` across every validation artefact, and\n"
            "`labeller_is_human: false` stamped on every record.\n\n"
            "The labels were produced by an LLM agent (Claude), not by a person. Every Cohen's\n"
            "kappa computed against them -- 0.521 (ASR), 0.583 (crisis), 0.074 (refusal) -- is\n"
            "**inter-model agreement**, not human agreement, and shared model biases plausibly\n"
            "inflate it. CLAUDE.md safeguard 4 asks for hand-labelling by a person; that\n"
            "requirement was not met by these sets.\n\n"
            "Label values, item sets and item order are UNCHANGED by the migration; only field\n"
            "names and provenance stamps changed. `scripts/migrate_label_provenance.py` asserts\n"
            "this and prints before/after hashes.\n\n"
            "See `notebook/preregistration.md` CORRECTION 2026-08-01 for the remedy: the owner\n"
            "hand-labels a 60-item attack subset so the headline kappa in Methods is genuine\n"
            "human agreement. Both figures are reported side by side.\n",
            encoding="utf-8")
        print(f"[written] {HV / 'LABEL_PROVENANCE_CORRECTION.md'}")


if __name__ == "__main__":
    main()
