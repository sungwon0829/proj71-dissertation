"""Corrected-path re-measurement of beaver-dam-7b's ASR-judge agreement (Amendment 18a).

WHAT THIS IS. The three beaver-dam kappa values that justified half of pre-registration
Revision 2 (0.3552 / 0.0857 / 0.1156) were measured under the EOS-omission interface
defect fixed on 2026-08-03 (Amendment 18): the scoring path omitted the end-of-sequence
terminator the BeaverTails pipeline appends before tokenising, so the classifier pooled at
an arbitrary content token. This script re-scores beaver-dam over THE SAME three
validation sets, against THE SAME reference labels, with THE SAME item construction
(`render_conversation` + raw response, identical to `run_calibration`) -- the only change
is the corrected `MultiLabelClassifierJudge` path.

WHAT THIS IS NOT. Per Amendments 11 and 18, this is a *permitted measurement* that can
never change judge roles. Whatever the corrected kappas are, beaver-dam remains the B3
filter (or is demoted to diagnostic-only under Amendment 19's Llama Guard branch) and the
behavioural judge remains the ASR judge. The output feeds the Amendment 18a contamination
register and, if quoted in the paper, is quoted as a corrected-path historical footnote.

CAVEATS CARRIED FROM THE ORIGINALS, unchanged by re-measurement:
  - Reference labels are AGENT labels (labeller_is_human false throughout) -- this is
    inter-model agreement, not human agreement, exactly as the originals were.
  - judge_validation_set.jsonl (50/100) and heldout2/3 (30/60, 18/36) contain B1 v1
    generations (the void checkpoint) -- see the 2026-08-03 retired-checkpoint sweep.
    Reusing the same sets is DELIBERATE: a like-for-like comparison isolates the bug fix.
    These kappas inherit both caveats and are labelled accordingly.

Originals are never overwritten; output is a new corrected-path report per set.

Usage: python scripts/remeasure_beaverdam_kappa.py [--seed 0] [--batch_size N]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from eval_score import (  # noqa: E402
    MultiLabelClassifierJudge, agreement_block, load_backend, load_yaml,
    render_conversation, resolve, sha256_file,
)

# (set path, the voided bug-era kappa it must be compared against)
SETS = [
    ("results/human_validation/judge_validation_set.jsonl", 0.3552),
    ("results/human_validation/heldout2_validation_set.jsonl", 0.0857),
    ("results/human_validation/heldout3_validation_set.jsonl", 0.1156),
]


def load_set(path: Path):
    items, refs = [], {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        for fld in ("id", "category", "messages", "response", "reference_label",
                    "labeller_is_human"):
            if fld not in r:
                raise ValueError(f"{path}:{lineno}: missing {fld!r}; refusing to guess.")
        if r["labeller_is_human"] is not False:
            # If a human-labelled row ever appears here this script must not silently
            # blend provenances -- that is the exact incident class of 2026-08-01.
            raise ValueError(f"{path}:{lineno}: labeller_is_human={r['labeller_is_human']!r}; "
                             f"this script documents AGENT-label agreement only.")
        items.append({
            "id": r["id"], "category": r["category"],
            "conversation_text": render_conversation(r["messages"], r.get("is_prefill", False)),
            "response_text": r["response"],
        })
        refs[r["id"]] = r["reference_label"]
    return items, refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/judges.yaml")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap CI seed")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--allow_overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(resolve(args.config))
    bcfg = dict(cfg["backends"]["beaver_dam"])
    bs = args.batch_size or int(bcfg["batch_size"])
    model, tok, device = load_backend(bcfg)
    judge = MultiLabelClassifierJudge(model, tok, device, bcfg, "unsafe")
    # The fix must actually be active, or this "re-measurement" would re-measure the bug.
    probe = judge.build_prompt({"conversation_text": "c", "response_text": "r"})
    if not probe.endswith(tok.eos_token):
        raise RuntimeError("build_prompt does not append the EOS terminator -- the P-K fix "
                           "is not active in this checkout. Refusing to run.")

    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    out_summary = []
    for rel, voided in SETS:
        path = resolve(rel)
        items, refs = load_set(path)
        verdicts = {}
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]
            for it, v in zip(batch, judge.judge_batch(batch)):
                verdicts[it["id"]] = v
        ids = [it["id"] for it in items]
        human = [refs[i] for i in ids]
        pred = [("unsafe" if verdicts[i]["is_positive"] else "safe") for i in ids]
        blk = agreement_block(human, pred, "unsafe", ids, seed=args.seed)
        report = {
            "record_type": "beaverdam_corrected_path_kappa",
            "amendment": "18a",
            "script": "scripts/remeasure_beaverdam_kappa.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "timestamp": stamp,
            "seed": args.seed,
            "validation_set": str(path), "validation_set_sha256": sha256_file(path),
            "n_items": len(items),
            "voided_bug_era_kappa": voided,
            "corrected_path": True,
            "eos_fix_verified_active": True,
            "agreement": blk,
            "reference_labels_are_agent_labels": True,
            "role_note": ("Permitted measurement under Amendments 11/18/19: documents the "
                          "corrected-path agreement of a mechanism whose judge role was and "
                          "remains closed. Not a paper number; never a role argument."),
            "rows": [{"id": i, "reference_label": refs[i],
                      "verdict": ("unsafe" if verdicts[i]["is_positive"] else "safe"),
                      "max_category": verdicts[i].get("beaverdam_max_category"),
                      "max_prob": verdicts[i].get("beaverdam_max_prob"),
                      "flagged_categories": verdicts[i].get("beaverdam_flagged")}
                     for i in ids],
        }
        dest = path.parent / (path.stem.replace("_validation_set", "")
                              + "_beaverdam_corrected_kappa.json")
        if dest.exists() and not args.allow_overwrite:
            raise FileExistsError(f"{dest} exists; originals and prior reports are never "
                                  f"silently replaced. Pass --allow_overwrite deliberately.")
        dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        k = blk["cohens_kappa"]
        ci = blk.get("cohens_kappa_ci95", {})
        print(f"[{path.name}] corrected kappa={k}  CI95=[{ci.get('lo')},{ci.get('hi')}]  "
              f"(voided bug-era value: {voided})  n={len(items)}  -> {dest.name}")
        out_summary.append((path.name, voided, k, ci))

    print("\nSummary (voided -> corrected):")
    for name, voided, k, ci in out_summary:
        print(f"  {name:42} {voided}  ->  {k}  [{ci.get('lo')}, {ci.get('hi')}]")


if __name__ == "__main__":
    main()
