"""
Project 71 -- P-K BEAVER-DAM SCORING-PIPELINE VERIFICATION.

WHICH PAPER NUMBER: none directly. This probe decides whether the B3 filter numbers
(the entire B3 row of Table 1 and B3 column of Table 2) are built on a correctly
implemented pipeline. The P-D probe's beaver-dam columns are QUARANTINED until this
either finds the bug or certifies the pipeline.

WHAT IT DOES
  1. Re-scores, through the PRODUCTION path (MultiLabelClassifierJudge exactly as
     scripts/apply_b3_filter.py constructs it: b3_filter threshold, backend template,
     2048/left truncation, right padding, bfloat16), the P-D probe's items:
     6 gold severe items, 5 safe controls, and the 8 hand-written smoke pairs
     (photosynthesis + CV among them) VERBATIM from the P-D artefact.
  2. Scores the same items under the BEAVERTAILS-CANONICAL input construction
     (examples/moderation/moderation.py appends tokenizer.eos_token to the text
     before tokenising; the training pipeline dataset.py line 101 does the same).
     Our production path does NOT append EOS -- this pass measures whether that
     omission changes any verdict.
  3. Re-runs the P-D in-distribution ground-truth check (100 PKU-SafeRLHF test
     responses) under both constructions and compares recall / FPR / accuracy.

The canonical construction pass is a DIAGNOSTIC. It changes no pinned value: the
threshold (0.5, matching the beavertails repo's own evaluate.py), the flag rule
(any_category_over_threshold == repo's max-over-threshold), and the input_template
(byte-identical to safe-rlhf PROMPT_INPUT) are untouched.

Usage:
    python scripts\\probe_pk_beaverdam_pipeline.py
"""

import argparse
import datetime
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_score import (  # noqa: E402
    MultiLabelClassifierJudge, load_backend, load_yaml, resolve, sha256_file,
)
from probe_gold_items import (  # noqa: E402 -- identical item selection, not a copy
    beaverdam_indistribution_check, load_controls, load_gold_items,
)

PROBE_ID = "P-K"
PD_ARTEFACT = "results/judge_probes/PD_gold_item_probe_20260802_seed0_run2_full.json"

# The 8 hand-written smoke ids, taken VERBATIM (conversation_text + response_text)
# from the P-D artefact's beaverdam_smoke_test rows -- never re-typed here.
SMOKE_IDS = ["smoke_harmful_1", "smoke_harmful_2", "smoke_harmful_3",
             "smoke_harmless_1", "smoke_harmless_2", "smoke_harmless_3",
             "smoke_diag_topic", "smoke_diag_replacement"]


class EOSAppendingJudge(MultiLabelClassifierJudge):
    """The beavertails-canonical variant: identical in every respect to the production
    judge except that tokenizer.eos_token is appended to the built text before
    tokenisation, mirroring examples/moderation/moderation.py predict() line 268 and
    the training-side dataset.py line 101. DIAGNOSTIC ONLY."""

    def build_prompt(self, item, assistant_so_far=None):
        t = super().build_prompt(item)
        return t if t.endswith(self.tok.eos_token) else t + self.tok.eos_token


def slim(v: dict) -> dict:
    return {"flagged": v["is_positive"],
            "flagged_categories": v["beaverdam_flagged"],
            "max_category": v["beaverdam_max_category"],
            "max_prob": v["beaverdam_max_prob"],
            "categories": v["beaverdam_categories"],
            "input_n_tokens": v["input_n_tokens"],
            "input_truncated_2048": v["input_truncated"]}


def main():
    ap = argparse.ArgumentParser(description="Project 71 P-K beaver-dam pipeline verification")
    ap.add_argument("--config", default="configs/judges.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    stamp = datetime.datetime.now().astimezone()
    out_path = resolve(args.out) if args.out else resolve(
        f"results/judge_probes/PK_beaverdam_pipeline_verification_"
        f"{stamp.strftime('%Y%m%d')}_seed{args.seed}.json")
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists. Probe artefacts are never overwritten.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = resolve(args.config)
    cfg = load_yaml(cfg_path)
    fcfg = cfg["b3_filter"]
    bcfg = dict(cfg["backends"][fcfg["backend"]])
    bcfg["threshold"] = float(fcfg["threshold"])   # the FILTER threshold, as apply_b3_filter.py

    # ---- P-D reference values -----------------------------------------------------------
    pd_path = resolve(PD_ARTEFACT)
    pd = json.loads(pd_path.read_text(encoding="utf-8"))
    pd_sha = sha256_file(pd_path)
    pd_smoke = {r["id"]: r for r in pd["beaverdam_smoke_test"]["rows"]}
    pd_c = {r["probe_id"]: r["C_b3_filter_beaverdam"] for r in pd["items"]}

    # ---- items --------------------------------------------------------------------------
    gold = load_gold_items()          # identical loader, identical fail-loudly guarantees
    controls, human_status = load_controls()
    expected_control_ids = [r["probe_id"] for r in pd["items"]
                            if r["role_in_probe"] == "negative_control_safe"]
    got_control_ids = [c["probe_id"] for c in controls]
    if got_control_ids != expected_control_ids:
        raise RuntimeError(f"Control selection diverged from P-D: {got_control_ids} vs "
                           f"{expected_control_ids}. Refusing to compare unlike with unlike.")
    smoke_items = []
    for sid in SMOKE_IDS:
        if sid not in pd_smoke:
            raise RuntimeError(f"Smoke id {sid} absent from the P-D artefact")
        r = pd_smoke[sid]
        smoke_items.append({"probe_id": f"smoke/{sid}", "id": sid,
                            "role_in_probe": "smoke_pair_verbatim_from_PD",
                            "expect": r["expect"],
                            "conversation_text": r["conversation_text"],
                            "response_text": r["response_text"]})
    items = gold + controls + smoke_items
    print(f"[{PROBE_ID}] {len(gold)} gold + {len(controls)} controls + {len(smoke_items)} "
          f"smoke pairs (verbatim from P-D)")

    # ---- load ONE backend, the production way ------------------------------------------
    print(f"[{PROBE_ID}] loading beaver-dam via eval_score.load_backend ...")
    model, tok, device = load_backend(bcfg)
    prod = MultiLabelClassifierJudge(model, tok, device, bcfg, "unsafe")
    eosj = EOSAppendingJudge(model, tok, device, bcfg, "unsafe")

    # EOS string really must tokenise to the EOS token id, or the canonical pass is fake.
    probe_ids = tok("hello" + tok.eos_token, add_special_tokens=True)["input_ids"]
    if probe_ids[-1] != tok.eos_token_id:
        raise RuntimeError(f"'{tok.eos_token}' does not tokenise to eos_token_id "
                           f"({probe_ids[-1]} != {tok.eos_token_id}); the canonical pass "
                           f"would not reproduce the repo pipeline. Aborting.")

    def run(judge, tag):
        out = []
        for s in range(0, len(items), args.batch_size):
            chunk = items[s: s + args.batch_size]
            out.extend(judge.judge_batch(chunk))
            print(f"  [{tag}] {min(s + args.batch_size, len(items))}/{len(items)}")
        return out

    prod_out = run(prod, "production")
    eos_out = run(eosj, "canonical-eos")

    # ---- in-distribution check, both constructions --------------------------------------
    print(f"[{PROBE_ID}] in-distribution check, production construction ...")
    indist_prod = beaverdam_indistribution_check(prod, batch_size=args.batch_size)
    print(f"    recall={indist_prod.get('recall_on_unsafe')} "
          f"fpr={indist_prod.get('false_positive_rate_on_safe')} "
          f"acc={indist_prod.get('accuracy')}")
    print(f"[{PROBE_ID}] in-distribution check, canonical-EOS construction ...")
    indist_eos = beaverdam_indistribution_check(eosj, batch_size=args.batch_size)
    print(f"    recall={indist_eos.get('recall_on_unsafe')} "
          f"fpr={indist_eos.get('false_positive_rate_on_safe')} "
          f"acc={indist_eos.get('accuracy')}")

    # ---- assemble ------------------------------------------------------------------------
    rows = []
    for it, pv, ev in zip(items, prod_out, eos_out):
        pd_ref = None
        if it["probe_id"] in pd_c:
            c = pd_c[it["probe_id"]]
            pd_ref = {"source": "P-D column C (production construction)",
                      "flagged": c["flagged_and_would_be_replaced"],
                      "max_category": c["max_category"], "max_prob": c["max_prob"]}
        elif it["id"] in pd_smoke:
            s = pd_smoke[it["id"]]
            pd_ref = {"source": "P-D smoke test (hand-written construction, same judge class)",
                      "flagged": s["flagged"],
                      "max_category": s["max_category"], "max_prob": s["max_prob"]}
        rows.append({
            "probe_id": it["probe_id"], "id": it["id"], "role_in_probe": it["role_in_probe"],
            "expect": it.get("expect"),
            "conversation_text": it["conversation_text"],
            "response_text": it["response_text"],
            "production_path": slim(pv),
            "canonical_eos_diagnostic": slim(ev),
            "pd_reference": pd_ref,
            "verdict_changed_vs_pd": (None if pd_ref is None
                                      else pv["is_positive"] != pd_ref["flagged"]),
            "verdict_changed_by_eos": pv["is_positive"] != ev["is_positive"],
        })

    n_changed_pd = sum(1 for r in rows if r["verdict_changed_vs_pd"])
    n_changed_eos = sum(1 for r in rows if r["verdict_changed_by_eos"])

    artefact = {
        "record_type": "judge_probe",
        "probe": PROBE_ID,
        "title": "beaver-dam scoring-pipeline verification (P-D quarantine resolution)",
        "timestamp": stamp.isoformat(timespec="seconds"),
        "script": "scripts/probe_pk_beaverdam_pipeline.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed, "batch_size": args.batch_size,
        "is_paper_number": False,
        "config_file": str(cfg_path), "config_sha256": sha256_file(cfg_path),
        "pd_artefact": str(pd_path), "pd_artefact_sha256": pd_sha,
        "canonical_reference": {
            "source": "github.com/PKU-Alignment/beavertails examples/moderation/ "
                      "(constants.py, moderation.py, dataset.py, evaluate.py), fetched "
                      "2026-08-03; copies in the session scratchpad",
            "prompt_input": "PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT = "
                            "'BEGINNING OF CONVERSATION: ' + 'USER: {input} ' + 'ASSISTANT:' "
                            "(constants.py 44-47; 'should not have a space at the end')",
            "qa_construction": "text = PROMPT_INPUT.format(input=question) + answer "
                               "(moderation.py QAModeration.predict, line 373/375)",
            "eos_handling": "predict() appends tokenizer.eos_token to the text before "
                            "tokenising (moderation.py 267-270); the TRAINING pipeline does "
                            "the same (dataset.py 101). OUR PRODUCTION PATH DOES NOT.",
            "max_length": "repo: model_max_length 512 (from_pretrained default; README and "
                          "evaluate.py pass 512), truncation LONGEST_FIRST i.e. right. "
                          "OURS: 2048 with LEFT truncation -- a deliberate, documented "
                          "deviation so the response under judgement is never truncated away.",
            "threshold": "repo evaluate.py passes threshold=0.5 (moderation.py's default is "
                         "0.4). Our pinned 0.5 MATCHES the repo's own evaluation script.",
            "flag_rule": "repo: flagged = sigmoid(logits).max(dim=1) > threshold "
                         "(moderation.py 291-292) == our any_category_over_threshold, both "
                         "strict '>'.",
            "label_order": "constants.py LABEL_NAMES (index order) is byte-identical to "
                           "config.json id2label 0..13; our code reads model.config.id2label "
                           "by integer index, so no permutation is possible.",
            "score_semantics": "problem_type=multi_label_classification; 14 INDEPENDENT "
                               "sigmoids P(category|QA), not a softmax.",
        },
        "control_provenance_human_label_status": human_status,
        "indistribution_check": {
            "production_construction_no_eos": indist_prod,
            "canonical_construction_with_eos": indist_eos,
        },
        "summary": {
            "n_items": len(rows),
            "n_verdicts_changed_vs_pd_under_production_path": n_changed_pd,
            "changed_vs_pd_ids": [r["probe_id"] for r in rows if r["verdict_changed_vs_pd"]],
            "n_verdicts_changed_by_eos_append": n_changed_eos,
            "changed_by_eos_ids": [r["probe_id"] for r in rows if r["verdict_changed_by_eos"]],
        },
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": str(next(model.parameters()).dtype),
        },
        "items": rows,
        "wallclock_seconds": round(time.time() - t0, 2),
    }
    out_path.write_text(json.dumps(artefact, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- verdict table -------------------------------------------------------------------
    print("\n===== " + PROBE_ID + " VERDICT TABLE (production path) " + "=" * 30)
    hdr = (f"{'item':<36}{'prod':<30}{'canonical+eos':<30}{'P-D':<26}{'chg?':<6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        p, e, d = r["production_path"], r["canonical_eos_diagnostic"], r["pd_reference"]
        ps = ("FLAG " if p["flagged"] else "pass ") + f"{p['max_category'][:16]}={p['max_prob']:.3f}"
        es = ("FLAG " if e["flagged"] else "pass ") + f"{e['max_category'][:16]}={e['max_prob']:.3f}"
        ds = "-" if d is None else ("FLAG " if d["flagged"] else "pass ") + f"{d['max_prob']:.3f}"
        ch = "" if r["verdict_changed_vs_pd"] in (None, False) else "YES"
        print(f"{r['probe_id']:<36}{ps:<30}{es:<30}{ds:<26}{ch:<6}")
    print("-" * len(hdr))
    print(json.dumps(artefact["summary"], indent=1))
    print(f"\n[indist no-eos ] recall={indist_prod.get('recall_on_unsafe')} "
          f"fpr={indist_prod.get('false_positive_rate_on_safe')} acc={indist_prod.get('accuracy')}")
    print(f"[indist with-eos] recall={indist_eos.get('recall_on_unsafe')} "
          f"fpr={indist_eos.get('false_positive_rate_on_safe')} acc={indist_eos.get('accuracy')}")
    print(f"\n[written] {out_path}")
    print(f"[wallclock] {artefact['wallclock_seconds']} s")


if __name__ == "__main__":
    main()
