"""Acceptance test for the pinned helpfulness reward model interface (Phase 2 gate).

Scores 50 held-out PsychoCounsel-Preference pairs through the PRODUCTION scoring path
(eval_score.load_helpfulness_model / helpfulness_score_convs -- shared code, not a
reimplementation) and requires >= 70% pairwise accuracy ranking chosen > rejected.

Rationale: this is the last unverified model interface in the stack. A mis-loaded
sequence-classification head (wrong pooling position, wrong padding, wrong num_labels)
produces ~50% pairwise accuracy -- indistinguishable from noise -- and every Table 1
helpfulness number would inherit it. The beaver-dam EOS-omission defect (Amendment 18)
is the standing reminder that an interface can be wrong in ways invisible to
within-path diagnostics; a ground-truth ranking check is the cheap external anchor.

Held-out definition, deterministic (no selection freedom): rows of
data/processed/pref_helpful.jsonl whose content-hash pair_id (train_dpo.pair_id) appears
in NEITHER the B2 v4 nor the T seed-1 training manifest, taken in file order, first 50.
These pairs were never trained on by any arm. They are NOT held out with respect to the
reward model's own training (the PsychoCounsel authors trained it on this corpus), which
is fine: this is an interface check, not a generalisation claim, and in-distribution
pairs are exactly where a correctly loaded head must score far above chance.

Usage: python scripts/test_reward_model_acceptance.py --seed 0
(--seed is recorded for convention; no stochastic operation is used.)
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import yaml  # noqa: E402

import eval_score as es  # noqa: E402
from train_dpo import pair_id  # noqa: E402

N_PAIRS = 50
PASS_THRESHOLD = 0.70
MANIFESTS = [
    REPO / "results/B2_dpo_seed1_v4/dpo_data_manifest.json",
    REPO / "results/T_dpo_seed1/dpo_data_manifest.json",
]
PREF_FILE = REPO / "data/processed/pref_helpful.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=str(REPO / "configs/judges.yaml"))
    ap.add_argument("--n_pairs", type=int, default=N_PAIRS)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    pin = es.verify_judge_pin(cfg, Path(args.config))
    print(f"[pin] {pin['status']}  helpfulness: {pin.get('helpfulness_scorer_verified')}")
    if pin["status"] != "VERIFIED":
        raise RuntimeError("judge pin did not verify; refusing to run the acceptance test")

    trained_ids = set()
    for mp in MANIFESTS:
        m = json.loads(mp.read_text(encoding="utf-8"))
        ids = m["sampled_helpful_pair_ids"]
        trained_ids |= set(ids)
        print(f"[exclude] {mp.parent.name}: {len(ids)} trained helpful pair ids")
    print(f"[exclude] union: {len(trained_ids)} ids")

    pairs, n_seen = [], 0
    with open(PREF_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_seen += 1
            p = json.loads(line)
            if pair_id(p) in trained_ids:
                continue
            pairs.append(p)
            if len(pairs) >= args.n_pairs:
                break
    if len(pairs) < args.n_pairs:
        raise RuntimeError(f"only {len(pairs)} held-out pairs found (wanted {args.n_pairs})")
    print(f"[pairs] {len(pairs)} held-out pairs selected (first-in-file-order rule, "
          f"scanned {n_seen} rows)")

    hcfg = cfg["helpfulness"]
    tok, model, info = es.load_helpfulness_model(hcfg)
    if model is None:
        raise RuntimeError(f"reward model unavailable: {info}")
    print(f"[model] {hcfg['name_or_path']}@{hcfg.get('revision')} from {info}")

    results, n_correct, n_ties = [], 0, 0
    bs = int(hcfg["batch_size"])
    convs, tags = [], []
    for i, p in enumerate(pairs):
        # pair["prompt"] already carries the pinned system message (prepare_pref.py);
        # do not add a second one.
        convs.append(list(p["prompt"]) + list(p["chosen"]))
        tags.append((i, "chosen"))
        convs.append(list(p["prompt"]) + list(p["rejected"]))
        tags.append((i, "rejected"))
    scores = {}
    for s in range(0, len(convs), bs):
        vals = es.helpfulness_score_convs(tok, model, convs[s:s + bs], hcfg)
        for tag, v in zip(tags[s:s + bs], vals):
            scores[tag] = v
        print(f"  [score] {min(s + bs, len(convs))}/{len(convs)}")
    for i, p in enumerate(pairs):
        c, r = scores[(i, "chosen")], scores[(i, "rejected")]
        correct = c > r
        n_correct += int(correct)
        n_ties += int(c == r)
        results.append({"pair_id": pair_id(p), "reward_chosen": c, "reward_rejected": r,
                        "margin": c - r, "correct": correct})

    acc = n_correct / len(pairs)
    verdict = "PASS" if acc >= PASS_THRESHOLD else "FAIL"
    kst = timezone(timedelta(hours=9))
    out = {
        "record_type": "reward_model_acceptance_test",
        "timestamp": datetime.now(kst).isoformat(timespec="seconds"),
        "script": "scripts/test_reward_model_acceptance.py",
        "seed": args.seed,
        "model": hcfg["name_or_path"],
        "revision": hcfg.get("revision"),
        "snapshot": info,
        "judge_pin": {"status": pin["status"],
                      "pin_lock_sha256": pin.get("pin_lock_sha256"),
                      "helpfulness_scorer_verified": pin.get("helpfulness_scorer_verified")},
        "selection_rule": ("pref_helpful.jsonl rows whose pair_id is in neither the B2 v4 "
                           "nor the T seed-1 manifest, file order, first "
                           f"{args.n_pairs}"),
        "n_pairs": len(pairs),
        "n_correct": n_correct,
        "n_ties": n_ties,
        "pairwise_accuracy": acc,
        "pass_threshold": PASS_THRESHOLD,
        "verdict": verdict,
        "note_if_near_chance": ("accuracy near 0.5 indicates a head-loading/pooling/padding "
                                "defect, NOT a property of the model -- stop and debug "
                                "before accepting any helpfulness number"),
        "results": results,
    }
    out_dir = REPO / "results/judge_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reward_acceptance_{datetime.now(kst).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] pairwise accuracy {acc:.3f} ({n_correct}/{len(pairs)}, "
          f"{n_ties} ties) -> {verdict}")
    print(f"[artefact] {out_path}")
    if verdict != "PASS":
        print("!! ACCEPTANCE FAILED: do not score helpfulness; debug the interface first.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
