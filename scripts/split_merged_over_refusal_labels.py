"""Split the filled MERGED over-refusal label stub into per-arm label files.

WHY: eval_score.py --over_refusal_labels consumes {suite_item_id: 'refusal'|'complied'}
for ONE arm, but the labelling happens on the MERGED, arm-masked worksheet whose stub is
keyed item001..itemNNN across both arms (B3 and T share suite ids, so a single merged
file keyed by suite id cannot exist). This script resolves item keys through the sealed
key and writes one label file per source tag, carrying provenance forward.

Checks, in order, all before anything is written (mirrors compute_kappa.py's
cheap-checks-first convention):
  1. every item key labelled 'refusal'/'complied', no placeholder labeller,
     _labeller_is_human present;
  2. gold-item pass (if over_refusal_GOLD_ANSWER_KEY.json exists): every gold item's
     label matches its pre-registered answer, else REFUSE (--allow_gold_failures
     overrides, stamping every output NOT_TRUSTWORTHY);
  3. relabel-pass consistency reported (raw agreement, n too small for kappa).

Usage:
    python scripts/split_merged_over_refusal_labels.py \
        --labels results/human_validation/over_refusal_labels_MERGED.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HV = REPO / "results/human_validation"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(HV / "over_refusal_labels_MERGED.json"))
    ap.add_argument("--key", default=str(HV / "over_refusal_MERGED_SEALED_KEY.json"))
    ap.add_argument("--gold_key", default=str(HV / "over_refusal_GOLD_ANSWER_KEY.json"))
    ap.add_argument("--out_dir", default=str(HV))
    ap.add_argument("--allow_gold_failures", action="store_true")
    args = ap.parse_args()

    labels_path, key_path = Path(args.labels), Path(args.key)
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))

    # 1. completeness + provenance
    if "_labeller_is_human" not in raw:
        raise SystemExit("REFUSED: labels file has no _labeller_is_human field")
    who = raw.get("_labeller")
    if not who or "FILL IN" in str(who):
        raise SystemExit(f"REFUSED: _labeller missing or placeholder ({who!r})")
    lab = {k: v for k, v in raw.items() if not k.startswith("_")}
    missing = sorted(set(key["items"]) - set(lab))
    extra = sorted(set(lab) - set(key["items"]))
    if missing or extra:
        raise SystemExit(f"REFUSED: labels do not match the sealed key "
                         f"(missing {missing[:5]}, extra {extra[:5]})")
    bad = {k: v for k, v in lab.items() if v not in ("refusal", "complied")}
    if bad:
        raise SystemExit(f"REFUSED: non-refusal/complied labels: {dict(list(bad.items())[:5])}")

    # 2. gold check BEFORE anything else is computed or written
    trust = "TRUSTWORTHY"
    gold_report = {"n_gold": 0}
    gp = Path(args.gold_key)
    if gp.is_file():
        gold = json.loads(gp.read_text(encoding="utf-8"))["items"]
        fails = [k for k, g in gold.items() if lab[k] != g["expected_label"]]
        gold_report = {"n_gold": len(gold), "n_passed": len(gold) - len(fails),
                       "failed_items": fails}
        if fails and not args.allow_gold_failures:
            raise SystemExit(
                f"REFUSED: gold-item check failed on {fails} "
                f"({len(fails)}/{len(gold)}). These items were chosen to be unambiguous, so "
                f"a failure is evidence about the labelling pass. Re-check those labels, or "
                f"re-run with --allow_gold_failures (stamps outputs NOT_TRUSTWORTHY).")
        if fails:
            trust = "NOT_TRUSTWORTHY"
        print(f"[gold] {gold_report['n_passed']}/{gold_report['n_gold']} passed")
    else:
        print("[gold] no gold answer key found; skipping (report this absence)")

    # 3. relabel-pass consistency
    relabel = {k: v for k, v in (raw.get("_relabel_pass") or {}).items() if v}
    rel_report = None
    if relabel:
        n_agree = sum(1 for k, v in relabel.items() if lab.get(k) == v)
        rel_report = {"n": len(relabel), "n_agree": n_agree,
                      "percent_agreement": round(100.0 * n_agree / len(relabel), 1),
                      "disagreements": sorted(k for k, v in relabel.items()
                                              if lab.get(k) != v)}
        print(f"[relabel] intra-rater {n_agree}/{len(relabel)} "
              f"({rel_report['percent_agreement']}%)")
    else:
        print("[relabel] WARNING: _relabel_pass is empty or absent -- the intra-rater "
              "check was not done; report this")

    # split by source tag
    by_tag = {}
    for item_key, meta in key["items"].items():
        by_tag.setdefault(meta["source_tag"], []).append((item_key, meta))
    out_dir = Path(args.out_dir)
    written = []
    for tag, items in sorted(by_tag.items()):
        suite_ids = [m["suite_id"] for _, m in items]
        if len(set(suite_ids)) != len(suite_ids):
            raise SystemExit(f"REFUSED: duplicate suite ids inside tag {tag!r}")
        out = {
            "_labeller": raw["_labeller"],
            "_labeller_is_human": raw["_labeller_is_human"],
            "_trust_stamp": trust,
            "_protocol": (raw.get("_protocol") or []) + [
                f"split from {labels_path.name} (sha256 {sha256_file(labels_path)}) via "
                f"{key_path.name} (sha256 {sha256_file(key_path)}) by "
                f"scripts/split_merged_over_refusal_labels.py",
                f"source_tag: {tag}; n_items: {len(items)}",
                f"gold check: {gold_report}",
                f"relabel pass: {rel_report}",
            ],
            **{m["suite_id"]: lab[k] for k, m in sorted(items, key=lambda x: x[1]["suite_id"])},
        }
        p = out_dir / f"over_refusal_labels_{tag}_from_merged.json"
        if p.exists():
            raise SystemExit(f"REFUSED: {p} exists; will not overwrite a labelling artefact")
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        n_ref = sum(1 for k, m in items if lab[k] == "refusal")
        print(f"[written] {p}  ({len(items)} items, {n_ref} labelled refusal)")
        written.append(str(p))
    print("\n[next] pass each file to eval_score.py --over_refusal_labels for its arm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
