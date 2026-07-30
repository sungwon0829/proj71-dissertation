"""Build the frozen red-team evaluation suite.

Which paper number this feeds: every ASR cell in Table 1, every cell in Table 2,
and the over-refusal column of Table 1 (via the benign_sensitive category).

Deterministic by construction: item text lives in scripts/redteam_items/*.py and
ids are assigned from list order. Re-running produces a byte-identical file.
--seed is accepted for convention only and is logged, not used.

Outputs:
  data/redteam/redteam_suite.jsonl          <- the suite (harness reads this)
  data/redteam/<category>.jsonl             <- per-category splits (same items)

THIS DATA IS EVALUATION-ONLY AND MUST NEVER ENTER ANY TRAINING SET.

Usage:
  C:\\proj71\\env\\Scripts\\python.exe C:\\proj71\\scripts\\build_redteam_suite.py --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from redteam_items import (  # noqa: E402
    benign_sensitive,
    crisis_adjacent,
    many_shot,
    persona,
    prefilling,
)

CATEGORIES = [
    ("prefilling", prefilling.ITEMS),
    ("persona", persona.ITEMS),
    ("many_shot", many_shot.ITEMS),
    ("crisis_adjacent", crisis_adjacent.ITEMS),
    ("benign_sensitive", benign_sensitive.ITEMS),
]

FIELD_ORDER = [
    "id",
    "category",
    "messages",
    "attack_note",
    "expected_safe_behaviour",
    "source_or_technique",
    "citation",
]


def build() -> list[dict]:
    records: list[dict] = []
    for category, items in CATEGORIES:
        for idx, item in enumerate(items, start=1):
            missing = [
                k
                for k in (
                    "messages",
                    "attack_note",
                    "expected_safe_behaviour",
                    "source_or_technique",
                    "citation",
                )
                if not item.get(k)
            ]
            if missing:
                raise SystemExit(
                    f"FAIL: {category} item {idx} missing fields: {missing}"
                )
            rec = {
                "id": f"{category}_{idx:03d}",
                "category": category,
                "messages": item["messages"],
                "attack_note": item["attack_note"],
                "expected_safe_behaviour": item["expected_safe_behaviour"],
                "source_or_technique": item["source_or_technique"],
                "citation": item["citation"],
            }
            records.append({k: rec[k] for k in FIELD_ORDER})
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="logged only; build is deterministic")
    ap.add_argument("--out-dir", default=str(REPO / "data" / "redteam"))
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    records = build()

    suite_path = out_dir / "redteam_suite.jsonl"
    write_jsonl(suite_path, records)

    for category, _ in CATEGORIES:
        subset = [r for r in records if r["category"] == category]
        write_jsonl(out_dir / f"{category}.jsonl", subset)

    print(f"seed (logged, unused): {args.seed}")
    print(f"wrote {len(records)} items -> {suite_path}")
    for category, _ in CATEGORIES:
        n = sum(1 for r in records if r["category"] == category)
        print(f"  {category:<18} {n}")
    print(f"SHA-256 (redteam_suite.jsonl): {sha256(suite_path)}")


if __name__ == "__main__":
    main()
