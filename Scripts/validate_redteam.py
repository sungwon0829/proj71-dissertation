"""Validate the frozen red-team suite against the schema contract.

Checks (all must pass):
  1. File parses as JSONL, one object per line.
  2. Every record has exactly the required fields (extras allowed only from the
     documented metadata set: source_or_technique, citation).
  3. category is in the allowed set.
  4. ids are unique and match the pattern <category>_<3-digit>.
  5. NO system messages anywhere (the harness prepends the pinned system prompt).
  6. roles are only 'user' or 'assistant'; turns strictly alternate starting with user.
  7. prefilling items END with an assistant turn (the forced prefix).
  8. Every non-prefilling item ENDS with a user turn.
  9. benign_sensitive items are exactly one user turn.
 10. many_shot items have >= 5 messages (>=2 demonstration pairs + final user turn).
 11. No empty content; no leading/trailing whitespace surprises in content.

Prints per-category counts and the SHA-256 of the suite file.

Usage:
  C:\\proj71\\env\\Scripts\\python.exe C:\\proj71\\scripts\\validate_redteam.py
  C:\\proj71\\env\\Scripts\\python.exe C:\\proj71\\scripts\\validate_redteam.py --suite <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO / "data" / "redteam" / "redteam_suite.jsonl"

ALLOWED_CATEGORIES = {
    "prefilling",
    "persona",
    "many_shot",
    "crisis_adjacent",
    "benign_sensitive",
}
REQUIRED_FIELDS = {
    "id",
    "category",
    "messages",
    "attack_note",
    "expected_safe_behaviour",
}
OPTIONAL_FIELDS = {"source_or_technique", "citation"}
ID_RE = re.compile(r"^(?P<cat>[a-z_]+)_(?P<num>\d{3})$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(suite_path: Path) -> int:
    errors: list[str] = []
    records: list[dict] = []

    raw = suite_path.read_text(encoding="utf-8")
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            errors.append(f"line {lineno}: blank line")
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: not valid JSON ({exc})")

    seen_ids: set[str] = set()

    for i, rec in enumerate(records, start=1):
        rid = rec.get("id", f"<no id, line {i}>")

        missing = REQUIRED_FIELDS - set(rec)
        if missing:
            errors.append(f"{rid}: missing required fields {sorted(missing)}")
        extra = set(rec) - REQUIRED_FIELDS - OPTIONAL_FIELDS
        if extra:
            errors.append(f"{rid}: unexpected fields {sorted(extra)}")

        cat = rec.get("category")
        if cat not in ALLOWED_CATEGORIES:
            errors.append(f"{rid}: category {cat!r} not in allowed set")

        m = ID_RE.match(str(rec.get("id", "")))
        if not m:
            errors.append(f"{rid}: id does not match <category>_<3 digits>")
        elif m.group("cat") != cat:
            errors.append(f"{rid}: id prefix does not match category {cat!r}")

        if rec.get("id") in seen_ids:
            errors.append(f"{rid}: duplicate id")
        seen_ids.add(rec.get("id"))

        for field in ("attack_note", "expected_safe_behaviour"):
            val = rec.get(field)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{rid}: {field} empty or not a string")
            elif "\n" in val:
                errors.append(f"{rid}: {field} must be a single line")

        msgs = rec.get("messages")
        if not isinstance(msgs, list) or not msgs:
            errors.append(f"{rid}: messages missing or empty")
            continue

        roles = []
        for j, msg in enumerate(msgs):
            if not isinstance(msg, dict) or set(msg) != {"role", "content"}:
                errors.append(f"{rid}: message {j} must have exactly role+content")
                continue
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                errors.append(f"{rid}: message {j} is a SYSTEM message (forbidden)")
            elif role not in ("user", "assistant"):
                errors.append(f"{rid}: message {j} has invalid role {role!r}")
            roles.append(role)
            if not isinstance(content, str) or not content.strip():
                errors.append(f"{rid}: message {j} content empty")

        if roles and roles[0] != "user":
            errors.append(f"{rid}: first message must be from user, got {roles[0]!r}")
        for a, b in zip(roles, roles[1:]):
            if a == b:
                errors.append(f"{rid}: consecutive {a!r} turns (must alternate)")
                break

        if cat == "prefilling":
            if roles[-1] != "assistant":
                errors.append(f"{rid}: prefilling item must END with an assistant turn")
        else:
            if roles[-1] != "user":
                errors.append(f"{rid}: {cat} item must END with a user turn")

        if cat == "benign_sensitive" and roles != ["user"]:
            errors.append(f"{rid}: benign_sensitive must be exactly one user turn, got {roles}")

        if cat == "many_shot" and len(msgs) < 5:
            errors.append(f"{rid}: many_shot needs >=5 messages, got {len(msgs)}")

    counts = Counter(r.get("category") for r in records)

    print(f"suite: {suite_path}")
    print(f"items: {len(records)}")
    print("per-category counts:")
    for cat in ("prefilling", "persona", "many_shot", "crisis_adjacent", "benign_sensitive"):
        print(f"  {cat:<18} {counts.get(cat, 0)}")
    other = {k: v for k, v in counts.items() if k not in ALLOWED_CATEGORIES}
    if other:
        print(f"  UNEXPECTED CATEGORIES: {other}")
    print(f"SHA-256: {sha256(suite_path)}")

    if errors:
        print(f"\nVALIDATION FAILED with {len(errors)} error(s):")
        for e in errors[:100]:
            print(f"  - {e}")
        if len(errors) > 100:
            print(f"  ... and {len(errors) - 100} more")
        return 1

    print("\nVALIDATION PASSED")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    args = ap.parse_args()
    raise SystemExit(validate(Path(args.suite)))


if __name__ == "__main__":
    main()
