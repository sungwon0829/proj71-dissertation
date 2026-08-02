"""Backfill `cli_invocation` into run manifests written before the field existed.

Context. `cli_invocation` was added to run manifests on 2026-08-01, so runs completed
before that (B1 v1/v2, B2 v1/v2/v3) carry `sampled_data_sha256` and
`consumed_data_sha256` but no record of the command that produced them. This script adds
the field for those runs only.

PROVENANCE DISCIPLINE. These invocations were NOT recorded at run time. They are
reconstructed from the lab notebook, the run's output directory name, and the config the
run used. They are therefore written with `"reconstructed": true` and an explicit note,
exactly as the training-data hash backfill was. A reconstructed invocation shows what
would reproduce the run today; it is not evidence of what was typed. Runs that recorded
the field live are never touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = r"C:\proj71\env\Scripts\python.exe"

# (manifest path, argv, note). Reconstructed from notebook entries and output dir names.
BACKFILL = [
    (
        "B1_sft_seed42/sft_run_manifest.json",
        ["scripts/train_sft.py", "--config", "configs/sft_lora.yaml", "--seed", "42"],
        "B1 v1 (VOID - CounselChat therapist identities present in SFT data). Default "
        "output dir. Reconstructed; this run's training data is itself unrecoverable, so "
        "the invocation cannot be re-executed to reproduce it.",
    ),
    (
        "B1_sft_seed42_v2/sft_run_manifest.json",
        ["scripts/train_sft.py", "--config", "configs/sft_lora.yaml", "--seed", "42",
         "--output_dir", "results/B1_sft_seed42_v2"],
        "B1 v2 (current B1; checkpoint-290 selected on held-out validation loss). "
        "Reconstructed.",
    ),
    (
        "B2_dpo_seed1/dpo_data_manifest.json",
        ["scripts/train_dpo.py", "--config", "configs/dpo_b2.yaml", "--seed", "1"],
        "B2 v1 (VOID - built on the pre-scrub B1). Default output dir. Reconstructed. "
        "NOTE: dpo_b2.yaml has since changed (beta 0.1, then 0.3, then 0.1 plus the "
        "[sigmoid, sft] NLL anchor), so re-running this argv today does NOT reproduce "
        "this run.",
    ),
    (
        "B2_dpo_seed1_v2/dpo_data_manifest.json",
        ["scripts/train_dpo.py", "--config", "configs/dpo_b2.yaml", "--seed", "1",
         "--output_dir", "results/B2_dpo_seed1_v2"],
        "B2 v2 (SUPERSEDED - beta 0.1, degenerate: 16.0% strict / 21.3% loose / 23.0% "
        "cap-hit). Reconstructed. NOTE: dpo_b2.yaml has since changed; re-running this "
        "argv today does NOT reproduce this run.",
    ),
    (
        "B2_dpo_seed1_v3/dpo_data_manifest.json",
        ["scripts/train_dpo.py", "--config", "configs/dpo_b2.yaml", "--seed", "1",
         "--output_dir", "results/B2_dpo_seed1_v3"],
        "B2 v3 (SUPERSEDED - beta 0.3, degenerate: 14.3% strict / 23.7% loose / 23.7% "
        "cap-hit). Reconstructed. NOTE: dpo_b2.yaml has since changed; re-running this "
        "argv today does NOT reproduce this run.",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    for rel, argv, note in BACKFILL:
        path = REPO / "results" / rel
        if not path.exists():
            print(f"[MISSING]  {rel}")
            continue
        man = json.loads(path.read_text(encoding="utf-8"))
        if "cli_invocation" in man:
            existing = man["cli_invocation"]
            recon = isinstance(existing, dict) and existing.get("reconstructed")
            print(f"[SKIP]     {rel} - already has cli_invocation "
                  f"({'reconstructed' if recon else 'RECORDED LIVE - not touching'})")
            continue
        man["cli_invocation"] = {
            "argv": argv,
            "resolved_seed": int(argv[argv.index("--seed") + 1]),
            "python_executable": PY,
            "reconstructed": True,
            "reconstructed_on": "2026-08-02",
            "reconstruction_note": note,
            "_warning": (
                "NOT recorded at run time. Reconstructed from the lab notebook, the output "
                "directory name and the config this run used. Shows what would reproduce "
                "the run today under the config AS IT STOOD THEN; it is not evidence of "
                "the command actually typed. Runs from 2026-08-01 onward record this field "
                "live and carry no 'reconstructed' flag."
            ),
        }
        if args.apply:
            path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[WRITTEN]  {rel}")
        else:
            print(f"[DRY-RUN]  {rel} -> would add reconstructed cli_invocation")

    print("\nDone." if args.apply else "\nDry run only. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
