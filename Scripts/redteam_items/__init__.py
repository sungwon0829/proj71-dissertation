"""Source-of-truth item definitions for the Project 71 frozen red-team suite.

One module per category. Each module exposes ITEMS: list[dict] with keys
  messages, attack_note, expected_safe_behaviour, source_or_technique, citation

IDs are NOT stored here -- they are assigned deterministically by
scripts/build_redteam_suite.py from list order, so the mapping id -> item is
stable and auditable.

THIS DATA IS EVALUATION-ONLY. It must never enter any training set.
"""
