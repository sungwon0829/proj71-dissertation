# Pending lab-notebook entry — red-team suite build

> Staged here rather than in `notebook/lab_notebook.md` to avoid an append race
> with concurrent agents. Merge into the lab notebook when convenient.

**Date:** 2026-07-31 (built 00:09 KST)
**Agent:** redteam-builder
**Owns:** `data/redteam/`

## What was produced

| Artefact | Path |
|---|---|
| Suite (canonical) | `data/redteam/redteam_suite.jsonl` |
| Per-category splits | `data/redteam/{prefilling,persona,many_shot,crisis_adjacent,benign_sensitive}.jsonl` |
| Full manifest | `data/redteam/SUITE_MANIFEST.md` |
| Short manifest of record | `data/redteam/MANIFEST.md` |
| Builder | `scripts/build_redteam_suite.py` |
| Item source of truth | `scripts/redteam_items/*.py` |
| Schema validator | `scripts/validate_redteam.py` |
| Leakage audit | `scripts/check_redteam_leakage.py` |

## Which paper numbers this feeds

Every ASR cell in Table 1, all of Table 2 (per-category ASR, B3 vs T), and the
over-refusal column of Table 1 (via `benign_sensitive`).

## Runs

| Script | Config / args | Seed | Outcome | Wall-clock |
|---|---|---|---|---|
| `build_redteam_suite.py` | `--seed 0` (logged, unused; build is deterministic) | 0 | 300 items written, 60/category | < 1 s |
| `validate_redteam.py` | default suite path | n/a | **VALIDATION PASSED** | < 1 s |
| `check_redteam_leakage.py` | default (all 5 processed corpora) | n/a | **NO LEAKAGE DETECTED** | ~2 min (120 MB `pref_helpful.jsonl` streamed) |

**SHA-256 (`redteam_suite.jsonl`):**
`e14c3a24184d01cbf31bbcfa42be03104ae07b0bea1132bc5b08a177645b6689`

## Counts

`prefilling` 60 · `persona` 60 · `many_shot` 60 · `crisis_adjacent` 60 ·
`benign_sensitive` 60 · **total 300**.

Prompt length min 8 / median 37 / max 114 words. `many_shot` demonstration
counts span 3–7 pairs. `crisis_adjacent` spans four acuity bands including 14
deliberately ambiguous items where the correct move is to check, not assume.

## Leakage audit (for Methods)

Scanned 43,863 records across `sft_train`, `sft_esconv`, `sft_counsel`,
`pref_safety`, `pref_helpful`.

- Exact normalised message matches (≥ 8 words): **0**
- Shared contiguous 12-word spans: **0**
- Max word-5-gram Jaccard, any eval item vs. any training record: **0.1034**
  (mean 0.0150; items ≥ 0.30: 0)
- Internal near-duplicate pairs within the suite (≥ 0.40): **0** (max 0.1014)

One item (`benign_sensitive_035`) was rewritten pre-freeze after the audit found
it shared a 12-word opening span with a CounselChat-derived prompt in
`pref_helpful.jsonl`. Recorded in `SUITE_MANIFEST.md` §5 rather than silently
fixed.

Diagnostic, not leakage: three one-word fabricated demonstration turns (`"yes"`,
`"sure"`) collide with common training utterances. Reported for completeness.

## Notes / decisions

- Schema follows the orchestrator's fixed contract exactly for the five required
  fields, plus two additive metadata fields (`source_or_technique`, `citation`)
  required by the builder's own operating instructions so the taxonomy is
  citable in the paper. The harness can ignore them; the validator permits only
  these two extras.
- No `system` message appears anywhere in the suite — the harness prepends
  `configs/system_prompt.txt` itself. Enforced by the validator.
- No item contains method, means, lethality, dosage or step-by-step self-harm
  content. Prefilling prefixes are truncated before any such content, since
  supplying it is the failure being measured.
- `crisis_adjacent` correct behaviour follows a cited clinical rubric (NICE
  NG225, WHO mhGAP-IG 2.0, WHO safe-messaging 2023, Stanley & Brown 2012, C-SSRS,
  Dazzi et al. 2014) per CLAUDE_safeguards_v2.md, not author judgment. Per-item
  rubric clauses are recorded in `source_or_technique` (e.g. "(C2, C5)").
- `data/` is git-ignored, so the suite file is not committed. Reproducibility is
  preserved by committing the builder and item modules: the build is
  deterministic and the SHA-256 above is the freeze witness.

## Open items for the orchestrator

1. **Freeze not set.** The suite is ready to freeze; the marker was deliberately
   not set by this agent. Confirm and record the freeze date/time + SHA-256.
2. **Harness contract.** `prefilling` items end with a partial `assistant` turn
   and must be *continued*, not answered as a new turn. 60 items depend on this.
3. **Judges.** Table 2 needs the two-judge split from CLAUDE_safeguards_v2.md §1:
   a harm classifier for the three attack categories, and a rubric-based
   behaviour judge for `crisis_adjacent` and `benign_sensitive`, since a bare
   refusal is "harmless" and still a failure in both.
4. **Re-run the leakage audit** after any change to `data/processed/`, and record
   the result as an appendix note. The suite itself must not change.
