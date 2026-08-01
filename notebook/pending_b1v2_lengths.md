# Pending notebook entry — B1 v2 response-length statistics (replaces stale v1 figures)

**For the coordinator to fold into `notebook\lab_notebook.md` / `notebook\preregistration.md`
§8 -- not edited directly by me, per instruction.**

## Why this was needed

The previously-quoted B1 brevity statistics ("median response 97 chars overall, 73 chars on
`crisis_adjacent`, versus base Qwen's 1118") were measured on **B1 v1**
(`results\B1_sft_seed42\checkpoint-290`), which is **VOID** (CounselChat therapist-identity
leak). B1's row in Table 1 comes from **v2**
(`results\B1_sft_seed42_v2\checkpoint-290`). Quoting v1 statistics next to v2 results would
be the stale-artefact error the harness now hard-fails on for adapters. Re-measured on v2
below, at **zero additional GPU cost** -- reused the generations already produced for the
repetition quantification (`results\B1_sft_seed42_v2\repetition_full_suite_generations.jsonl`,
n=300, full frozen suite, pinned decoding config `repetition_penalty=1.05`).

## Method

Character length of `response_full_turn` (the complete assistant turn, including any given
prefix for `prefilling` items) and, separately, `response_continuation` (excludes the given
prefix -- relevant only for `prefilling`'s 60 items; identical to `response_full_turn` for
every other category), computed directly from the raw generation JSONL files, not
re-generated. All three files compared below were verified to (a) match the frozen suite
hash `e14c3a24184d...5b6689` and (b) share the **identical pinned decoding config**
(`do_sample=false, repetition_penalty=1.05, max_new_tokens=512`) before being compared, so
the B0-vs-B1v1-vs-B1v2 comparison isolates the adapter, not a decoding-config confound:

| Arm | Source file | Adapter | Decoding rp |
|---|---|---|---|
| B0 | `results\b0_seed42\generations_realsuite_SUPERSEDED_pre_b1v2_regen.jsonl` | none (base) | 1.05 |
| B1 v1 (VOID) | `results\b1_seed42\generations_realsuite_SUPERSEDED_b1v1_signatures.jsonl` | `B1_sft_seed42/checkpoint-290` | 1.05 |
| B1 v2 (clean) | `results\B1_sft_seed42_v2\repetition_full_suite_generations.jsonl` | `B1_sft_seed42_v2/checkpoint-290` | 1.05 |

**On the "SUPERSEDED" filenames for B0 and B1 v1:** both are marked superseded by the
eval-harness track's batch-regeneration convention (B1 v1 for the signature leak; B0
because it predates the B1 v2 regeneration round), **not** because their own decoding
config is stale -- both were independently verified above to already be at the current
pinned `repetition_penalty=1.05`. B0 has no adapter, so it is unaffected by the B1 scrub
either way. Used as-is for this length comparison; not re-generated (per the "leave B0
pending rather than mixing configs" instruction -- config match was verified, not assumed,
so B0 is not left pending).

## Results -- `response_full_turn` (the complete output as a reader would see it)

| | B0 (base) | B1 v1 (VOID) | B1 v2 (clean) |
|---|---|---|---|
| **overall median** | 1165 | 114 | **110** |
| overall mean | 1224.8 | 318.8 | 298.0 |
| `prefilling` median | 742 | 143 | 144 |
| `persona` median | 1489 | 108 | 89 |
| `many_shot` median | 402 | 28 | 28 |
| **`crisis_adjacent` median** | 992 | 100 | **86** |
| `benign_sensitive` median | 2118 | 515 | 460 |

## Results -- `response_continuation` (prefix excluded; only differs from full_turn for `prefilling`)

| | B1 v1 (VOID) | B1 v2 (clean) |
|---|---|---|
| overall median | 93 | 85 |
| `prefilling` median | 62 | 66 |
| `crisis_adjacent` median | 100 | 86 (unchanged -- crisis_adjacent has no prefill items) |

## Direct answer to the question asked: did the scrub-and-retrain change B1's character?

**No, not materially.** B1 v1 and B1 v2 are close to each other on every category and both
are dramatically shorter than base (B0) on every category -- e.g. overall median 114 (v1)
vs 110 (v2) vs 1165 (B0); `crisis_adjacent` median 100 (v1) vs 86 (v2) vs 992 (B0). The
scrub removed 70 CounselChat answers' author signatures (a small, targeted edit -- see
`notebook\pending_scrub.md`) and did not change the SFT corpus's overall length
distribution enough to move B1's generation-length character. This is consistent with (and
does not require revising) the earlier "faithful arm property" analysis -- B1's brevity is
explained by the ESConv-dominated turn-length distribution of the training corpus, and that
corpus was materially unchanged by the scrub.

**The `crisis_adjacent` median specifically, for the Discussion argument** ("B1's crisis
failure mode is insufficient engagement, not unsafe content"): **B1 v2's `crisis_adjacent`
median is 86 characters** (`response_full_turn`; `response_continuation` is identical for
this category since it contains no prefill items), vs. B0's 992 -- an ~11.5x reduction,
even more pronounced than the v1 figure (100 chars, ~9.9x reduction). If anything the v2
retrain sharpens this Discussion point slightly, not weakens it.

## Discrepancy with the previously-quoted figures -- flagged honestly, not reconciled

The previously-quoted numbers (97 / 73 / 1118) do **not** exactly match either measurement
above (v1 `response_full_turn`: 114 / 100 / -; v1 `response_continuation`: 93 / 100 / -).
`response_continuation`'s overall figure (93) is close to the quoted 97 (4-char gap,
plausibly explained by a minor methodology or snapshot difference), which suggests the
original figure may have used `response_continuation` rather than `response_full_turn`.
**However, the `crisis_adjacent` figure (73) does not match under either field or file I
could locate** (my measurement gives 100 for v1 under both fields, since `crisis_adjacent`
has no prefill items to strip). I checked the other B1 generation files on disk as
candidate sources (a 20-item dev-fixture run at the old `repetition_penalty=1.0`, suite
`_fixture_dev.jsonl` -- median 220.5 overall, crisis n=4 median 380, does not match either)
and found no file that reproduces 73. **I could not trace the exact provenance of the
original 73-char figure within this session** and am not asserting my number is "more
correct" than theirs without knowing their method -- flagging this rather than silently
picking one. Recommend the paper-writer/coordinator check whether the original figure came
from a different suite snapshot, a different subset (e.g. excluding empty/near-empty
responses), or a different field, and reconcile against the number above
(**B1 v2 `crisis_adjacent` median = 86 chars, `response_full_turn`, n=60, full frozen
suite, pinned decoding config, hash-verified**) accordingly.

## Files referenced (all pre-existing, none regenerated for this task)

- `results\b0_seed42\generations_realsuite_SUPERSEDED_pre_b1v2_regen.jsonl`
- `results\b1_seed42\generations_realsuite_SUPERSEDED_b1v1_signatures.jsonl`
- `results\B1_sft_seed42_v2\repetition_full_suite_generations.jsonl`
