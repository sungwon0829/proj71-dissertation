# Pending notebook entry — training-data hash verification (2026-08-01)

**Purpose (owner's request, relayed by the coordinator):** prove the seed fully determines
the training data, now rather than in September.

## 1. The two hash definitions (exact, machine-checkable)

Implemented once in `scripts\train_sft.py` (`HASH_DEFINITION`, `canonical_record_repr()`,
`hash_ordered_records()`), imported into `scripts\train_dpo.py` -- both scripts use the
identical definition, not two similar-but-different ones.

**`HASH_DEFINITION`** (also written verbatim into every run manifest):
> SHA-256 of the ordered sequence of records (order matters -- this hashes a sequence, not
> a set). Each record is serialised as canonical JSON: `json.dumps(record, sort_keys=True,
> ensure_ascii=False, separators=(',', ':'))` -- i.e. keys sorted recursively, no extra
> whitespace, non-ASCII characters kept literal (not `\uXXXX`-escaped). Records are
> newline-joined (`'\n'.join(...)`) in dataset order, encoded as UTF-8, with NO trailing
> newline after the last record. `hashlib.sha256(...).hexdigest()` of that byte string is
> the reported hash.

**Two hashes, both defined, both recorded, per the coordinator's instruction:**
- **`sampled_data_sha256`** -- the pairs (DPO) or examples (SFT) our own sampling/loading
  step produced, in order, **before** TRL touches them. This is the one the seed should
  fully determine.
- **`consumed_data_sha256`** -- the dataset actually handed to the trainer **after** TRL's
  own filtering, obtained from `trainer.train_dataset` post-`__init__` (DPO:
  `DPOTrainer._prepare_dataset`; SFT: `SFTTrainer._prepare_dataset`), which both preserve
  the original `prompt`/`chosen`/`rejected` (DPO) or `messages` (SFT) columns alongside
  their own added tokenized columns -- verified by reading the installed trl 1.9.0 source
  (not assumed), and empirically confirmed (`trainer.train_dataset.column_names` inspected
  directly). Hashed with the exact same `hash_ordered_records()` function, so the two
  hashes are directly, meaningfully comparable.
- **`n_dropped_by_trl`** recorded either way (`len(sampled) - len(consumed)`), even when
  the hashes match, per instruction.

## 2. The subtlety: what "Dropping fully truncated/masked examples" actually does

Read the installed trl 1.9.0 source directly (`DPOTrainer._prepare_dataset`,
`SFTTrainer._prepare_dataset`) rather than assuming:
- **DPO**: drops any row whose **prompt alone** (not prompt+completion) already reaches
  `max_length` under `truncation_mode="keep_start"` -- with `keep_start`, such a row would
  have every completion token truncated away, leaving no learning signal. TRL tokenizes
  prompt-alone and prompt+completion **separately** (not via one combined
  `apply_chat_template` call the way our own `filter_by_max_length`/
  `assert_zero_truncation_dpo` do it), so its measured length is not guaranteed
  byte-identical to ours -- a real (if narrow) way our own pre-filter and TRL's internal
  one could disagree, hence measuring rather than assuming.
- **SFT**: drops any row left **fully masked** (`all(label == -100)`) after truncation --
  can only happen if the prompt/scaffolding alone already fills `max_seq_length`.
- **The progress bar reaching 100% does NOT mean anything was dropped** -- `.filter()`
  always iterates every input row to evaluate the keep/drop predicate; the bar shows rows
  *checked*, not rows *removed*. The actual drop count must be measured from the resulting
  dataset's length, which is what `n_dropped_by_trl` does.

## 3. B2 seed 1 verification (the owner's specific ask) -- **PASS**

Re-ran the sampling step alone for B2 seed 1, **in a fresh process**, from
`configs\dpo_b2.yaml` and `--seed 1`, **without constructing a trainer or loading the
model**:
```
sampled_data_sha256 (fresh re-sample) = 4ddb48309740eb5459c08740b54452311dafc995c9faa88cf8ae9477fa3bf705
```
Then, separately, **also without loading the model** (`DPOTrainer._prepare_dataset` was
called standalone on a dummy `self` -- verified safe: its only `self.` reference is inside
the non-conversational EOS-adding branch, which our conversational prompt/chosen/rejected
data never enters):
```
consumed_data_sha256 (same fresh sample, post-TRL-filter) = 4ddb48309740eb5459c08740b54452311dafc995c9faa88cf8ae9477fa3bf705
n_dropped_by_trl = 0  (of 19,924)
```
**sampled == consumed**: TRL's "Dropping fully truncated examples" step drops nothing for
B2's data -- our own pre-filter is already sufficient, confirmed by measurement.

**Compared against the manifest of the completed run:** B2 seed 1's most recent completed
run (v3, beta=0.3) did not itself record `sampled_data_sha256` (it ran before this feature
was added), so the comparison was made against the artefact that WAS recorded live at run
time: `sampled_helpful_pair_ids` (19,924 per-pair content-hash IDs, in order, using the
pre-existing `pair_id()` function). **Exact match, in order, all 19,924 IDs.**

**Additional, unplanned corroboration**: this same live-recorded `sampled_helpful_pair_ids`
list was compared across **all three** B2-seed-1 runs (v1/void, v2/superseded, v3/current)
-- **all three are byte-identical to each other and to the fresh re-sample**, despite
spanning the CounselChat-signature scrub (which changed B1, not B2's DPO pools) and the
beta 0.1→0.3 change (a training hyperparameter, not a data-sampling one). This is exactly
what should happen if the seed fully determines the sampled data, and it is what was
found.

**Verdict: PASS. The seed fully determines the sampled data for B2**, checked two
independent ways (whole-sequence canonical hash match against a fresh re-sample; per-pair
ID list match against three separate real historical runs).

## 4. Backfill table

| Run | `sampled_data_sha256` | `consumed_data_sha256` | `n_dropped_by_trl` | `reconstructed` |
|---|---|---|---|---|
| B1 v1 (VOID) | **UNRECOVERABLE** | UNRECOVERABLE | unknown | n/a -- see note below |
| B1 v2 (clean, current) | `e3023863b0bb9d4e6894390f198a067c4f9e6fd9b1a3da3bb3efdf42af9098a5` | same (identical) | 0 | **true** |
| B2 seed1 v1 (VOID) | `4ddb48309740eb5459c08740b54452311dafc995c9faa88cf8ae9477fa3bf705` | same (identical) | 0 | **true** |
| B2 seed1 v2 (superseded, beta=0.1) | `4ddb48309740eb5459c08740b54452311dafc995c9faa88cf8ae9477fa3bf705` | same (identical) | 0 | **true** |
| B2 seed1 v3 (current, beta=0.3) | `4ddb48309740eb5459c08740b54452311dafc995c9faa88cf8ae9477fa3bf705` | same (identical) | 0 | **true** |

**B1 v1 is genuinely unrecoverable, stated plainly, not papered over.** That run consumed
the PRE-SCRUB `data\processed\sft_train.jsonl` (containing the CounselChat
therapist-identity leak). The scrub overwrote that file in place; there is no backup
(`data\` is not git-tracked, per `CLAUDE.md`). The only surviving trace is an
**abbreviated** SHA-256 fingerprint recorded in `notebook\lab_notebook.md`
(`52A7074D...20DA3DE3`, elided with "...") -- and that is (a) itself incomplete (the full
64-hex-char hash was never written down anywhere, so even this fallback is partial) and
(b) a **different hash definition** (whole-file-bytes SHA-256, computed with
`Get-FileHash`/`hashlib.sha256(file_bytes)`, not this project's canonical
per-record/newline-joined/sorted-keys definition) -- **not directly comparable** to
`sampled_data_sha256` even if it were complete. This is stated as a limitation, not
silently worked around.

**Every `reconstructed: true` value above is corroborated beyond a bare "recompute and
trust it" reconstruction**, specifically for the reasons given in §3: the B2 rows are
independently confirmed against three separate real runs' own live-recorded
`sampled_helpful_pair_ids`; the B1 v2 row is confirmed against the still-unchanged, hash-
verified `data\processed\sft_train.jsonl` (current SHA-256
`46E87A39F239982F9E9994535B4A44209617800AFFDEEBFEAEC0AF8922DBF662`, matching the
coordinator's own report and this agent's own earlier `Get-FileHash` check). This is
recorded explicitly because a `reconstructed: true` flag alone only proves reproducibility
*today* -- the additional corroboration is what makes these specific backfilled values
trustworthy as a proxy for what those historical runs actually consumed, and that
reasoning is written out here rather than asserted.

## 5. TRL drop counts -- the matched-volume implication

**Zero TRL-side drops observed everywhere checked** (B1 v2: 0/2305; B2 all three versions:
0/19,924). This means:
- Our own pre-filters (`assert_zero_truncation` for SFT, `filter_by_max_length` +
  `assert_zero_truncation_dpo` for DPO) are already sufficient -- TRL's internal filter is
  a genuine no-op for this project's data, not a silent source of divergence.
- **Matched volume is confirmed matched** for the data measured so far: B2 trains on
  exactly 19,924 pairs, with 0 silently dropped by TRL. T and T_ctrl have not yet been
  launched with this instrumentation; their `n_dropped_by_trl` will be recorded live (not
  backfilled) on their first real runs, and **must both be 0** (or, if not, must be
  **equal**) for the matched-volume claim in Methods to hold -- flagged here as the
  specific thing to check once T/T_ctrl run, per the coordinator's own framing ("B2 and T
  must drop the same number or matched volume is not matched").

## Manifests updated

- `results\B1_sft_seed42\sft_run_manifest.json` (new, backfilled, unrecoverable stated)
- `results\B1_sft_seed42_v2\sft_run_manifest.json` (new, backfilled, reconstructed)
- `results\B2_dpo_seed1\dpo_data_manifest.json` (updated in place, hash fields added)
- `results\B2_dpo_seed1_v2\dpo_data_manifest.json` (updated in place, hash fields added)
- `results\B2_dpo_seed1_v3\dpo_data_manifest.json` (updated in place, hash fields added)

Going forward (T, T_ctrl, and any future B2/B1 seeds), both hashes are written **live**,
in two stages (sampled-data fields before any GPU work; consumed-data fields after the
trainer is constructed), by `scripts\train_dpo.py`/`scripts\train_sft.py` automatically --
no further backfilling should be needed for new runs.
