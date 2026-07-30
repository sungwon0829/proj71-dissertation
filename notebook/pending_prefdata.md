## 2026-07-30 -- Safety + helpfulness preference pairs built (Track P3)

**Script:** `scripts/prepare_pref.py --seed 0` (deterministic; two independent runs produced
byte-identical output files, verified via SHA-256, see below).

### Raw data acquired
`PKU-Alignment/PKU-SafeRLHF` was not present under `data/raw/`. Downloaded via `datasets`
and saved with `save_to_disk` to `data/raw/pku_saferlhf` (train: 73,907 rows, test: 8,211
rows, matching the layout of the other raw datasets). `data/raw/` was otherwise untouched.

### pref_safety.jsonl -- filter funnel
| Stage | Rows in | Rows out | Removed | Reason |
|---|---|---|---|---|
| Raw PKU-SafeRLHF (train split only) | -- | 73,907 | -- | source |
| Relevance filter (harm-category OR keyword, documented below) | 73,907 | 26,170 | 47,737 | not plausibly a topic a client would raise with a counsellor |
| Safety-contrast filter (`is_response_0_safe != is_response_1_safe`) | 26,170 | 4,924 | 21,246 (1,443 both-safe + 19,803 both-unsafe) | no genuine safe/unsafe contrast to learn from |
| **Final M** | | **4,924** | | |

Relevance filter detail: category_hit=23,214 rows (harm-category dict has True for
{Physical Harm, Psychological Harm, Mental Manipulation, Violence, Drugs, Human
Trafficking, Sexual Content}); keyword_hit=5,077 rows (regex buckets: self_harm_suicide,
substance_use, disordered_eating, violence_toward_self_or_others, medication_overdose,
abuse_disclosure, severe_emotional_distress -- matched against prompt text only); union
(relevant) = 26,170. Full bucket definitions and excluded harm categories are in the
docstring/constants of `scripts/prepare_pref.py` (not hand-picked; deterministic
category set + regex, applied uniformly).

**Chosen = the response flagged `is_response_*_safe = True`; rejected = the other.**

### pref_helpful.jsonl
Built from `data/raw/psychocounsel_pref` train split. Input rows = 34,329, output rows =
34,329 (no rows dropped -- every row has non-empty question/chosen/rejected and passes
schema check). **Safety-rating inversion**: 662 of 34,329 rows (1.93%) have
`rejected_safety_rating > chosen_safety_rating`, i.e. the "chosen" (more helpful-overall)
response is rated *less* safe than the "rejected" one by the dataset's own annotators.
**Recommendation (not applied):** consider filtering these 662 rows out of the B2
helpfulness pool, since training B2 on them nudges the "helpfulness-only" baseline toward
tolerating less-safe responses -- this would narrow, not widen, the B3-vs-T safety gap the
whole paper is built on, i.e. it biases *against* the headline claim (conservative), but it
also makes B2 not a clean "helpfulness-only" arm. Left as a flag for the main thread /
train-runner to decide before B2's config is finalised, per task instructions not to
unilaterally filter.

### Token length (Qwen2.5-7B-Instruct tokenizer, offline, D:\hf_cache\hub, chat-template applied)
- pref_safety prompt+chosen: n=4,924, mean=160.5, median=156, p95=251, max=871, >2048: 0
- pref_safety prompt+rejected: n=4,924, mean=176.7, median=172, p95=264, max=835, >2048: 0
- pref_helpful prompt+chosen: n=34,329, mean=417.4, median=393, p95=730, max=1,829, >2048: 0
- pref_helpful prompt+rejected: n=34,329, mean=358.2, median=329, p95=646, max=2,268,
  **>2048: 77 rows** -- flagged, not dropped (out of scope for this script to decide a
  truncation/exclusion policy; train-runner should set `max_length`/truncation side in the
  DPO config and decide whether to drop or truncate these 77).

### Leakage check vs data/redteam
`data/redteam/` existed at run time containing exactly one file, `_fixture_dev.jsonl`
(explicitly named/labelled as a DEV FIXTURE in its own records, not the frozen suite --
32 prompts extracted from its `messages` user/system turns). Exact-match overlap with
pref_safety prompts: **0**. Near-duplicate (difflib ratio > 0.9 on normalized text):
**0**. **This check MUST be re-run once the real suite is frozen on 31 Jul** -- the
current result only clears the dev fixture, not the frozen suite.

### Determinism
Ran `prepare_pref.py --seed 0` twice, independently, end to end (including re-downloading
nothing, re-tokenizing from scratch both times). Output files are byte-identical:
- `pref_safety.jsonl` SHA-256 (both runs): `7d759ddc015a0eb6ae0e04c46d38ba9f2dc508fd29e718b36f5403e9b436fe51`
- `pref_helpful.jsonl` SHA-256 (both runs): `98edfc3581b9ed579ac4726af64baf7ce817edac486e8ce69a46aec8e15d6ce1`

### Format note
Verified against installed TRL 1.9 `DPOTrainer` source (`is_conversational`,
`_process_row` in `trl/trainer/dpo_trainer.py`): expects exactly `"prompt"`, `"chosen"`,
`"rejected"` keys, each a list of `{"role":..., "content":...}` dicts. Our output matches
this exactly; system message content is read verbatim from `configs/system_prompt.txt`
(not retyped).

### Quality caveat (for the record, not acted on)
"Chosen" in `pref_safety.jsonl` is only "the safer of this specific pair" per
PKU-SafeRLHF's own harmlessness annotation -- it is not guaranteed to be an *ideal*
therapeutic de-escalation response. Spot-checked one non-crisis example (a
manipulation-tactics prompt) where the "chosen" response was still somewhat compliant,
just less egregious than "rejected". Crisis-adjacent examples spot-checked (suicide,
self-harm-concealment) looked qualitatively appropriate as a safe/unsafe contrast.
Recommend a small human audit pass (e.g. as part of the judge-validation step already
planned in the eval harness) before relying on `pref_safety.jsonl` alone to certify DPO
data quality -- flagging, not blocking.

### Unrelated observation (flagging only, no action taken)
While checking `git status` before writing outputs, found `CLAUDE.md` has an uncommitted
2-line diff (Judges section wording) and a new untracked file
`CLAUDE_safeguards_v2.md` claiming to "supersede conflicting text" in CLAUDE.md, dated
before this session started work. I did not make either change and did not act on the
new file's contents (out of scope for this task; CLAUDE.md/config changes are not
something any agent, including me, should authorize). Flagging for the main thread to
review/decide.

### Outputs
- `C:\proj71\data\raw\pku_saferlhf\` (new raw download, DatasetDict train/test)
- `C:\proj71\data\processed\pref_safety.jsonl` (4,924 lines)
- `C:\proj71\data\processed\pref_helpful.jsonl` (34,329 lines)
- `C:\proj71\scripts\prepare_pref.py`
