# Pending lab notebook entry — Task 1 verdict: B1 brevity is a faithful arm property, not a defect

**For the maintainer to merge into `notebook\lab_notebook.md`** (written to a pending file
rather than appended directly, out of caution — the concurrent-write risk flagged for the
val-loss task appears to still apply: `lab_notebook.md` already shows edits from at least
one other track, e.g. the "Observed B1 behaviours" section, since I last read it).

## Verdict: FAITHFUL ARM PROPERTY, not a training defect. Do not retrain.

This corroborates and quantifies what `notebook\preregistration.md` §6/§8 and
`lab_notebook.md` § "Observed B1 behaviours" already state (B1's terseness is "an expected
property of the arm, not a failure"). The evidence below is new: a direct comparison of the
SFT corpus's assistant-turn length distribution against B1's generation lengths, plus a
checkpoint-290-vs-435 comparison to rule out a checkpoint-selection artifact.

## Evidence

### 1. Corpus length distribution vs. B1 generation length (the core check)

Character-length distribution of every assistant turn in the SFT training data
(`data\processed\sft_esconv.jsonl` + `sft_counsel.jsonl`, n=11,586 assistant-turn instances
total):

| Source | n turns | median | mean | p25 | p75 | p95 |
|---|---|---|---|---|---|---|
| ESConv | 10,191 | **95** | 117 | 55 | 151 | 290 |
| CounselChat | 1,395 | 827 | 1,016 | 552 | 1,270 | 2,382 |
| Combined | 11,586 | 107 | 225 | 60 | 190 | 954 |

**ESConv supplies 88% of all assistant-turn *instances* seen in training** (10,191 of
11,586), even though it contributes a roughly balanced ~47% of total assistant *token*
volume (293,097 CounselChat vs 264,104 ESConv tokens, per the earlier token-stats
lab-notebook entry) — ESConv turns are short but numerous (910 multi-turn dialogues,
~11.2 assistant turns/dialogue), CounselChat turns are long but singular (1,395 one-shot
Q&A rows). Per-token assistant-only loss balances token *mass* roughly evenly across the
two sources, but the *frequency* with which the model sees "a turn ends here, emit
`<|im_end|>`" is dominated 88:12 by short ESConv turns.

**Comparison to the reported red-team generation stats:**
- B1 overall median = **97 chars** → sits almost exactly on the **ESConv median (95
  chars)**, and below the ESConv p75 (151). 51.5% of ESConv turns are ≤97 chars.
- B1 crisis_adjacent median = **73 chars** → within the ESConv distribution's lower-middle
  range; 36.6% of ESConv turns are ≤73 chars (≈ESConv's 35th–37th percentile — short but
  well inside the observed range, not a tail/outlier value).
- Base Qwen's much larger median (1,118 chars, per the coordinator's report) is consistent
  with the un-tuned instruct model's default long, structured, numbered-list register,
  which is exactly what SFT on ESConv overwrote (see own sanity-check evidence below).

**Reading:** B1 is not inventing a new, anomalously short register. It is reproducing the
dominant per-turn length statistics of the corpus it was trained on — specifically the
ESConv component, which structurally outnumbers CounselChat's long-form turns 7:1 by
instance count. This is exactly what assistant-only SFT on a turn-taking dialogue corpus is
expected to do.

### 2. checkpoint-290 (selected) vs checkpoint-435: not meaningfully different

Re-ran the Gate 2 sanity-check prompts (same 8 ESConv-validation prompts, same seed 42,
identical generation params) through checkpoint-435 for comparison (`--checkpoint
checkpoint-435`, appended to `results\B1_sft_seed42\sanity_generations.txt`, never
overwriting the checkpoint-290 run already there):

| Checkpoint | median chars | mean chars | min | max |
|---|---|---|---|---|
| checkpoint-290 (selected) | 80.0 | 98.8 | 31 | 229 |
| checkpoint-435 | 83.0 | 94.4 | 31 | 186 |
| base (no adapter), same run | 687.5 | 722.9 | 212 | 1,384 |

n=8 is too small for a formal test, but the two checkpoints are visibly indistinguishable
in length (80 vs 83 median chars) while both are ~8.5x shorter than base (687.5 median).
**The Gate 1 selection criterion (val loss) did not select for terseness** — checkpoint-435
is equally terse — so this is not an artifact of *which* epoch was picked; it is present
throughout SFT training from at least epoch 2 onward. (Selection itself is not being
revisited; this is reported per the coordinator's request, not as grounds to reselect.)

### 3. Ruled out: generation-config / template artifacts

- **Not `max_new_tokens` truncation.** Every B1 output (31–229 chars ≈ 8–60 tokens) is far
  under the 256-token budget; generation stopped well before the limit.
- **Not premature/mid-token EOS.** Inspected all 16 B1 completions (checkpoint-290 +
  checkpoint-435) verbatim in `sanity_generations.txt`: every one ends on a complete
  sentence with terminal punctuation (e.g. "...get you back on track with your life.",
  "...how may I assist you today?"), never mid-word or mid-clause. This is a real,
  grammatically clean stop, not a corrupted/garbled cutoff.
- **Not the custom chat-template's trailing-newline handling.** The masking-verification
  gate (Gate B, `train_sft.py`) already confirmed byte-identical rendering against the
  stock Qwen2.5 template and correct span-level masking; the only change the custom
  template makes is *which* tokens are scored (assistant content + `<|im_end|>` + one
  trailing `\n`), not their content or count. There is no mechanism by which that template
  could bias *how many* content tokens the model emits before choosing to stop.
- **Separately:** the coordinator's repetition-penalty finding (44/300 items looping at
  `repetition_penalty=1.0`, being fixed at 1.05) is an unrelated decoding-config bug that
  makes some outputs pathologically *long* via repeats — it does not explain, and is not
  needed to explain, the median-length pattern above, which holds independent of it.

## Substantive point worth flagging even though this is not a defect

The brevity is faithful and explainable, but it is separately worth noting for
Discussion/Limitations: "I'm so sorry to hear that. I'm here to listen." is directionally
appropriate (empathetic, non-dismissive, not a refusal) but, on `crisis_adjacent` items
specifically, is short of what a documented crisis-response rubric (per §8 of
`preregistration.md`: NICE NG225 / WHO mhGAP-IG 2.0 / WHO safe-messaging 2023 / Stanley &
Brown 2012 / C-SSRS) would typically expect — e.g. no safety question, no hotline/resource
mention, no explicit continuation of engagement. That gap is exactly the sort of thing the
Behaviour judge (Qwen2.5-7B + the documented rubric, per preregistration §4) is positioned
to catch on `crisis_adjacent` outcomes, and exactly the sort of thing safety-pair "chosen"
responses in T's DPO data could plausibly move, in a way generic helpfulness pairs (B2)
would not. This is context for Discussion, not a re-scope of Task 2 below.

## Provenance

All generation-length numbers above are drawn from `results\B1_sft_seed42\
sanity_generations.txt` (checkpoint-290 run + checkpoint-435 comparison run appended to the
same file, never overwritten) and freshly computed corpus statistics over
`data\processed\sft_esconv.jsonl` / `sft_counsel.jsonl` (read-only, not modified). The
300-item red-team median figures (97 / 73 chars) and the base-model 1,118-char median are
as reported by the coordinator from the eval-harness track; not independently recomputed
here (no access to the harness's generation file from this thread).
