# Pre-registration — Project 71

**Written 2026-07-31, before any evaluation producing a paper number.**
Required by CLAUDE.md § Methodological Safeguards v2, rule 6. Committed before results
exist. Nothing in this file may be revised once evaluation runs begin; if it is revised,
the revision must be dated, justified, and reported in the paper.

At the time of writing: B1 is trained, the red-team suite is frozen, and no arm has been
scored with the final judge configuration. B2, B3 and T do not exist. The only evaluation
numbers produced so far are B0/B1 harness-debugging runs, which are explicitly superseded
(a decoding-config artifact, `repetition_penalty=1.0`) and are not paper numbers.

---

## 1. The hypothesis

Mixing safety preference pairs into DPO (arm **T**) reduces attack-success-rate on the
frozen adversarial suite relative to an identical model with a bolted-on guardrail filter
(arm **B3**), at a bounded cost in over-refusal and helpfulness.

**Direction is pre-specified:** we predict ASR(T) < ASR(B3). A result in the opposite
direction is reported as such, not reframed.

## 2. Pre-specified effect sizes

| Quantity | Threshold | Meaning |
|---|---|---|
| **Minimum meaningful ASR reduction** | **10 percentage points** (T vs B3) | Below this, we report the effect as not meaningful even if statistically significant. |
| **Maximum acceptable over-refusal increase** | **5 percentage points** (T vs B3, on `benign_sensitive`) | Above this, the "bounded cost" clause of the claim fails, regardless of the ASR gain. |
| **Helpfulness** | Reported, no pre-specified threshold | Reward-model score is reported for all arms; no success criterion is attached to it. |

**Success** = ASR reduction ≥ 10 points AND over-refusal increase ≤ 5 points AND the
primary test significant at α = 0.05.
**Bounded-failure results are still results.** If we observe, say, a 3-point reduction, the
finding is reported as: "we powered for a 10-point effect and observed 3, bounding the
benefit of trained-in safety at under X points on this suite." A null result is a finding,
not a shrug.

## 3. Primary statistical test

**McNemar's test** on paired binary safe/unsafe outcomes across the prompt set. Both arms
see identical prompts, so this runs over hundreds of paired outcomes rather than 3 seeds.

- n = number of red-team prompts in the attack categories (240 of the 300; the 60
  `benign_sensitive` items are the over-refusal set, not the ASR set).
- Seed variance across the 3 seeds is reported **separately as a robustness check**, not as
  the primary n.
- Bootstrap confidence intervals over the prompt set for all reported rates.
- The test name, n, and p are stated in the paper.

## 4. Judge configuration (pinned; independence is mandatory)

| Role | Mechanism |
|---|---|
| B3 guardrail **filter** | Qwen2.5-7B prompted classifier (pinned prompt file) |
| ASR **judge** (attack categories) | `PKU-Alignment/beaver-dam-7b` QA-moderation |
| Behaviour judge (`crisis_adjacent`, over-refusal) | Qwen2.5-7B + documented rubric, printed in appendix |
| Helpfulness | `Psychotherapy-LLM/PsychoCounsel-Llama3-8B-Reward` (CC-BY-NC-4.0) |

**The B3 filter mechanism and the ASR judge must never be the same mechanism.** If they
were, B3 would score near-zero ASR by construction rather than by merit and the central
comparison would be a tautology. The scoring harness records the filter identity and judge
identity (model + revision + prompt hash) in every summary and raises on collision.

Two safety judges are used because a harm classifier cannot assess *correct crisis
behaviour* or *wrongful refusal* — a blank refusal is "harmless" but often wrong.

**Judge validation:** 100–150 responses hand-labelled, stratified across arms and attack
categories, Cohen's κ reported against each judge in Methods. This validation set must be
independent of any set used to tune judge prompts.

## 5. Frozen evaluation suite

`data\redteam\redteam_suite.jsonl`, 300 items (60 each: `prefilling`, `persona`,
`many_shot`, `crisis_adjacent`, `benign_sensitive`).
**SHA-256 `e14c3a24184d01cbf31bbcfa42be03104ae07b0bea1132bc5b08a177645b6689`**, frozen
2026-07-31 01:22:46 +09:00. Any evaluation whose suite hash differs is not a paper number.
The suite is never trained on. Leakage checks against all training files returned zero
exact matches, zero shared 12-word spans, and max 5-gram Jaccard 0.1034.

## 6. Arms and training volume

| Arm | Definition |
|---|---|
| B0 | Base Qwen2.5-7B-Instruct, no training |
| B1 | SFT on therapy dialogue (LoRA), `checkpoint-290`, selected on held-out validation loss by a criterion fixed in advance |
| B2 | B1 + DPO on **19,924 helpfulness pairs** |
| B3 | B2 + guardrail filter at inference (**baseline to beat**) |
| T | B1 + DPO on **15,000 helpfulness + 4,924 safety pairs** |

**Matched data volume:** B2 and T both train on 19,924 preference pairs — same total, same
optimiser steps, only composition differs. Otherwise a T improvement would be confounded
with data volume rather than data type. Both counts are logged at run time.
Helpfulness pool excludes 662 pairs where `rejected_safety_rating > chosen_safety_rating`,
so B2 is a clean helpfulness-only arm rather than one nudged toward less-safe outputs.
LoRA adapter config is identical across B1, B2 and T (r=32, α=64, dropout 0.05, all
attention + MLP projections).

## 7. What would falsify or complicate the claim

Stated in advance so it cannot be rationalised later:

- ASR(T) ≥ ASR(B3), or a reduction below 10 points → the claim is not supported at the
  pre-specified effect size, and we say so in the abstract.
- Over-refusal increase > 5 points → "bounded cost" fails even with a large ASR gain.
- If B3's filter and the ASR judge cannot be kept independent, the B3 comparison is
  reported as unreliable rather than reported anyway.
- If judge κ is low on the independent validation set, every downstream number inherits
  that uncertainty and it is stated in Results, not buried in an appendix.

## 8. Known limitations fixed at pre-registration time

- LoRA rank was chosen for helpfulness adaptation; safety learning may need different
  capacity. Untested here.
- PsychoCounsel data and reward model are CC-BY-NC-4.0 (non-commercial); resulting
  checkpoints inherit this restriction. Stated in Methods and in the scope fence.
- `crisis_adjacent` "correct behaviour" follows a cited clinical-guidance rubric (NICE
  NG225, WHO mhGAP-IG 2.0, WHO safe-messaging 2023, Stanley & Brown 2012, C-SSRS), not
  author judgment, or the category would be unfalsifiable.
- B1 exhibits fabricated self-disclosure and terse under-response; both are inherited by
  all downstream arms and are invisible to every taxonomy used here. See
  `lab_notebook.md` § "Observed B1 behaviours". This is a limitation of the evaluation,
  not a difference between arms.
