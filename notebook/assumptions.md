# Assumptions register

Compiled 2026-08-29 (post RESULTS LOCK, pre-drafting) at the owner's direction. Every
assumption the headline claim rests on, each with how it was tested or an explicit
statement of why it was not. This file becomes the Methods "Assumptions and audit
trail" subsection; nothing here modifies any locked number. Companion record:
`notebook/results_manifest.md` (the audit trail — every exhibit → artifact → config →
seed → hash).

Status legend: **TESTED** (direct evidence exists, artifact cited) · **PARTIAL**
(evidence bounds it but does not close it) · **UNTESTED** (stated limitation, carried
into Discussion).

## A1. The judge rubrics operationalise the constructs — PARTIAL

**Assumption.** "Attack success", "correct crisis behaviour", and "over-refusal" as
judged equal the constructs the claim is about.
**Evidence.** ASR judge: precision 1.00 / recall 0.55 vs blind hand labels
(κ = 0.521, n = 36, lock v6 `judges` block); 60-item human pass κ = 0.668
[0.453, 0.851], gold 6/6, intra-rater 10/10
(`results/human_validation/human_asr_kappa_report_20260826_231917.json`). Crisis
judge: κ = 0.583 (n = 24, CI [0.22, 0.83]) against a closed five-item clause list
from cited clinical guidance (SUITE_MANIFEST 2.4) — it cannot see quality failures
the rubric does not name. Refusal judge FAILED validation twice (κ ≈ 0.075) and was
**replaced** as primary instrument by hand labels (Revision 4); it appears only as a
flagged cross-check.
**Residual risk.** Recall 0.55 means ASR is a conservative lower bound, and equal
recall across arms is an assumption, not a measurement (the judge is documented as
under-counting terse arms). Stated in the headline qualifier itself and in
`notebook/power_analysis.md` (recall-sensitivity sweep).

## A2. PKU-SafeRLHF harmlessness preferences transfer to therapy-domain attacks — TESTED (it is the claim)

**Assumption.** General-harm safety pairs (11.8% mental-health keyword coverage)
move behaviour on therapy-domain adversarial prompts.
**Evidence.** This is not a background assumption but the pre-registered hypothesis;
the B3-vs-T comparison is its test, and the result is quantified (−5.56 pts raw,
p = 0.227, CI [−13.89, +2.78]) rather than assumed. Item-level train/eval
independence was verified separately: exact and near-duplicate overlap between the
PKU rows and the frozen suite reported (safeguard 3; overlap statistics in Methods).
**Residual risk.** Transfer magnitude below the pre-registered threshold; the study
is inconclusive at ±10 points (TOST bound δ = 12.22). Reported as the bound, not as
absence.

## A3. Greedy decoding represents deployment behaviour — UNTESTED

**Assumption.** ASR measured under deterministic greedy decoding predicts behaviour
under deployment sampling (temperature > 0).
**Why untested.** Determinism was chosen so every number is exactly reproducible and
judge scoring is re-runnable on frozen generations (a design trade recorded in
`configs/eval_generation.yaml`); a sampling sweep was cut with the other ablations
(CLAUDE.md scope fence). Sampling could raise or lower ASR and could do so
differentially across arms. Discussion limitation, one sentence.

## A4. The frozen suite spans the relevant attack space — PARTIAL

**Assumption.** The four attack categories plus benign_sensitive cover the attack
surface the claim quantifies over.
**Evidence.** Taxonomy fixed and frozen before any model was evaluated (31 Jul,
sha e14c3a24…, never trained on — leakage checks in `check_redteam_leakage.py`);
category balance validated (`validate_redteam.py`). The scope fence limits the claim
to this one frozen taxonomy explicitly.
**Residual risk.** Known blind spot recorded at judge validation: no category for
evasive non-answers, so failure is under-counted for terse arms. New attack styles
(post-freeze jailbreaks) are out of scope by construction and the paper must not
generalise beyond the taxonomy.

## A5. LoRA-scale adaptation carries the safety behaviour being measured — UNTESTED

**Assumption.** Rank/alpha chosen for helpfulness adaptation gives safety
preferences enough capacity; conclusions would survive full fine-tuning.
**Why untested.** Full fine-tuning and QLoRA are environment-impossible here (no
bitsandbytes on native Windows; CLAUDE.md hard constraints); rank ablation was cut.
All arms share one identical adapter config, so the B3-vs-T *contrast* is
capacity-matched even if the level is capacity-limited. Stated limitation
(CLAUDE.md Discussion list).

## A6. A single training seed supports the primary estimate — PARTIAL (measured post-hoc)

**Assumption.** Seed-1 point estimates are not seed artifacts.
**Evidence.** Amendment 22.1 descoped to one seed (blind, time-forced); Amendment 23
then trained seeds 2–3 as a sighted, pre-committed robustness panel: primary
direction consistent (−5.56/−3.33/−0.56, seed 1 the largest), crisis co-primary NOT
seed-robust (sign flip; seed 2 nominally adverse, p = 0.048, reported unsoftened).
Bitwise training determinism demonstrated by the T_r200 replication (adapter
sha-identical given same data+seed).
**Residual risk.** The primary n remains seed 1 exactly as pre-registered; the panel
is labelled post-hoc and cannot be promoted.

## A7. The hand labeller is reliable — PARTIAL

**Assumption.** The owner's over-refusal and ASR hand labels measure what a
competent independent rater would measure.
**Evidence.** Gold-item gate 6/6 (after a documented construct correction of three
items, logged 2026-08-27 with the rubric clauses quoted); intra-rater 10/10 on the
relabel pass; ASR pass gold 6/6, intra-rater 10/10, judge-vs-human κ = 0.668. The
2/120 items with external-reasoning assistance are disclosed in the notebook.
**Residual risk.** Single annotator, no inter-annotator agreement — stated
limitation in the lock's `labelling_caveat` and Methods (m5: the correction event is
stated once, in one place).

## A8. The helpfulness reward model is loaded and scored correctly — TESTED

**Assumption.** The reward head ranks genuine chosen > rejected (a wrong pooling /
head-loading / padding path would silently produce garbage — the beaver-dam EOS bug
is the precedent).
**Evidence.** Acceptance test 2026-08-27, threshold ≥ 0.70 pairwise accuracy:
**PASS 50/50 (accuracy 1.000, 0 ties)** on pairs held out from BOTH trained arms'
manifests, scored through the production code path
(`scripts/test_reward_model_acceptance.py`;
`results/judge_probes/reward_acceptance_20260827_004017.json`; lab notebook
2026-08-27 "PHASE 2 COMPLETE"). Model revision-pinned in the lock hash chain
(Amendment 21).
**Residual risk.** The held-out pairs are trivially separable (median chosen/rejected
similarity 0.108), so 50/50 certifies the interface, not fine discrimination;
helpfulness remains a relative measure on one reward model (CC-BY-NC-4.0, stated).

## A9. Judge and filter are mechanistically independent — TESTED

**Assumption.** The B3 filter shares no mechanism with any ASR judge (else B3's ASR
is low by construction and the claim is a tautology).
**Evidence.** Enforced in code (`judge_independence_block`,
`assert_filter_is_not_an_asr_judge`), tested by 15 negative controls
(`scripts/test_judge_independence.py`, 15/15, also passing from a fresh clone),
stated in the lock's `independence_statement`. The three prompted judges share a
Qwen base **with each other**, which the rule permits and Methods must state.

## A10. The pipeline is deterministic and the record complete — TESTED

**Assumption.** Every reported number can be regenerated from config + seed, and the
artifact trail proves it.
**Evidence.** b1 canonical regeneration byte-identical (300/300); T_r200 adapter
bit-identical under same data+seed; fresh-clone test passes suite hash, judge pin
(v6), and independence suite; `notebook/results_manifest.json` (38 exhibits,
adversarially verified: 74 hash + 169 value checks) maps every exhibit to artifact,
config, seed, sha256, and commit. This register plus that manifest are the "explicit
assumptions and audit trail" pair.
