# Pre-registration amendments — Project 71

**Amendments 8–12 dated 2026-08-02; Amendment 13 dated 2026-08-03.** Companion to
`preregistration.md` (written 2026-07-31, Revisions 1–7 and Amendment 7a dated 2026-08-01).
This file continues the same numbering sequence.

Same discipline as the parent file: nothing here is deleted or rewritten later; a
superseded entry is struck through and the replacement dated. Every entry carries a
one-line justification and an explicit blindness statement.

**What "decided blind to arm-comparison results" means here, precisely.** As of
2026-08-02 no arm has been scored with the pinned judge configuration. `B3` has never been
generated. `T` seed 1 finished training at 05:53 today and has produced **no** evaluation
output — `results/T_dpo_seed1/` contains the adapter, `checkpoint-1246`, the run manifest
and the reference verification, and no `degeneracy_analysis.json`, no generations, no
scores. The only cross-arm numbers that exist anywhere in the repository are the superseded
B0/B1 harness-debugging runs (a `repetition_penalty=1.0` decoding artifact, explicitly not
paper numbers). Every decision below was therefore made without sight of any B3-vs-T,
T-vs-T_ctrl, or ASR/over-refusal comparison. This is verifiable from the file system, not
merely asserted.

---

## Amendment 8 — Loss function

The preference-optimisation objective for **all** DPO arms (B2, T, T_ctrl) is **DPO with an
RPO-style NLL anchor on the chosen completion**: TRL `loss_type: [sigmoid, sft]`,
`loss_weights: [1.0, 1.0]`, `beta = 0.1`. Vanilla sigmoid-only DPO is abandoned.

**Justification (one line):** vanilla DPO produced a non-termination measurement confound —
23% of responses hit the 512-token cap without emitting EOS versus 5% for the SFT model —
and the failure was invariant to β (23.0% at β=0.1, 23.7% at β=0.3), so it could not be
tuned away.

*Decided blind to arm-comparison results.* The diagnostic was run **within B2 only**
(v2 β=0.1 and v3 β=0.3, plus a four-way β sweep); no B3, T or T_ctrl model existed and no
ASR, over-refusal or helpfulness number was computed on any arm. The decision rule — that
a cap-hit rate above the SFT baseline is disqualifying — was recorded before the NLL-anchor
run was launched, and the anchored run (B2 v4) is accepted against it: **1.7% strict / 3.0%
loose / cap-hit below the SFT baseline**.

**Consequence for the paper:** "DPO" unqualified is a wrong description of this method and
must not appear. The v2 and v3 runs are reported **solely** in the termination-failure
analysis and are never a row in Table 1 or Table 2; "B2" in every paper artifact means
B2 v4.

---

## Amendment 9 — T_ctrl seed count

**T_ctrl runs at 1 seed.**

**Justification (one line):** T_ctrl is a control arm, not a primary contrast; GPU hours are
reallocated to seeds 2–3 of the core arms (B2, T), which carry the headline comparison.

*Decided blind to arm-comparison results.* T_ctrl has not been launched and no arm has been
scored.

> **Correction to the amendment as dictated.** This was recorded as "T_ctrl **reduced** to
> 1 seed". It was not reduced: T_ctrl was specified at 1 seed at the moment it was created,
> in Revision 5 (2026-08-01), and the seed table in `preregistration.md` § 6 has read
> "T_ctrl | 1 | 1-seed weak control (Revision 5)" ever since. There was never a 3-seed
> T_ctrl plan to cut. This entry therefore **confirms and justifies** an existing
> commitment rather than changing one, and is logged so that no reader infers a descope
> that did not happen. Recording it the other way would have manufactured a decision.

**Consequence for the paper:** T_ctrl carries no confidence interval across seeds and is
reported as a single-run weak control, in italics, exactly as B0 and B1 are. It cannot
support a significance test against T, and no such test will be reported.

---

## Amendment 10 — Human validation set size

Human validation proceeds in two stages: **60 items labelled now**, extended to **~120
arm-blinded stratified items** once all arms have generations.

**Justification (one line):** the 60-item worksheet can be labelled immediately against the
arms that exist, unblocking judge-agreement reporting, while the full set required by
CLAUDE.md safeguard 4 (100–150 responses stratified across arms **and** attack categories)
cannot be drawn until B3, T and T_ctrl generations exist.

*Decided blind to arm-comparison results.* The 60-item worksheet
(`results/human_validation/human_asr_worksheet.txt`) is already drawn and sealed: uniform
random within each of the three attack categories, 20 each, seed 2026, over a
deterministically sorted pool, with the labeller blind to arm. No labels have been entered.

**Stated limitations, not to be discovered later:**

- The 60-item stage is **below** the 100–150 range CLAUDE.md safeguard 4 requires. It is an
  interim measurement, and any κ computed from it must be reported as such, with its
  confidence interval, and must not be presented as satisfying safeguard 4.
- **No gold or attention-check items are embedded** in the 60-item worksheet. There is
  therefore no internal check on labelling attention; the only cross-check available is
  human-vs-judge agreement itself, which cannot separate a judge error from a labeller
  lapse. The ~120-item extension **should** embed gold items, and if it does not, that
  absence is reported in Methods.
- The ~120-item extension is a *superset in kind*, not necessarily a superset in items;
  whether the original 60 are carried forward or redrawn will be recorded here before it is
  done.

This entry concerns the **ASR** validation set. It is distinct from the 120-item
**over-refusal** worksheet, which is a separate instrument on the `benign_sensitive`
category; the coincidence of the number 120 is not a relationship between them.

---

## Amendment 11 — Judge contingency

The **primary pre-registered ASR judge remains primary regardless of the outcome of the
gold-item probe.** If the probe shows misses, a clearly-labelled **exploratory**
human-anchored metric is reported **alongside** the primary — never replacing it, never
promoted to headline, and never used to compute the pre-registered effect size.

**Justification (one line):** a judge may not be swapped out after its weaknesses are
measured on the very data it will score, because the swap would then be selected — however
honestly — against the results it produces.

*Decided blind to arm-comparison results.* The gold-item probe has not been run, and no arm
has been scored with the pinned configuration.

> **Correction to the amendment as dictated.** This was recorded as "**beaver-dam-7b**
> remains the primary pre-registered ASR judge". Under Revision 2 (2026-08-01) that is not
> beaver-dam's role. The roles were **swapped**: `PKU-Alignment/beaver-dam-7b` is the **B3
> guardrail filter**, and the **ASR judge is the pinned rubric-based behavioural judge on
> Qwen2.5-7B** (κ = 0.521 [0.29, 0.76], recall 0.55, precision 1.00 on a held-out set
> disjoint from any tuning set). Revision 2 made that swap precisely because beaver-dam was
> measured as *not* a usable ASR judge for this suite — κ unstable across samples
> (0.355 / 0.086 / 0.116), behaving as a topic detector rather than a harm detector. The
> role assignment is enforced in code: `eval_score.py` enumerates every ASR judge and raises
> on collision with the filter mechanism, and `configs/judges_pinned.lock.json` records the
> pin. **This amendment is therefore written against the behavioural judge**, which is the
> arm that actually holds the primary ASR role. Writing beaver-dam into a commitment
> document as the ASR judge would have contradicted both the pin lock and the independence
> statement a reviewer will check.

**Scope, stated so the contingency cannot be stretched:** "exploratory" means the metric
appears in Results with an explicit exploratory label, is excluded from Table 1's ASR
column, is excluded from the headline sentence, and is not used to evaluate the
pre-registered 10-point threshold. If the probe shows misses, that fact is reported whether
or not it changes any conclusion.

---

## Amendment 12 — Interpretation of a T differential-degeneracy failure

If T fails the pre-registered degeneracy differential —
**|loose(B2 v4) − loose(T)| > 3 points overall, or > 5 points in any attack category** —
under the **identical frozen configuration**, the response is **investigate and report**.
The configuration is **not** abandoned and T is **not** silently retrained into compliance.

**Justification (one line):** with the config held byte-identical between arms, a
differential is plausibly an effect of the safety data itself — a real property of the
treatment and a finding — rather than evidence of a broken run.

*Decided blind to arm-comparison results.* This is the strongest blindness claim in this
file and it is checkable: **T seed 1 has no degeneracy artefacts of any kind.** No
`degeneracy_analysis.json`, no full-suite generations, no cap-hit count. The interpretation
rule below is being fixed **before the number that triggers it exists**.

**The reference point, recorded now** (B2 v4, `degeneracy_analysis.json`, n=300):

| | overall | benign_sensitive | crisis_adjacent | many_shot | persona | prefilling |
|---|---|---|---|---|---|---|
| strict | 1.7% (5/300) | 0/60 | 0/60 | 3/60 | 1/60 | 1/60 |
| **loose** | **3.0% (9/300)** | 0/60 | 0/60 | 4/60 (6.7%) | 1/60 (1.7%) | 4/60 (6.7%) |

**What "investigate and report" obliges, enumerated in advance so it cannot be softened
into inaction:**

1. Confirm the configurations are byte-identical (`assert_hyperparams_match_sibling()`) and
   that the only difference between B2 v4 and T is pair composition.
2. Manually re-scan the flagged items — the automated content flag has produced a false
   negative twice, so the ids are obtained before the judge pass, not after.
3. Report the differential in Table 2's neighbourhood **whatever its sign**, and state
   whether the degeneracy concentrates in the categories where T is claimed to win. A
   treatment that wins on ASR while degenerating in the same categories is a confound, and
   the paper says so.
4. Only a *diagnosed defect in the run itself* — not an unwelcome number — licenses a
   relaunch, and any relaunch is logged here with the defect named.

**The distinction this rests on, restated:** absolute degeneration is a **limitation**;
**differential** degeneration is a **confound**. Only the second threatens the B3-vs-T
comparison, and only the second is fatal to the claim. Amendment 12 governs the second and
does not license explaining away the first.

---

## Amendment 13 — The primary statistical test and the primary n

**Dated 2026-08-03.** This amendment records a deviation that **already exists in shipped
code** and was never written into the pre-registration. It is a correction of the record,
not a new decision, and it is labelled as such.

`preregistration.md` § 3 and CLAUDE.md § Methodological Safeguards v2 rule 5 both name
**McNemar's test** on paired binary safe/unsafe outcomes, at **n = 240** attack prompts
(300 minus the 60 `benign_sensitive` items). `scripts/stats.py` ships neither. It runs:

| | pre-registration § 3 | what `stats.py` actually does |
|---|---|---|
| primary test | McNemar's exact test | **two-sided paired sign-flip permutation test** on per-prompt differences (`stats.py:230`, invoked as `primary` at `:635`) |
| primary n | 240 attack prompts | **180** — `prefilling`, `persona`, `many_shot` (`PRIMARY_ATTACK_CATEGORIES`, `stats.py:71`) |
| `crisis_adjacent` (60) | pooled into the 240 | **co-primary, reported separately, never pooled** |
| McNemar's role | primary | retained per seed as a **robustness check**, exact rather than chi-square |

**Justification (one line each).**

1. **Test.** Revision 6 replaced the reduction with an attenuation-corrected **seed mean**
   per prompt. Those values are no longer binary, so McNemar's discordant-pairs
   construction does not apply to them; a sign-flip permutation test is distribution-free
   and valid on the continuous per-prompt differences. McNemar is kept per seed, where
   outcomes genuinely are binary.
2. **n.** `crisis_adjacent` is scored by a **different judge** against a positively
   specified clinical rubric in which a **refusal is a failure** — the opposite refusal
   semantics from the three attack categories, where a refusal is a success. Pooling them
   averages a rate over items where refusing is right with items where refusing is wrong,
   and the two effects partially cancel. Splitting them makes each interpretable.

*Decided blind to arm-comparison results.* No arm has been scored with the pinned
configuration; B3 has never been generated; T seed 1 has produced no evaluation output.
Both changes are therefore still being fixed **before** any number they could be tuned
against exists. That is the only reason this is a correction rather than a
post-hoc rationalisation, and the window closes the moment anything is scored.

**Provenance of the discovery, stated plainly.** This was not caught by reviewing the
pre-registration. It surfaced on 2026-08-03 when `paper-writer` drafted Methods from the
lab notebook and the shipped code, and found the two sources disagreeing. The reasoning for
both changes was already written into `stats.py`'s module docstring at the time the
estimator changed under Revision 6 — the omission was that **Revision 6 never propagated
into § 3**. A revision that changes the estimator and silently invalidates the
pre-registered test is exactly the failure this document exists to catch, and it went
undetected for two days.

**Consequences.**

- The headline sentence names the test and the n. Both change: the test is a paired
  permutation test, and the ASR figure it reports covers **180 prompts**, not 240.
- `crisis_adjacent` gets its **own** reported comparison and its own n = 60. It is not
  absorbed into the headline ASR number. Methods and Results must both show it separately.
- The 10-point pre-registered effect size (Revision 6, TRUE scale) is unchanged and is
  evaluated against the **180-prompt** primary comparison.
- `preregistration.md` § 3 and CLAUDE.md rule 5 are **superseded on these two points** and
  now carry a pointer to this amendment. Their original text stays visible.
- The per-seed exact McNemar results are still computed and still reported, as a robustness
  check. Nothing is lost from the pre-registered analysis; it is demoted, not deleted.
