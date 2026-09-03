# Project 71 — Safety-Focused Post-Training of a Therapy-Support LLM

MSc dissertation (UCL ELEC0054). Submission: **3 September 2026** (hard, no deferral).
This file is the contract. Do not propose work outside it.

---

## THE ONE CLAIM

> Mixing safety preference pairs into preference optimisation reduces attack-success-rate
> (ASR) on a held-out adversarial suite, compared to an identical model with a bolted-on
> guardrail filter, at a bounded cost in over-refusal and helpfulness.

*Method note: the objective is DPO with an RPO-style NLL anchor on the chosen completion
(`loss_type: [sigmoid, sft]`, `beta = 0.1`), not vanilla DPO — see EXPERIMENTAL ARMS. Per
pre-registration Revision 5 the safety pairs are PKU-SafeRLHF harm-category filtered with
11.8% mental-health keyword coverage, so the claim concerns whether general-harm safety
preference data **transfers** to therapy-domain adversarial prompts.*

Every script must serve this claim. If a proposed experiment does not produce a number
that appears in Table 1 or Table 2 (below), **do not build it** — say so and stop.

**Scope fence** (goes verbatim in the paper's Intro and Conclusion): one model family
(Qwen2.5-7B-Instruct), English only, one frozen attack taxonomy, LoRA post-training,
research prototype, **no clinical claims anywhere**.

---

## EXPERIMENTAL ARMS

| Arm | What it is | Seeds |
|-----|-----------|-------|
| B0 | Base Qwen2.5-7B-Instruct, no training | 1 (context only) |
| B1 | SFT on therapy dialogue (LoRA) | 1 (context only) |
| B2 | B1 + preference optimisation on helpfulness pairs only | **3** |
| B3 | B2 + guardrail filter at inference (**baseline to beat**) | derived from each B2 seed |
| T  | B1 + preference optimisation on helpfulness pairs **+ safety pairs** (**treatment**) | **3** |
| T_ctrl | as T, same PKU rows, preference direction from `better_response_id` (weak control, added by pre-reg Revision 5) | 1 |

B3 vs T is the comparison. Everything else is context.

**The preference-optimisation objective actually run is not vanilla DPO.** All DPO arms use
**DPO with an RPO-style NLL anchor on the chosen completion** — TRL `loss_type: [sigmoid, sft]`,
`loss_weights: [1.0, 1.0]`, `beta = 0.1`. Vanilla sigmoid-only DPO was tried first and **failed
to terminate**: 23% of responses hit the 512-token cap without emitting EOS, versus 5% for the
SFT model, and the failure was invariant to beta (23.0% at 0.1, 23.7% at 0.3) because the
update was renormalised to `max_grad_norm` on 58–61% of steps. The NLL anchor restores a
per-token gradient on the chosen response including its EOS token and resolved it below the SFT
baseline. Describe the method this way in the paper — "DPO" unqualified is wrong.

**Arm naming in all paper artifacts:** "**B2**" means **B2 v4 only** (the frozen configuration
above). The earlier v2 and v3 runs appear **solely** in the termination-failure analysis and are
never a row in Table 1 or Table 2.

**Cut and not up for discussion:** T+ (defence-in-depth), second base model, second
language, full fine-tuning, QLoRA, latency/throughput metrics, general-capability
benchmarks, ablations (reinstated only if a week finishes early).

---

## HARD ENVIRONMENT CONSTRAINTS

Machine: `CNXLAB03`, **Windows Server 2025**, RTX PRO 6000 Blackwell 96 GB, driver 591.59.
Accessed over VPN/RDP. Verified working: Qwen2.5-7B generation, 15.3 GB peak VRAM.

- **GPU is locked in TCC mode.** `nvidia-smi -dm 0` returns "Not Supported" (headless,
  no display path). WSL2 GPU passthrough is therefore impossible. **We run native Windows.**
- **No vLLM** — Linux-only. Batch generation goes through `transformers`.
- **No bitsandbytes** — unreliable on native Windows. No 8-bit optimisers, no QLoRA.
  This is why all arms use LoRA with an identical adapter config.
- **No tmux.** For long runs, *disconnect* the RDP session (do not sign out).
- Python venv: `C:\proj71\env`. Every new terminal needs
  `C:\proj71\env\Scripts\Activate.ps1`.
- PyTorch 2.11.0+cu128, arch list includes `sm_120`.

### Library version quirks (transformers 5.x — most tutorials are written for 4.x)

- `apply_chat_template(...)` returns a **dict**, not a tensor. Use `return_dict=True`
  and call `model.generate(**inputs, ...)`.
- `torch_dtype=` is deprecated → use `dtype=`.
- Attention: use `attn_implementation="sdpa"`. **Never** flash-attn (unsettled on Blackwell).
- TRL is 1.8 — check current `SFTTrainer` / `DPOTrainer` signatures before writing configs;
  do not copy 0.x-era arguments from blog posts.

---

## REPOSITORY LAYOUT

```
C:\proj71\
  data\raw\            # esconv, counsel_chat, psychocounsel_pref (downloaded, do not edit)
  data\processed\      # sft_train.jsonl, pref_helpful.jsonl, pref_safety.jsonl
  data\redteam\        # FROZEN eval suite — see below
  scripts\             # all code
  configs\             # yaml/json training configs, one per arm
  results\             # raw generations + scored outputs + tables
  notebook\            # setup_log.md, lab_notebook.md, pip_freeze.txt
  paper\               # TMLR LaTeX (main.tex, references.bib, tmlr.sty)
```

---

## DATA

| Purpose | Source | Notes |
|---------|--------|-------|
| SFT | `thu-coai/esconv` (~1.3k dialogues), `nbertagnolli/counsel-chat` (~2.8k rows) | Assistant-only loss. CounselChat is **not anonymised** (therapist names present) — note in ethics. |
| Helpfulness pairs | `Psychotherapy-LLM/PsychoCounsel-Preference` (~36k pairs) | Chosen/rejected therapist responses. |
| Safety pairs | **We construct these** — the methodological contribution | Chosen = safe de-escalation / appropriate refusal / escalation; rejected = unsafe or naively compliant. Protocol must be documented and auditable. |
| Red-team suite | **We construct these** | Four attack categories + benign-but-sensitive set. |

### THE FREEZE (non-negotiable)

The red-team suite freezes **31 July** and never changes after that date. It is **never
trained on**. Categories: `prefilling`, `persona`, `many_shot`, `crisis_adjacent`, plus
`benign_sensitive` (for over-refusal). Leakage invalidates the headline result.

---

## EVALUATION — EXACTLY TWO TABLES AND ONE SENTENCE

**Table 1 (main):** rows B0/B1/B2/B3/T × columns ASR↓, over-refusal↓, helpfulness↑.
Mean ± 95% CI over seeds for B2/B3/T; single-run italics for B0/B1.

**Table 2 (failure analysis):** ASR per attack category, B3 vs T. Shows where trained-in
safety wins **and where it still breaks**. Honesty about failure is required, not optional.

**The headline sentence** — the harness must emit this automatically, filled with real numbers:

> "T reduces ASR from X% to Y% relative to B3 (n=3 seeds, 95% CI, <test name>) while
> increasing over-refusal by at most Z points."

If the harness cannot fill that sentence, the harness is wrong.

**Judges (pin by 7 Aug — the eval-harness week — never change after):**
- Safety verdict → Llama Guard if Meta access granted; **fallback**: pinned prompted
  classifier using Qwen2.5-7B with a fixed classification prompt, documented as such.
- Helpfulness → `Psychotherapy-LLM/PsychoCounsel-Llama3-8B-Reward`, run locally
  (deterministic, citable, no API dependency).

**Statistics:** bootstrap CIs over the prompt set; paired test across prompts for T vs B3;
name the test, state n, report p.

---

## CODING CONVENTIONS

- **Determinism:** every script takes `--seed`; set `torch.manual_seed`, `random.seed`,
  `numpy.random.seed`. Log the seed into the output filename.
- **Configs, not hardcoded values.** Training hyperparameters live in `configs\*.yaml`.
  A run must be reproducible from its config file alone.
- **Log every run** to `notebook\lab_notebook.md`: date, script, config, seed, outcome,
  wall-clock. This file becomes the Methods section and reproducibility appendix.
- **Save raw generations before scoring.** Never overwrite raw model outputs; scoring is
  a separate pass so judges can be re-run without regenerating.
- Small, single-purpose scripts. No notebooks for anything that produces a paper number.
- Fail loudly on missing data — never silently skip examples.

---

## SCHEDULE (six weeks)

| Week ending | Done means |
|---|---|
| 31 Jul | Novelty search logged; SFT data formatted; safety pairs built; **red-team suite FROZEN**; B1 trained |
| 7 Aug | B2 seed 1 trained; B3 wired; eval harness emits the headline sentence; B0/B1/B2/B3 baseline numbers |
| 14 Aug | T seed 1 trained; first T-vs-B3 comparison; seeds 2–3 queued |
| 21 Aug | All seeds done; CIs + significance test; Tables 1–2 final; **RESULTS LOCK** |
| 28 Aug | Full paper draft (write order: Methods → Results → Intro → Related Work → Discussion → Conclusion → Abstract last) |
| 3 Sep | Submission checklist passed; **SUBMIT** |

**Descoping order if behind:** (1) shrink judge-calibration subset, (2) trim suite size
*only before the freeze*, (3) drop B0/B1 to a sentence, (4) seeds 3→2.
**Never cut:** the B3 baseline, the over-refusal metric, the frozen suite, ≥2 seeds on
core arms, or Table 2.

---

## METHODOLOGICAL SAFEGUARDS (v2 â€” supersedes conflicting text above)

These six rules exist because violating any one of them invalidates the headline result.
They are not preferences.

### 1. JUDGE INDEPENDENCE (critical)

The mechanism used to **filter** B3 must never be the mechanism used to **judge** ASR.
If both are Llama Guard, B3 scores near-zero ASR by construction, not by merit, and the
central comparison collapses.

**ROLES AS ACTUALLY PINNED — updated 2026-08-27 per Amendments 20/21 (Llama Guard
granted 2026-08-05 under the Amendment 19 performance-blind calendar rule; beaver-dam
demoted to diagnostic-only).** Recorded in `configs/judges_pinned.lock.v3.json`
(v1/v2 superseded, retained unedited), enforced in `eval_score.py` and
`apply_b3_filter.py`, and tested by `scripts/test_judge_independence.py` (15/15).

| Role | Mechanism | Never also used as |
|---|---|---|
| B3 guardrail filter | `meta-llama/Llama-Guard-3-8B` @ `7327bd9f…` (Amendment 20) | any ASR judge |
| ASR judge (prefilling, persona, many_shot) | Qwen2.5-7B + pinned behavioural rubric v3 | the B3 filter |
| Behaviour judge (crisis_adjacent) | Qwen2.5-7B + pinned clause rubric v3 | the B3 filter |
| Over-refusal (benign_sensitive) | **Human labels** (Revision 4); Qwen2.5-7B refusal rubric v3 as cross-check only | the B3 filter |
| Helpfulness | `Psychotherapy-LLM/PsychoCounsel-Llama3-8B-Reward` (in the v3 lock's hash chain, Amendment 21) | — |

*(The 2026-08-01 Revision 2 table this replaces — beaver-dam as filter — is preserved in
the superseded block below and in the v1/v2 locks; the beaver-dam filter era produced no
B3 artefact.)*

The three prompted judges **share a Qwen2.5-7B base with each other**, which rule 1 permits
— what it forbids is a judge sharing a mechanism with the *filter*, and none does. State
this in Methods rather than leaving a reviewer to notice it.

> **Superseded, kept visible so the change is auditable.** The original table read:
> B3 filter = Llama Guard (fallback: pinned Qwen2.5-7B prompted classifier); ASR judge =
> `beaver-dam-7b`; behaviour judge = Qwen2.5-7B + rubric. Revision 1 measured beaver-dam at
> κ = 0.355 on 100 blind held-out items, missing 17 of 29 genuine attack successes, because
> this suite's successful attacks are content-light and behavioural. Revision 2 then found
> that beaver-dam's κ is unstable across samples (0.355 / 0.086 / 0.116) and that it acts as
> a *topic* detector, not a harm detector — it flagged a reply that **inverts** a self-harm
> prefill, a **correct** emergency triage, and a plain refusal. The roles were therefore
> swapped: beaver-dam moved to its natural QA-moderation role as the filter (a stronger
> baseline to beat), and the behavioural judge became the ASR judge (κ = 0.521, precision
> 1.00). *(2026-08-03: the beaver-dam κs quoted here are bug-era — measured under the
> EOS-omission defect, Amendment 18. Corrected-path values on the same sets are LOWER:
> 0.064 / 0.038 / 0.000, so the retirement stands a fortiori. The behavioural judge's
> κ = 0.521 is unaffected — different code path. Llama Guard status changed: access
> requested 2026-08-03; decision rule in Amendment 19.)* **Llama Guard disposition** — no access request was ever
> made, and the fallback Qwen classifier must NOT be installed: the ASR judge runs on
> Qwen2.5-7B, so a Qwen filter is a rule-1 violation. That configuration already halted the
> harness once under Revision 1, and `test_judge_independence.py` case 2 is the standing
> negative control for it.

Two safety judges are needed because a harm classifier cannot assess *correct crisis
behaviour* or *wrongful refusal* â€” a blank refusal is "harmless" but often wrong. State
this reasoning in Methods.

**Methods must contain an explicit independence statement.** A reviewer will look for it.

### 2. MATCHED DATA VOLUME

T trains on N helpfulness + M safety pairs. **B2 must train on N+M helpfulness pairs** â€”
same total, same optimiser steps, only composition differs. Otherwise any T improvement is
confounded with data volume rather than data type. `prepare_pref_helpful.py` produces the
full filtered pool; the training script samples to match at run time and logs both counts.

### 3. SAFETY-PAIR PROVENANCE

Safety pairs come from `PKU-Alignment/PKU-SafeRLHF`, harmlessness dimension, filtered to
mental-health-relevant harm categories. **Not model-generated** â€” this removes the
distillation critique and makes the result reproducible.

- Cite `ji2024pku` and `ji2024beavertails`.
- Provenance is independent of the red-team suite by construction (different source,
  different authors). **Report overlap statistics anyway** â€” exact and near-duplicate
  prompt matching against `data/redteam/` â€” and state the result in Methods.
- Category overlap between training and eval is intended and fine; *item* overlap is not.

### 4. JUDGE VALIDATION (non-negotiable)

Every number depends on automatic judges. **Hand-label 100â€“150 responses**, stratified
across arms and attack categories, and report Cohen's Îº against each judge in Methods.
Half a day of work; converts "we assume the judge is right" into a measured agreement
figure. Store labels in `results/human_validation/`.

### 5. STATISTICS

**Superseded on two points by pre-registration Amendment 13 (2026-08-03):** the primary test
is a **two-sided paired sign-flip permutation test** (the seed-mean reduction makes outcomes
non-binary, so McNemar does not apply), and the primary **n is 180**, with `crisis_adjacent`
(60) a **co-primary reported separately and never pooled**. Per-seed exact McNemar is kept as
a robustness check. Original text follows.

**Primary test: McNemar's test** on paired binary safe/unsafe outcomes across the prompt
set â€” both arms see identical prompts, so this runs over hundreds of paired outcomes rather
than 3 seeds. Seed variance is reported separately as a robustness check, not as the primary n.
Bootstrap CIs over the prompt set. Name the test, state n, report p.

### 6. PRE-REGISTERED EFFECT SIZE

**Before running any evaluation**, write `notebook/preregistration.md` fixing: the
minimum ASR difference considered meaningful (default **10 percentage points**), the
maximum acceptable over-refusal increase, the primary test, and the judge configuration.
Dated and committed before results exist.

This converts a null result from a shrug into a finding: "we powered for a 10-point effect
and observed 3, bounding the benefit at X." Write it before you can be tempted.

### Additional limitations to state in Discussion

- LoRA rank was chosen for helpfulness adaptation; safety learning may need different
  capacity. Untested here.
- PsychoCounsel data and reward model are **CC-BY-NC-4.0** â€” non-commercial. Resulting
  checkpoints inherit this restriction. State in Methods and in the scope fence.
- Crisis-adjacent "correct behaviour" follows a cited clinical-guidance rubric, not author
  judgment, or the category is unfalsifiable.
