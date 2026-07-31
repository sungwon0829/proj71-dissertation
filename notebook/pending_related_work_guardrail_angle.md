## Search: guardrail-vs-trained angle, continued (2026-08-01, lit-scout) — filter-side and
therapy-domain focus

**PENDING MERGE NOTE:** `related_work.md` was under concurrent modification by other
lit-scout runs during this session (repeated "file has been modified since read" errors on
Write). Per the project's no-concurrent-write convention (see `lab_notebook.md`'s
`pending_valloss.md` / `pending_prefdata.md` / `pending_dpo_configs.md` /
`pending_brevity_verdict.md` pattern), this content is written here for the orchestrator to
merge into `related_work.md` as a new top-level `## Search:` section, continuing the entry
numbering from wherever the file's other sessions ("safety-dpo angle," entries 1–15, and
"attacks-shallow angle," entries 16–27) land after merge. **Do not delete this file until
the merge is confirmed in `related_work.md` itself.**

**Question (same escalation target as the two prior sessions found in `related_work.md`,
approached from the filter side and the therapy-domain side, which those two sessions
covered less deeply):** find prior work that directly compares an external
guardrail/moderation filter (Llama Guard, beaver-dam, OpenAI moderation, NeMo Guardrails)
against safety trained into model weights, plus work on guardrail failure modes, filter
over-refusal, and whether a bolt-on filter is a fair baseline. This pass deliberately did
not re-derive the DPO-safety-training side or the attack-taxonomy side, both already
covered in detail in the existing file (entries 1–27 there, especially #4 Egida and #23
Ackerman & Panickssery, the two previously-flagged closest matches).

**Method caveat, stated plainly:** several arXiv PDFs in this pass did not extract cleanly
via WebFetch (compressed PDF streams returned as binary); where that happened, the entry
below is based on the abstract page only, and this is stated explicitly. This is a weaker
evidence standard than the full-text HTML fetch used for Egida and Ackerman & Panickssery
in the existing sessions — treat any "unconfirmed" tag below as a follow-up to-do, not a
cleared check. Where `arxiv.org/pdf/<id>` failed, `arxiv.org/html/<id>` or
`arxiv.org/abs/<id>` sometimes rendered more reliably and was used instead.

**Result, consistent with both prior sessions already in the file: still no paper found
running the exact controlled comparison** (identical base model, matched training data
volume, DPO-safety-trained arm vs. bolted-on-filter arm, ASR + over-refusal + helpfulness,
in a therapy/mental-health domain, on a frozen red-team suite). Two new papers not found in
the prior two sessions come closest, one from each side of the comparison, and neither
changes the "no scoop found" verdict:

- **Xin, Chen, Yang, Backes, Zhang, "Jailbreaking Attacks vs. Content Safety Filters"**
  (below) — the closest *filter-side* paper found across all three search sessions: it
  evaluates jailbreak ASR across a full deployment pipeline *with* content moderation
  filters attached, which is structurally the filter half of our B3 arm. It does not build a
  DPO-safety-trained arm to compare the filter against.
- **Kim, Wang, Yoon, Huang, Saha, "AI Content Moderation in Therapy Conversations"**
  (below) — the closest *domain* paper on the filter side: it audits exactly the class of
  general-purpose guardrails our claim is about (Llama Guard among them) against real
  therapy conversation content, and finds they over-flag it — direct evidence that testing a
  bolted-on filter as B3 in a therapy-support system is a motivated, non-strawman comparison.
  It does not compare against a DPO-trained-in-safety alternative.

Read together with Egida (existing #4, training side) and Ackerman & Panickssery (existing
#23, the earlier near-miss), the shape of the gap is now clearer across all three sessions:
the *training* side (Egida, LLUMI, B-DPO below, RED QUEEN GUARD below) has no filter
baseline; the *filter* side (Xin et al. below, Young below, LEG below) has no
trained-safety baseline; the *domain* side (LLUMI, Kim et al. below, MHSafeEval below,
TherapyGym below) has neither an adversarial ASR arm nor a filter-vs-training comparison;
and the one paper with a training-vs-inference-time shape (Ackerman & Panickssery) compares
different mechanisms (SFT-refusal vs. input sanitization) in a different domain. No single
paper spans training-mechanism + filter-mechanism + domain + ASR triad.

---

### [next-number]. Xin, Chen, Yang, Backes, Zhang. "Jailbreaking Attacks vs. Content Safety
Filters: How Far Are We in the LLM Safety Arms Race?" arXiv:2512.24044, Dec 2025 (preprint);
accepted **ACL Findings 2026** (peer-reviewed venue, per ACL Anthology listing
`2026.findings-acl.20`).
- **Claims (verbatim abstract):** prior jailbreak-ASR evaluations "focused solely on the
  models, neglecting the full deployment pipeline, which typically incorporates additional
  safety mechanisms like content moderation filters"; when the full pipeline (model +
  input/output filters) is evaluated, "nearly all evaluated jailbreak techniques can be
  detected by at least one safety filter, suggesting that prior assessments may have
  overestimated the practical success of these attacks," but "there remains room to better
  balance recall and precision" in the filters themselves.
- **Evaluates:** jailbreak ASR of multiple attack techniques against LLMs with and without
  content moderation filters in the inference pipeline (model-alone vs. model+filter). The
  exact model list, exact filter list (e.g. whether Llama Guard/OpenAI moderation/NeMo
  Guardrails specifically are tested), and whether over-refusal is reported as a named,
  quantified metric beyond the general precision/recall discussion were **not confirmed** —
  the abstract was retrieved reliably; PDF full text did not extract, and
  `arxiv.org/html/2512.24044` was not independently opened to check.
- **Relation to our claim: overlaps on the filter-effectiveness half only, not the full
  comparison.** This is the closest *filter-side* prior work found across all three search
  sessions — structurally close to measuring "how well does a bolted-on filter reduce ASR,"
  the B3 half of our comparison. It does **not** construct or measure a DPO-safety-trained
  model as the alternative arm; its comparison is filtered-pipeline vs. unfiltered-pipeline,
  not filter vs. trained-in-safety on a matched base model. No therapy/mental-health domain.
  **Recommend a follow-up fetch via `arxiv.org/html/2512.24044` (not the PDF) to confirm the
  exact filter/model list and whether over-refusal is quantified, before citing model/filter
  specifics in the final paper.**

```bibtex
@inproceedings{xin2026jailbreaking,
  title={Jailbreaking Attacks vs. Content Safety Filters: How Far Are We in the {LLM} Safety Arms Race?},
  author={Xin, Yuan and Chen, Dingfan and Yang, Linyi and Backes, Michael and Zhang, Xiao},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026},
  note={arXiv preprint arXiv:2512.24044}
}
```

---

### [next-number]. Zhao, Wang, Xiong, Chen, Zhu, Ruan, Xiao, Duan, Chen, Wei. "Improving
Safety Alignment via Balanced Direct Preference Optimization" (B-DPO). arXiv:2603.22829,
Mar 2026 (preprint; venue not confirmed).
- **Claims (verbatim abstract):** DPO safety alignment "suffers from severe overfitting,"
  which the paper traces to an "Imbalanced Preference Comprehension" phenomenon between
  chosen/rejected responses; B-DPO "adaptively modulates optimization strength between
  preferred and dispreferred responses based on mutual information" to fix this, enhancing
  "safety capability while maintaining ... competitive general capabilities ... compared to
  state-of-the-art methods."
- **Evaluates:** per the abstract, safety capability and general-capability benchmarks
  against unnamed "state-of-the-art methods" — **no external guardrail/moderation filter is
  named as a baseline anywhere in the abstract.** Whether ASR, over-refusal, and
  helpfulness are all reported as distinct columns (a lossy PDF-based automated summary
  suggested this) is **not confirmed** from the abstract text alone — flagged as unconfirmed,
  do not rely on the metric-triad claim without a follow-up fetch.
- **Relation to our claim: overlaps on the "mix/reweight safety preference data in DPO"
  idea**, in the same family as Egida (existing #4) and RED QUEEN GUARD (below), but with no
  guardrail-filter baseline confirmed or found anywhere in the retrieved material. Adds to
  the pattern, now observed across several separate DPO-safety papers, that the
  training-side literature does not test itself against a post-hoc filter.

```bibtex
@article{zhao2026balanced,
  title={Improving Safety Alignment via Balanced Direct Preference Optimization},
  author={Zhao, Shiji and Wang, Mengyang and Xiong, Shukun and Chen, Fangzhou and Zhu, Qihui and Ruan, Shouwei and Xiao, Yisong and Duan, Ranjie and Chen, Xun and Wei, XingXing},
  journal={arXiv preprint arXiv:2603.22829},
  year={2026}
}
```

---

### [next-number]. (Author list not independently verified in this search pass — only the
GitHub handle `kriti-hippo` and the paper title were confirmed; do not cite an author
string without checking `arxiv.org/abs/2409.17458` directly). "RED QUEEN: Safeguarding
Large Language Models against Concealed Multi-Turn Jailbreaking." arXiv:2409.17458, Sept
2024 (preprint; venue not confirmed in this search).
- **Claims:** a multi-turn jailbreak (RED QUEEN ATTACK) that conceals malicious intent
  under a "preventing harm" framing achieves high ASR (87.6% on GPT-4o, 75.4% on
  Llama3-70B, per a 56K-example, 40-scenario, 14-category attack set); the paper's own
  defence, RED QUEEN GUARD, is a DPO fine-tune on an 11.2K preference set sampled from
  successful jailbreaks plus safe completions, cutting ASR to below 1% while preserving
  standard-benchmark performance.
- **Evaluates:** ASR pre/post the DPO defence; standard capability-benchmark retention. No
  inference-time guardrail/filter baseline found or confirmed anywhere in the retrieved
  material.
- **Relation to our claim: adjacent — same DPO-with-safety-preference-data mechanism as our
  T arm, as Egida, and as B-DPO above, applied specifically to multi-turn concealment
  jailbreaks.** No guardrail-filter comparison arm, no over-refusal metric confirmed, no
  therapy domain. Directly relevant background for our `many_shot` category (concealed
  multi-turn framing is a variant of the many-shot/persona attack family), but not a
  competing comparison.

```bibtex
@article{redqueen2024,
  title={{RED QUEEN}: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking},
  author={{Author list not independently verified --- verify at arxiv.org/abs/2409.17458 before citing}},
  journal={arXiv preprint arXiv:2409.17458},
  year={2024}
}
```

---

### [next-number]. Young, R. J. "Evaluating the Robustness of Large Language Model Safety
Guardrails Against Adversarial Attacks." arXiv:2511.22047, 2025 (preprint; University of
Nevada Las Vegas, per search-result author affiliation — not independently verified beyond
the search snippet).
- **Claims:** guardrail classifier models (10 tested, including Llama-Guard variants,
  Qwen3Guard, Granite-Guardian, Nemotron-Safety) score well on public safety benchmarks but
  generalise poorly to novel adversarial prompts — e.g. Qwen3Guard drops from 91.0% to
  33.8% accuracy on novel prompts, a 57.2-point gap the paper attributes to possible
  benchmark overfitting/contamination — and some guardrails can be induced into a
  "helpful mode" where they generate harmful content instead of classifying it.
- **Evaluates:** guardrail-only accuracy and false-positive/false-negative trade-offs on
  1,445 prompts from JailbreakBench, TrustAIRLab, and OpenAssistant, plus 145 custom
  adversarial prompts across 21 attack categories; explicitly reports a benign-accuracy-vs-
  harmful-detection trade-off (e.g. LlamaGuard variants: ~97–99% benign accuracy but only
  4.5–21.8% harmful-content catch rate in some settings) — i.e. a guardrail-side
  over-refusal/under-detection trade-off. Does not compare guardrails against DPO/RLHF-
  trained-in safety anywhere; does not measure attack success against a base model at all —
  it evaluates guardrail classifiers exclusively as standalone components.
- **Relation to our claim: different in scope (no trained-safety arm at all), but directly
  useful supporting evidence for our B3 baseline's expected weaknesses.** Strengthens the
  premise, already independently observed in our own preregistration (Revision 2: beaver-dam
  behaves as a topic detector, unstable κ across samples — 0.355/0.086/0.116), that bolted-on
  guardrail filters have real, measured, generalisation-limited failure modes rather than
  being a strawman baseline. Cite in Discussion/Limitations alongside our own beaver-dam
  findings, not as a competing claim.

```bibtex
@article{young2025evaluating,
  title={Evaluating the Robustness of Large Language Model Safety Guardrails Against Adversarial Attacks},
  author={Young, Richard J.},
  journal={arXiv preprint arXiv:2511.22047},
  year={2025}
}
```

---

### [next-number]. Kim, Wang, Yoon, Huang, Saha. "AI Content Moderation in Therapy
Conversations." arXiv preprint (cs.HC), submitted 25 May 2026 (**not confirmed
peer-reviewed**).
- **Claims:** general-purpose content-moderation guardrails, applied to therapy
  conversations, flag genuine, clinically-necessary therapeutic content as undesirable
  often enough to impair the model's capacity to function as a therapist-support system.
- **Evaluates:** an algorithmic audit of three moderation systems — **OpenAI's moderation
  endpoint, Meta's Llama Guard, and Google's ShieldGemma** — against real-life therapy
  session content, measuring how often each flags legitimate therapeutic content. Specific
  dataset size/source and quantitative flag-rate figures were **not confirmed** from the
  abstract-level material retrieved.
- **Relation to our claim: adjacent, but the single most directly supportive paper found
  for our motivation across all three search sessions, and specific to the therapy
  domain.** It is direct evidence that a bolted-on, general-purpose filter — exactly the
  class of mechanism our B3 arm instantiates, and structurally the same category as our
  `beaver-dam-7b` — mis-triggers specifically on therapy content, which is precisely the
  "bounded cost" concern our claim addresses from the opposite direction (over-refusal). It
  does **not** compare against a DPO-trained-in-safety alternative, has no ASR concept (it
  studies over-flagging of benign/necessary content, not attack success), and is not a
  controlled B3-vs-T design. **Strongly recommend citing in Introduction/Motivation** as
  evidence the guardrail-vs-training question specifically matters in therapy-support
  deployment, not only in general-purpose chat — this is a stronger domain-motivation
  citation than anything found in the two prior search sessions.

```bibtex
@article{kim2026moderation,
  title={AI Content Moderation in Therapy Conversations},
  author={Kim, Jiwon and Wang, Claire and Yoon, Taeung and Huang, Sabelle and Saha, Koustuv},
  journal={arXiv preprint},
  year={2026},
  note={cs.HC, submitted 25 May 2026; peer-review status not confirmed}
}
```

---

### [next-number]. Islam, M. A., Surdeanu, M. "A Lightweight Explainable Guardrail for
Prompt Safety" (LEG). arXiv:2602.15853, 2026 (accepted **ACL 2026** per arXiv metadata; not
independently confirmed beyond that metadata).
- **Claims:** proposes an interpretable, lightweight external guardrail module (LEG) and
  explicitly frames itself, in its own related-work discussion, against "alignment-based
  training methods" (RLHF, DPO) as the other of two branches of LLM safety work — i.e. its
  own framing names the exact two-branch taxonomy (train-in vs. bolt-on) our claim is
  structured around.
- **Evaluates:** guardrail-only performance, referencing ToxicChat and BeaverTails in its
  bibliography, and includes an over-refusal-style analysis on "benign prompts containing
  harmful words." Whether it runs any DPO/RLHF-trained comparison model as an experimental
  arm (rather than only discussing alignment-based training in prose) was **not confirmed**
  — PDF extraction failed; only a lossy AI-generated summary of the binary was available.
- **Relation to our claim: adjacent on taxonomy, unconfirmed on experiment.** Its framing
  corroborates that the field treats "alignment-based training" and "external guardrails" as
  the two standard categories — useful citation for defining that taxonomy in Related Work —
  but I cannot confirm from what was retrieved whether it runs an empirical head-to-head.
  **Flag as a follow-up full-text check** (try `arxiv.org/html/2602.15853`) before citing
  its experimental section specifically.

```bibtex
@inproceedings{islam2026lightweight,
  title={A Lightweight Explainable Guardrail for Prompt Safety},
  author={Islam, Md Asiful and Surdeanu, Mihai},
  booktitle={Proceedings of ACL 2026},
  year={2026},
  note={arXiv preprint arXiv:2602.15853; venue acceptance per arXiv metadata, not independently confirmed}
}
```

---

### [next-number]. Lee, Achananuparp, Yadav, Lim, Deng. "MHSafeEval: Role-Aware
Interaction-Level Evaluation of Mental Health Safety in Large Language Models."
arXiv:2604.17730, 2026 (preprint).
- **Claims:** proposes a role-aware, interaction-level (rather than single-turn/isolated-
  output) evaluation framework for mental-health safety in LLMs.
- **Evaluates:** safety across multi-turn mental-health dialogues with distinct participant
  roles (e.g. client vs. therapist); specific models, datasets, and whether it includes a
  guardrail-filter or DPO-training comparison were **not confirmed** — PDF extraction
  failed, abstract-level material only.
- **Relation to our claim: adjacent — same application domain (mental-health-support LLM
  safety), different question (an evaluation-methodology framework, not an
  intervention/training-vs-filter comparison).** Worth citing to situate our frozen
  red-team suite within the mental-health-LLM-safety evaluation literature, and — read
  alongside the existing file's "Slow Drift of Support" and "TherapyProbe" entries — as
  further evidence that our suite's single-turn design is a known limitation relative to
  where the field is heading (interaction-level, multi-turn evaluation).

```bibtex
@article{lee2026mhsafeeval,
  title={{MHSafeEval}: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models},
  author={Lee, Suhyun and Achananuparp, Palakorn and Yadav, Neemesh and Lim, Ee-Peng and Deng, Yang},
  journal={arXiv preprint arXiv:2604.17730},
  year={2026}
}
```

---

### [next-number]. Huang et al. (Stanford; full author list not independently verified in
this search pass). "TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in
Therapy Chatbots." arXiv:2603.18008, 2026 (preprint; listed toward ICML, acceptance not
confirmed).
- **Claims:** proposes an evaluation-to-alignment framework for therapy chatbots using a
  composite reward — a CTRS-based (Cognitive Therapy Rating Scale) clinical-fidelity score
  minus therapy-specific safety-violation penalties — optimised via **GRPO (online RL, not
  DPO)** on Qwen3-4B/1.7B therapist models, with simulated-patient conversations
  (GPT-o3-mini as patient, Claude-3.7-Sonnet as judge).
- **Evaluates:** human-rated CTRS fidelity improvement (0.10→0.60) and safety-violation
  reduction (0.38→0.20) on simulated-patient dialogues with predefined cognitive profiles.
  **Confirmed: no guardrail/filter baseline of any kind; no over-refusal or helpfulness-cost
  trade-off measured; no adversarial red-teaming at all — evaluation is limited to
  non-adversarial simulated-patient conversations.**
- **Relation to our claim: adjacent — the closest single *domain* match found across all
  three search sessions (therapy chatbot + explicit safety training), but a different
  training method (online RL/GRPO vs. our DPO) and, critically, no guardrail baseline and no
  adversarial ASR evaluation at all — the core of our claim.** Does not overlap our
  comparison; useful as the nearest "safety-trained therapy chatbot" citation, and as an
  explicit contrast case worth stating in Related Work: their safety measure is a
  violation-rate on benign simulated dialogue, ours is ASR against an adversarial frozen
  red-team suite — a methodological difference reflecting different goals (clinical fidelity
  vs. adversarial robustness), not an oversight on their part.

```bibtex
@article{huang2026therapygym,
  title={{TherapyGym}: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots},
  author={{Huang et al. --- full author list not independently verified, verify at arxiv.org/abs/2603.18008 before citing}},
  journal={arXiv preprint arXiv:2603.18008},
  year={2026}
}
```

---

### [next-number]. Verily research team (author list not independently confirmed — PDF
metadata showed a contact email `bwn@verily.com`, no full author string retrieved in this
search). "An AI-Based Behavioral Health Safety Filter and Dataset for Identifying Mental
Health Crises in Text-Based Conversations." arXiv:2510.12083, Oct 2025 (preprint); published
as "An AI-based mental health guardrail and dataset for identifying psychiatric crises in
text-based conversations," **npj Digital Medicine**, 2026, DOI 10.1038/s41746-026-02579-5
(peer-reviewed).
- **Claims:** the Verily Behavioral Health Safety Filter (VBHSF) detects 8 dimensions of
  mental-health crisis (abuse, neglect, eating-disorder behaviours, psychosis, self-harm,
  suicide, substance misuse, violence towards others) in text more accurately than two
  general-purpose guardrails.
- **Evaluates:** VBHSF vs. **NVIDIA NeMo** and **OpenAI Omni Moderation Latest** on two
  clinician-labelled datasets — the 1,800-message Verily Mental Health Crisis Dataset
  (sensitivity 0.990, specificity 0.992 for VBHSF) and a 794-message subset of the NVIDIA
  Aegis AI Content Safety Dataset.
- **Relation to our claim: DIFFERENT — a guardrail-vs-guardrail comparison, not
  guardrail-vs-trained-in-safety.** No DPO/RLHF training arm is built or compared; the
  entire comparison is between three off-the-shelf filters. Relevant supporting evidence,
  alongside the Young et al. entry above, that general-purpose moderation filters (NeMo,
  OpenAI Omni Moderation — not our beaver-dam-7b specifically, but the same mechanism class)
  underperform domain-specialised ones on mental-health content, independently corroborating
  our own preregistration finding that beaver-dam behaves as a topic detector rather than a
  harm detector on our suite. Not a competitor to our claim; useful in Discussion.

```bibtex
@article{verily2026mentalhealthfilter,
  title={An {AI}-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations},
  author={{Author list not independently verified --- verify at the npj Digital Medicine DOI 10.1038/s41746-026-02579-5 before citing}},
  journal={npj Digital Medicine},
  year={2026},
  note={Preprint version: arXiv:2510.12083 (Oct 2025)}
}
```

---

### [next-number]. Two further foundational guardrail citations, complementing the existing
file's Llama Guard entry (#25)

CLAUDE.md names three example mechanisms when describing a "bolted-on guardrail filter":
Llama Guard (already in the existing file), OpenAI moderation, and NeMo Guardrails. The
latter two were not yet in the bibliography; adding them here for completeness, since our
own B3 filter (`beaver-dam-7b`, existing entry #3, BeaverTails) is one specific instance of
this class and the paper's Related Work should define the class with its standard
citations.

**Rebedea, T., Dinu, R., Sreedhar, M. N., Parisien, C., Cohen, J.** "NeMo Guardrails: A
Toolkit for Controllable and Safe LLM Applications with Programmable Rails." EMNLP 2023
System Demonstrations (ACL Anthology `2023.emnlp-demo.40`; peer-reviewed); arXiv:2310.10501.
- **Claims:** an open-source toolkit for adding rule/dialogue-based "programmable rails" to
  LLM applications (topic restriction, dialogue-path enforcement, style constraints),
  independent of and interpretable relative to the underlying LLM.
- **Evaluates:** a systems/demo paper; illustrative case studies of controllability across
  several LLM providers, not a quantitative ASR/over-refusal benchmark, and no comparison
  against training-time safety alignment.
- **Relation to our claim: different (tooling background).** Rule-based rather than
  classifier-based, so a structurally different bolt-on mechanism than beaver-dam-7b/Llama
  Guard, but the same "external, post-hoc" category as a whole. No overlap with our
  comparison; cited only to define the guardrail category CLAUDE.md names explicitly.

```bibtex
@inproceedings{rebedea2023nemo,
  title={{NeMo} Guardrails: A Toolkit for Controllable and Safe {LLM} Applications with Programmable Rails},
  author={Rebedea, Traian and Dinu, Razvan and Sreedhar, Makesh Narsimhan and Parisien, Christopher and Cohen, Jonathan},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year={2023},
  organization={Association for Computational Linguistics}
}
```

**Markov, T., Zhang, C., Agarwal, S., Eloundou Nekoul, F., Lee, T., Adler, S., Jiang, A.,
Weng, L.** "A Holistic Approach to Undesired Content Detection in the Real World." AAAI
2023 / IAAI 2023 / EAAI 2023 (peer-reviewed) — describes the design of OpenAI's moderation
classifier (sexual content, hate, violence, self-harm, harassment categories).
- **Relation to our claim: different (tooling background), same reasoning as above.**
  Cited only to define "OpenAI moderation" as a named example in CLAUDE.md's guardrail
  category; no comparison against trained-in safety, no overlap with our claim.

```bibtex
@inproceedings{markov2023holistic,
  title={A Holistic Approach to Undesired Content Detection in the Real World},
  author={Markov, Todor and Zhang, Chong and Agarwal, Sandhini and Eloundou Nekoul, Florentine and Lee, Theodore and Adler, Steven and Jiang, Angela and Weng, Lilian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

---

### [next-number]. Chen et al. (author list not independently verified in this search pass
— see note). "Preference Learning Unlocks LLMs' Psycho-Counseling Skills."
arXiv:2502.19731, Feb 2025 (preprint; venue not confirmed). **Provenance gap-fill: this
paper appears to be missing from the existing `related_work.md` sessions despite being
directly load-bearing for our data pipeline — check before merging in case another
concurrent session already added it.**
- **Claims:** preference learning (reward modelling + alignment) on a large,
  psycho-counseling-specific preference dataset substantially improves an LLM's
  counseling-skill quality; the best aligned model reaches an 87% win rate against GPT-4o on
  the paper's own evaluation.
- **Evaluates:** introduces `PsychoCounsel-Preference` (36k preference pairs, 26,483 unique
  client utterances across 8 coarse/42 fine-grained topics) and trains
  `PsychoCounsel-Llama3-8B` plus a reward model, `PsychoCounsel-Llama3-8B-Reward`. Not a
  red-team/ASR paper and not a guardrail comparison.
- **Relation to our claim: different (data/tool provenance, not a competing experiment) —
  but a required citation.** This is the source paper for
  `Psychotherapy-LLM/PsychoCounsel-Preference` (our B2/T helpfulness-pair pool, 34,329-pair
  pool per `lab_notebook.md`) and `PsychoCounsel-Llama3-8B-Reward` (our Table 1 helpfulness
  judge, per CLAUDE.md's evaluation section). It does not touch the
  guardrail-vs-training-time-safety comparison, so it cannot scoop us, but it must be cited
  for data/model provenance alongside the existing file's Safe RLHF / PKU-SafeRLHF /
  BeaverTails entries. **Author list beyond "Chen et al." not independently verified in this
  search pass — confirm the full author string at `arxiv.org/abs/2502.19731` before the
  bibliography is finalised.**

```bibtex
@article{psychocounsel2025,
  title={Preference Learning Unlocks {LLMs}' Psycho-Counseling Skills},
  author={{Author list not independently verified --- verify at arxiv.org/abs/2502.19731 before citing}},
  journal={arXiv preprint arXiv:2502.19731},
  year={2025}
}
```

---

### [next-number]. "Goal-Conditioned DPO: Prioritizing Safety in Misaligned Instructions."
NAACL-HLT 2025 (peer-reviewed, ACL Anthology `2025.naacl-long.369`, pp. 7196–7211). **Author
list not independently verified in this search pass — verify directly on the ACL Anthology
page before citing.**
- **Claims:** goal-conditioned DPO (GC-DPO), trained to prioritise a system-prompt safety
  goal over a conflicting user prompt, cuts average jailbreak ASR from 67.1% to 5.0% on
  Vicuna-7B "without compromising general performance," per search-result summaries of the
  paper.
- **Evaluates:** ASR across jailbreak attack families, general-performance retention on
  Vicuna-7B. Its motivating framing explicitly contrasts DPO-based defenses against
  "post-processing or input perturbation" defenses, describing the latter as "prone to
  general performance degradation" and lacking robustness to varied attacks — **conceptually
  the same two-branch contrast our claim is built on**, but whether this is argued in prose
  (citing other papers) or demonstrated with an actual guardrail-model baseline as an
  experimental arm in this paper was **not confirmed** — PDF extraction failed in this
  search pass; only search-engine summaries were available.
- **Relation to our claim: adjacent, and — depending on the unconfirmed point above —
  potentially the closest match to our exact comparison found in any of the three search
  sessions.** If it turns out on full read to build and test an actual guardrail classifier
  as a baseline arm, alongside its GC-DPO model, on a shared base model, with an
  over-refusal or utility-cost metric, that would be the strongest candidate for "scoops us"
  found across this entire search. **This is therefore the single highest-priority
  follow-up full-text check from this session** — higher priority than the Springer survey
  below or the other unconfirmed entries flagged in the file's first session (SafeDPO,
  C-DPO, Equilibrate RLHF), because its framing is the closest verbal match to our claim's
  structure of any paper found. Recommend fetching `aclanthology.org/2025.naacl-long.369/`
  (the HTML abstract/landing page) or the ACL Anthology PDF directly (not via the lossy
  automated-PDF-summary route that failed in this session) before Related Work is
  finalised.

```bibtex
@inproceedings{goalconditioneddpo2025,
  title={Goal-Conditioned {DPO}: Prioritizing Safety in Misaligned Instructions},
  author={{Author list not independently verified --- verify at aclanthology.org/2025.naacl-long.369/ before citing}},
  booktitle={Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={7196--7211},
  year={2025},
  organization={Association for Computational Linguistics}
}
```

---

### [next-number]. Not independently verified enough to cite — flagged only as follow-up
items

**"Survey on LLM Safety: Attacks, Defenses, Alignment, Metrics, and Guardrails."**
*Machine Learning* (Springer), published online 2026, DOI 10.1007/s10994-026-07060-8.
Search-result snippets describe a survey unifying attacks, defenses, alignment (RLHF,
constitutional AI, instruction tuning), metrics, and guardrails (input/output filtering,
access control, compliance) as five distinct categories — exactly the scope that could
contain a prior guardrail-vs-alignment quantitative comparison. **The page redirected to a
Springer authentication wall and could not be read. Author list, exact claims, and whether
it reports any guardrail-vs-training-time-safety empirical comparison are all unverified.**
Do not cite without institutional access and a direct read.

**Two additional papers surfaced with low relevance, not developed into full entries:**
"Assessing the impact of safety guardrails on large language models using irritability
metrics" (npj Digital Medicine, 2026, DOI 10.1038/s41746-025-02333-3; author list not
retrieved) compares affective-realism (irritability psychometrics) across models chosen for
guardrail intensity, not a matched-arm ASR/over-refusal comparison — different outcome
variable entirely, low priority. "When Medical Safety Alignment Fails: A Benchmark for
Evaluating LLMs on High-Risk Medical Queries" (Li, Sun, Zhao, Li, Wu, Huang, Zheng, Ma,
arXiv:2606.28332, 2026, preprint) benchmarks general *medical* (not mental-health-specific)
safety alignment; whether it includes a guardrail baseline was not confirmed (PDF extraction
failed) — lower priority given the domain mismatch, but worth a follow-up check if medical
(not just mental-health) framing becomes relevant to Discussion.

Combined, these three follow-ups (the Springer survey, and a closer read of the B-DPO, LEG,
and Goal-Conditioned DPO entries above where PDF extraction failed) are the remaining open
items before the "no scoop found" verdict across all three search sessions can be treated
as fully closed rather than provisional.

---

## Summary table (this session's additions)

| Paper | Trained-in-safety arm? | Guardrail/filter arm? | Both, matched, same model? | Therapy/mental-health domain? | Same/adjacent/different |
|---|---|---|---|---|---|
| Jailbreaking Attacks vs. Content Safety Filters (Xin et al., ACL Findings'26) | No | **Yes** (filter pipeline) | No | No | overlaps (filter side only) |
| B-DPO (Zhao et al., 2026) | Yes, DPO | Unconfirmed (none in abstract) | No | No | overlaps (training side only) |
| RED QUEEN GUARD (2024) | Yes, DPO | Not found | No | No | adjacent |
| Guardrail robustness eval (Young, 2025) | No | Yes (10 guardrails) | No | No | different scope; supports B3 premise |
| AI Content Moderation in Therapy Conversations (Kim et al., 2026) | No | Yes (3 moderation systems) | No | **Yes** | adjacent; strongest motivation citation found |
| LEG (Islam & Surdeanu, 2026) | Discussed in prose only | Yes (proposes one) | Unconfirmed | No | adjacent (taxonomy match) |
| MHSafeEval (Lee et al., 2026) | Unconfirmed | Unconfirmed | Unconfirmed | **Yes** | adjacent (domain, different question) |
| TherapyGym (Huang et al., 2026) | Yes, GRPO (not DPO) | No | No | **Yes** | adjacent; closest domain match, no ASR |
| Verily behavioral health filter (2025/26) | No | Yes (filter-vs-filter) | No | **Yes** | different (guardrail-vs-guardrail) |
| NeMo Guardrails / OpenAI moderation (foundational) | N/A | N/A — *are* the filters | N/A | No | different (tool provenance) |
| Preference Learning Unlocks Psycho-Counseling Skills (2025) | N/A | N/A | N/A | **Yes** | different (data/tool provenance) |
| Goal-Conditioned DPO (NAACL 2025) | Yes, DPO | **Unconfirmed — highest-priority follow-up** | Unconfirmed | No | adjacent, potentially closer pending full read |
| Springer ML survey; irritability metrics; medical safety benchmark | Unverified | Unverified | Unverified | Partial | unresolved — follow-up only |

**Combined verdict across all three search sessions (this one plus the two already in
`related_work.md`):** no paper builds a matched pair of arms — one with safety mixed into
DPO, one with an otherwise-identical model wrapped in a bolted-on guardrail filter — and
compares them on ASR, over-refusal, and helpfulness, in a therapy/mental-health domain or
otherwise. The comparison exists in pieces: the training side (Egida, B-DPO, RED QUEEN
GUARD, LLUMI), the filter side (Xin et al., Young, LEG), the domain side (Kim et al.,
MHSafeEval, TherapyGym, TherapyProbe, Slow Drift of Support, VERA-MH — all in the existing
file or this one), and the one paper with a training-vs-inference-time shape but different
mechanisms (Ackerman & Panickssery, existing file) — but none of these are assembled into
one paper. The two open items most likely to change this verdict, in priority order, are:
(1) Goal-Conditioned DPO, whose framing is the closest verbal match to our claim's structure
found in any session but whose experimental section could not be read in this pass, and (2)
the paywalled Springer survey, whose scope is the closest possible match to a paper that
could contain the exact comparison our claim depends on being novel.
