# Related Work — Novelty Search Log

## Search: safety-dpo angle (2026-08-01, lit-scout)

**Question:** what has already been shown about mixing safety preference pairs into DPO
(or DPO-family preference optimisation), and — critically — has anyone already run our
exact controlled comparison (training-time safety-DPO vs a bolted-on/post-hoc guardrail
filter, measured by attack-success-rate, over-refusal, and helpfulness, on the same base
model)?

**Escalation check, stated up front: no paper found running the same controlled
comparison.** Every paper below that trains safety into DPO either (a) has no
inference-time-filter baseline at all (it compares the DPO-safety model only to the
unaligned/base model or to other *training-time* methods), or (b) is a guardrail/filter
paper with no DPO-training arm. The closest single paper (Garcia-Gasulla et al., "Egida",
arXiv 2502.13603) matches our metric triple (ASR, over-refusal via a benchmark, helpfulness/
utility) and even uses Qwen2.5-7B-Instruct as one of its four models, but its only baseline
is the unaligned base model — it does not construct or compare against a post-hoc guardrail
filter. Full-text (HTML) checked directly, not inferred from the abstract. This is evidence,
not a novelty verdict — see `closest_prior_work` / `novelty_assessment` in the structured
report to the main thread.

---

### 1. Dai, Pan, Sun, Ji, Xu, Liu, Wang, Yang. "Safe RLHF: Safe Reinforcement Learning from
Human Feedback." ICLR 2024 (peer-reviewed, poster).
- **Claims:** decoupling helpfulness and harmlessness into separate reward and cost models,
  then optimising a Lagrangian-constrained RLHF objective, yields a policy that is both more
  helpful and more harmless than single-objective RLHF/DPO baselines.
- **Evaluates:** Alpaca-7B fine-tuned over three rounds; human evaluation of helpfulness and
  harmlessness; no report (found) of an inference-time guardrail/filter baseline or of
  attack-success-rate on an adversarial red-team suite.
- **Relation to our claim: adjacent.** It establishes the foundational
  helpfulness/harmlessness-mixing result that our safety-pair provenance (PKU-SafeRLHF,
  itself downstream of this line of work) depends on, but it is RL-based (not DPO) and does
  not compare training-time safety to a bolted-on filter. Seminal / load-bearing for our
  Methods section (we already cite `ji2024pku`, `ji2024beavertails`, which are the dataset
  papers from the same group) but does not overlap our specific claim.

```bibtex
@inproceedings{dai2024safe,
  title={Safe {RLHF}: Safe Reinforcement Learning from Human Feedback},
  author={Dai, Josef and Pan, Xuehai and Sun, Ruiyang and Ji, Jiaming and Xu, Xinbo and Liu, Mickel and Wang, Yizhou and Yang, Yaodong},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

---

### 2. Ji, Hong, Zhang, Chen, Dai, Zheng, Qiu, Li, Yang. "PKU-SafeRLHF: Towards Multi-Level
Safety Alignment for LLMs with Human Preference." arXiv:2406.15513, 2024 (preprint —
not confirmed peer-reviewed at time of search).
- **Claims:** releases a large, multi-level (per-severity) harmlessness preference dataset
  with paired safe/unsafe responses across explicit harm categories.
- **Evaluates:** dataset construction and annotation quality; downstream use for Safe-RLHF-
  style training. Not a red-team ASR-vs-filter comparison paper — it is the data source, not
  a competing method paper.
- **Relation to our claim: different (data provenance, not a competing experiment).** This
  is the dataset our safety pairs are drawn from (`pref_safety.jsonl`, filtered to
  mental-health-relevant harm categories per `lab_notebook.md`). It does not run or discuss
  any guardrail-filter comparison, so it cannot scoop us, but it is a required citation for
  data provenance (already pinned as `ji2024pku` in CLAUDE.md §3).

```bibtex
@article{ji2024pku,
  title={{PKU-SafeRLHF}: Towards Multi-Level Safety Alignment for {LLMs} with Human Preference},
  author={Ji, Jiaming and Hong, Donghai and Zhang, Borong and Chen, Boyuan and Dai, Josef and Zheng, Boren and Qiu, Tianyi and Li, Boxun and Yang, Yaodong},
  journal={arXiv preprint arXiv:2406.15513},
  year={2024}
}
```

---

### 3. Ji, Liu, Dai, Pan, Zhang, Bian, Chen, Sun, Wang, Yang. "BeaverTails: Towards Improved
Safety Alignment of LLM via a Human-Preference Dataset." NeurIPS 2023 (peer-reviewed).
- **Claims:** introduces a human-preference QA-moderation dataset separating helpfulness and
  harmlessness annotations, and the associated moderation/QA-classifier model family
  (`beaver-dam-7b`, used in our project as the B3 filter).
- **Evaluates:** annotation methodology, moderation-model accuracy; not a DPO-vs-filter
  comparison.
- **Relation to our claim: different (tooling provenance).** `beaver-dam-7b` is the exact
  model we use as the B3 guardrail filter (per `preregistration.md` Revision 2). This paper
  is the provenance citation for that tool, not competing work — it does not itself compare
  a filter against a training-time-safety model.

```bibtex
@inproceedings{ji2023beavertails,
  title={BeaverTails: Towards Improved Safety Alignment of {LLM} via a Human-Preference Dataset},
  author={Ji, Jiaming and Liu, Mickel and Dai, Josef and Pan, Xuehai and Zhang, Chi and Bian, Ce and Chen, Boyuan and Sun, Ruiyang and Wang, Yizhou and Yang, Yaodong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  year={2023}
}
```

---

### 4. Garcia-Gasulla, Tormos, Arias-Duart, Hinjos, Molina-Sedano, Gururajan, Cardello.
"Efficient Safety Retrofitting Against Jailbreaking for LLMs." arXiv:2502.13603, Feb 2025
(preprint; Barcelona Supercomputing Center; **no conference/journal venue found** — appears
unpublished beyond arXiv as of this search).
- **Claims:** DPO fine-tuning on a purpose-built dataset ("Egida": 27 safety topics × 18
  attack styles) reduces attack-success-rate by 10–30 percentage points with as few as 2,000
  training pairs and low compute cost, and the reduction generalises to unseen attack styles.
- **Evaluates:** Llama-3.1-8B/70B-Instruct and **Qwen2.5-7B/72B-Instruct** (our exact base
  model family, smaller size variant), measuring (a) ASR = "proportion of unsafe answers
  over total responses" on their own attack suite, (b) over-refusal via **OR-Bench** (80K +
  Hard-1K), (c) general-purpose utility via OpenLLM-leaderboard-style benchmarks and
  ROUGE/MMLU-generative. **Confirmed by direct full-text (HTML) fetch, not abstract
  inference: the only baseline in every reported table/figure is the unaligned base model.
  No inference-time guardrail, moderation filter, or safety classifier is constructed or
  compared as a baseline anywhere in the paper.**
- **Relation to our claim: overlaps on method and metric triad, but not on the
  comparison arm.** This is the closest paper found. It shares: DPO as the mechanism, a
  purpose-built safety preference/attack dataset, the same base-model family
  (Qwen2.5-Instruct, albeit not the identical 7B B1→B2/T pipeline), and the same three-metric
  frame (ASR, over-refusal, utility/helpfulness). It differs from our claim in the load-bearing
  way: **it never builds or measures a bolted-on guardrail-filter baseline**, so it cannot
  make (and does not make) any claim about training-time safety vs. post-hoc filtering. Our
  B3 arm — DPO-helpfulness-only model wrapped in a `beaver-dam-7b` filter — has no analogue
  here. Cite as the nearest prior demonstration that "DPO safety-data mixing lowers ASR at a
  measured over-refusal/utility cost," which our B2-vs-T (not B3-vs-T) internal contrast
  would also show, but their paper stops short of the filter comparison that is our
  headline claim.
- **Hyperparameter addendum (2026-08-01, DPO-degeneration search — see the "hyperparameter
  comparator" supplementary note appended at the end of this file):** the
  HPAI-BSC/Qwen2.5-7B-Instruct-Egida-DPO model card (Hugging Face) lists **full
  fine-tuning** (not LoRA), **learning_rate ≈ 1e-7**, batch size 8, trained ~1.59h on 4×
  H100 64GB. This is 1–2 orders of magnitude lower than our LoRA DPO learning rate
  (5e-6) and is full-parameter rather than adapter-based.

```bibtex
@article{garciagasulla2025efficient,
  title={Efficient Safety Retrofitting Against Jailbreaking for {LLMs}},
  author={Garcia-Gasulla, Dario and Tormos, Adrian and Arias-Duart, Anna and Hinjos, Daniel and Molina-Sedano, Oscar and Gururajan, Ashwin Kumar and Cardello, Maria Eugenia},
  journal={arXiv preprint arXiv:2502.13603},
  year={2025}
}
```

---

### 5. Wang, Chen, et al. "More is Less: The Pitfalls of Multi-Model Synthetic Preference
Data in DPO Safety Alignment." arXiv:2504.02193, Apr 2025 (preprint; not confirmed
peer-reviewed).
- **Claims:** for DPO safety alignment, preference pairs generated by **multiple/stronger
  external models** produce higher downstream attack-success-rate than pairs generated by
  the **same model being trained** (self-generated, reward-filtered), i.e. the *source* of
  safety preference data matters more than intuition suggests, and more diverse data can
  paradoxically be less safe (reward hacking / linear separability collapse between chosen
  and rejected).
- **Evaluates:** Llama, Mistral, Qwen model families; ASR under jailbreak prompts; general
  capability benchmarks (ARC, HellaSwag, MMLU, TruthfulQA, Winogrande). No inference-time
  guardrail/filter baseline mentioned.
- **Relation to our claim: adjacent, and worth citing as a limitation/threat-to-validity.**
  Our safety pairs are sourced from PKU-SafeRLHF's own human-labelled "chosen" side (not
  model-generated), which is explicitly the *safer* condition by this paper's own finding —
  supports our provenance choice (CLAUDE.md §3's "not model-generated" rationale) rather
  than threatening it. Does not touch the filter-vs-training-time comparison; not a scoop.

```bibtex
@article{wang2025moreisless,
  title={More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in {DPO} Safety Alignment},
  author={Wang, Yifan and Chen, Runjin and others},
  journal={arXiv preprint arXiv:2504.02193},
  year={2025}
}
```
*(Author list incomplete — only first two authors verified from search snippets; full
author list not independently confirmed. State this explicitly if cited — do not present
as a complete author list without checking the arXiv page directly before submission.)*

---

### 6. Kim et al. "SafeDPO: A Simple Approach to Direct Preference Optimization with
Enhanced Safety." arXiv:2505.20065; listed as ICLR 2026 submission (**not yet a confirmed
accepted/published venue at time of search** — treat as preprint until confirmed).
- **Claims:** a closed-form-optimal-policy reformulation of constrained safety-DPO that
  needs only one extra hyperparameter (no separate reward/cost models, no online sampling),
  and reports the highest "harmless ratio" (100%) on its benchmark while keeping helpfulness
  competitive.
- **Evaluates:** PKU-SafeRLHF-30K (same dataset family as our safety pairs). No confirmed
  mention of an inference-time guardrail baseline, ASR on an adversarial red-team suite, or
  over-refusal metric in the material retrieved (abstract-level only; full text not
  independently fetched — flagged as a gap below).
- **Relation to our claim: adjacent (algorithmic variant of DPO-for-safety, same data
  family), unconfirmed on the filter comparison.** Worth a second pass before the paper is
  finalised, given how close the title/data overlap is — see gaps below.

```bibtex
@article{kim2025safedpo,
  title={{SafeDPO}: A Simple Approach to Direct Preference Optimization with Enhanced Safety},
  author={Kim, Geon-Hyeong and others},
  journal={arXiv preprint arXiv:2505.20065},
  year={2025}
}
```
*(Author list incomplete beyond first author — verify before citing.)*

---

### 7. Liu, Sun, Zheng. "Enhancing LLM Safety via Constrained Direct Preference
Optimization" (C-DPO). arXiv:2403.02475, Mar 2024 (preprint at time of search — venue not
confirmed).
- **Claims:** a dual-gradient-descent extension of DPO ("C-DPO") that adds safety
  constraints directly into the DPO objective, avoiding full constrained-RL machinery while
  giving formal safety guarantees absent from vanilla DPO.
- **Evaluates:** not independently confirmed beyond the abstract (baselines, ASR/over-refusal/
  helpfulness metrics, and models used were not retrievable from the fetched abstract text).
  **Gap: full text not checked** — recommend a follow-up fetch before Related Work is
  finalised, since "constrained DPO for safety" is close enough in framing that its
  experimental section needs verifying.
- **Relation to our claim: adjacent (unconfirmed).** Algorithmic variant of DPO for safety;
  no evidence found (nor ruled out) of a guardrail-filter baseline. Flagged as an open item.

```bibtex
@article{liu2024constrained,
  title={Enhancing {LLM} Safety via Constrained Direct Preference Optimization},
  author={Liu, Zixuan and Sun, Xiaolin and Zheng, Zizhan},
  journal={arXiv preprint arXiv:2403.02475},
  year={2024}
}
```

---

### 8. Tan, Jiang, Li, Liu, Bu, Su, Yue, Zhu, Zheng. "Equilibrate RLHF: Towards Balancing
Helpfulness-Safety Trade-off in Large Language Models." arXiv:2502.11555, Feb 2025
(preprint).
- **Claims:** naively scaling up safety training data pushes models to an "overly safe"
  rather than a "truly safe" state; proposes fine-grained data categorisation + gradient
  masking (adaptive message-wise alignment) to fix the safety/helpfulness balance with less
  data.
- **Evaluates:** not confirmed to include an inference-time guardrail baseline, ASR on a
  red-team suite, or the specific over-refusal metric framing we use (abstract-level
  material only).
- **Relation to our claim: adjacent.** Directly relevant to the "bounded cost in
  over-refusal" half of our claim (it is essentially about the same over-safety failure
  mode), but is an RLHF-variant training method paper, not a training-vs-filter comparison.
  No overlap with the B3 baseline concept found.

```bibtex
@article{tan2025equilibrate,
  title={Equilibrate {RLHF}: Towards Balancing Helpfulness-Safety Trade-off in Large Language Models},
  author={Tan, Yingshui and Jiang, Yilei and Li, Yanshi and Liu, Jiaheng and Bu, Xingyuan and Su, Wenbo and Yue, Xiangyu and Zhu, Xiaoyong and Zheng, Bo},
  journal={arXiv preprint arXiv:2502.11555},
  year={2025}
}
```

---

### 9. Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy. "XSTest: A Test Suite for
Identifying Exaggerated Safety Behaviours in Large Language Models." NAACL 2024
(peer-reviewed, Volume 1: Long Papers, pp. 5377–5400).
- **Claims:** introduces a 250 safe-prompt / 200 unsafe-contrast hand-crafted diagnostic
  suite showing that safety-tuned LLMs exhibit "exaggerated safety" (over-refusal) on
  superficially-unsafe-looking but benign prompts.
- **Evaluates:** refusal rates of various LLMs on the hand-crafted suite; establishes
  over-refusal as a distinct, measurable failure mode from under-refusal/ASR.
- **Relation to our claim: adjacent — methodological ancestor of our over-refusal metric,
  not a competing experiment.** We do not use XSTest directly (our `benign_sensitive`
  category is purpose-built and frozen), but this is the paper that established over-refusal
  as a named, measured quantity and should be cited when we justify measuring it as a
  cost term. No DPO-vs-filter comparison; no overlap with the ASR-reduction claim.

```bibtex
@inproceedings{rottger2024xstest,
  title={{XSTest}: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models},
  author={R{\"o}ttger, Paul and Kirk, Hannah and Vidgen, Bertie and Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={5377--5400},
  year={2024},
  address={Mexico City, Mexico},
  organization={Association for Computational Linguistics}
}
```

---

### 10. Cui, Chiang, Stoica, Hsieh. "OR-Bench: An Over-Refusal Benchmark for Large Language
Models." arXiv:2405.20947, 2024; **ICML 2025** (peer-reviewed, per GitHub repo tag
"[ICML 2025] Official repository").
- **Claims:** scales over-refusal measurement to 80,000 prompts across 10 rejection
  categories plus a 1,000-prompt hard subset and 600 toxic-contrast prompts, showing
  over-refusal is systematic and model-family-dependent, not just a handful of edge cases.
- **Evaluates:** many LLM families on the 80K-prompt suite; establishes over-refusal
  benchmarking methodology used by paper #4 above (Egida) as its over-refusal metric.
- **Relation to our claim: adjacent — a benchmark/metric paper, not a comparison paper.**
  Same relationship as XSTest: relevant citation for how "over-refusal" is operationalised
  in the field, useful to justify our own frozen `benign_sensitive` category's role, but no
  DPO-vs-filter content. **Re-confirmed independently under the therapy-eval angle pass
  below (2026-08-01) — same paper, same conclusion, not duplicated as a separate entry.**

```bibtex
@article{cui2024orbench,
  title={{OR-Bench}: An Over-Refusal Benchmark for Large Language Models},
  author={Cui, Justin and Chiang, Wei-Lin and Stoica, Ion and Hsieh, Cho-Jui},
  journal={arXiv preprint arXiv:2405.20947},
  year={2024}
}
```

---

### 11. Zhang, Huang, Wu, Nakov, Ji, Naseem. "Health-ORSC-Bench: A Benchmark for Measuring
Over-Refusal and Safety Completion in Health Context." ACL 2026 Findings (peer-reviewed).
- **Claims:** binary refusal boundaries are the wrong frame for health-context safety;
  proposes measuring "Safe Completion" (maximal helpfulness on dual-use/borderline health
  queries without crossing into actionable harm) jointly with over-refusal, and shows
  frontier models trade off badly between the two (up to 80% refusal on benign prompts for
  some large models; domain-specific models sacrifice safety for utility).
- **Evaluates:** 30 LLMs (GPT-5, Claude-4, Llama-4, Qwen-3-Next, etc.) against the new
  benchmark; **evaluation/benchmark-only paper — does not propose or test a training method
  (no DPO, no safety preference pairs) and does not construct or compare an inference-time
  guardrail filter.** Health scope is general (7 categories including self-harm and medical
  misinformation), not specifically mental-health/therapy.
- **Relation to our claim: adjacent (domain- and metric-relevant, not method-relevant).**
  Strengthens the case that over-refusal-vs-safety is a live, actively-benchmarked problem in
  health-adjacent LLM deployment (useful Intro/Motivation citation), but contributes no
  training method and no filter-vs-training comparison. Not a scoop.

```bibtex
@inproceedings{zhang2026healthorsc,
  title={Health-{ORSC}-Bench: A Benchmark for Measuring Over-Refusal and Safety Completion in Health Context},
  author={Zhang, Zhihao and Huang, Liting and Wu, Guanghao and Nakov, Preslav and Ji, Heng and Naseem, Usman},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```

---

### 12. Schoene, Canca. "'For Argument's Sake, Show Me How to Harm Myself!': Jailbreaking
LLMs in Suicide and Self-Harm Contexts." arXiv:2507.02990, Jul 2025 (preprint; venue not
confirmed).
- **Claims:** multi-step, prompt-level jailbreaks reliably elicit detailed harmful content
  from six widely-available LLMs when framed around suicide/self-harm.
- **Evaluates:** attack methodology and success against six LLMs (specific models,
  quantitative ASR figures, and any defence/mitigation discussion were not extractable from
  the material retrieved — **gap: full-text ASR numbers and model list not confirmed**).
  Purely an attack/red-team paper — **no defence, no DPO training, no guardrail filter is
  proposed or evaluated.**
- **Relation to our claim: adjacent (same threat domain — crisis-adjacent/self-harm attacks
  on a support-context LLM — but no method overlap).** Directly relevant citation for
  motivating why `crisis_adjacent` is one of our four frozen attack categories, and as
  evidence the threat model is real and published, not invented for this dissertation. No
  overlap with the B3-vs-T comparison.
  **Re-confirmed independently under the therapy-eval angle pass below (2026-08-01) —**
  authors: Annika M. Schoene, Cansu Canca (Northeastern University); models tested: ChatGPT-4o
  (subscription + free), Perplexity AI, Gemini Flash 2.0, Claude 3.7 Sonnet, Pi AI; attack
  succeeds via context-reframing ("academic argument"/"hypothetical") in fewer than 2 turns;
  no persona/prefilling/many-shot techniques used; results reported as binary pass/fail per
  model (5/6 failed self-harm tests, 2/6 failed suicide tests), no aggregate ASR percentage;
  no defences evaluated. Same conclusion stands: no overlap with our comparison.

```bibtex
@article{schoene2025argument,
  title={``For Argument's Sake, Show Me How to Harm Myself!'': Jailbreaking {LLMs} in Suicide and Self-Harm Contexts},
  author={Schoene, Annika M. and Canca, Cansu},
  journal={arXiv preprint arXiv:2507.02990},
  year={2025}
}
```

---

### 13. Kim, Ajit, Gong, Shimgekar, Yoo, Chandrasekharan, Saha. "LLUMI: Improving LLM
Writing Assistance for Mental Health Support with Online Community Feedback."
arXiv:2605.30273, May 2026 (preprint; venue not confirmed).
- **Claims:** DPO trained on chosen/rejected pairs derived from online mental-health
  community feedback (readability, empathy, connection, actionability, **and safety** as the
  five preference dimensions) improves LLM writing assistance for mental-health support.
- **Evaluates:** human evaluation across the five dimensions; **no confirmed mention of
  attack-success-rate, a red-team/adversarial suite, over-refusal, or a guardrail-filter
  baseline** in the material retrieved.
- **Relation to our claim: adjacent — nearest *domain* match found (mental-health support +
  DPO + a safety preference dimension), but not a method match.** LLUMI mixes a "safety"
  preference dimension into DPO for a therapy-adjacent writing-assistance task, which is
  structurally similar to our T arm's *idea*, but there is no adversarial suite, no ASR
  metric, and — critically — no guardrail-filter comparison arm. Recommend a follow-up check
  once this preprint is further along (it is very recent, May 2026, and could plausibly
  extend toward a B3-style comparison in a later revision); currently does not overlap our
  claim's evaluation design.
  **Cross-reference (2026-08-01, therapy-eval pass):** this research group (Kim, Yoo,
  Chandrasekharan, Saha) also authored PAIR-SAFE (arXiv:2601.12754, see entry #36 below), a
  runtime-auditing guardrail for mental-health support — the same group therefore has papers
  on both sides of our comparison (a DPO-with-safety-dimension paper here, and a separate
  runtime-guardrail paper there) but has not, in either paper, combined the two into a direct
  comparison. Worth watching for a future paper from this group that does.

```bibtex
@article{kim2026llumi,
  title={{LLUMI}: Improving {LLM} Writing Assistance for Mental Health Support with Online Community Feedback},
  author={Kim, Jiwon and Ajit, Maya and Gong, Sherry and Shimgekar, Soorya Ram and Yoo, Dong Whi and Chandrasekharan, Eshwar and Saha, Koustuv},
  journal={arXiv preprint arXiv:2605.30273},
  year={2026}
}
```

---

### 14. Cheng, Kang, Jiang, Sun, Pan. "The Slow Drift of Support: Boundary Failures in
Multi-Turn Mental Health LLM Dialogues." arXiv:2601.14269, Jan 2026 (preprint; not
peer-reviewed).
- **Claims:** LLM safety boundaries in mental-health dialogues erode gradually over
  multi-turn conversations under adversarial "pressure" probing, rather than failing via
  single obvious violations; adaptive probing more than halves the number of turns needed
  to induce a boundary crossing (9.21 → 4.64 turns).
- **Evaluates:** 3 LLMs, 50 virtual patient profiles, up to 20 dialogue rounds, two pressure
  methods; pure evaluation/stress-testing paper — **no DPO, no safety preference data, no
  guardrail filter proposed or tested.**
- **Relation to our claim: adjacent — a limitation-relevant citation, not a competing
  method.** Directly relevant to Discussion/Limitations: our frozen suite is single-turn
  (per CLAUDE.md's `many_shot` category aside, it is not an adaptive multi-turn red-team),
  so this paper is useful evidence that multi-turn drift is a known, separate failure mode
  our evaluation does not capture. No overlap with the training-vs-filter claim.

```bibtex
@article{cheng2026slowdrift,
  title={The Slow Drift of Support: Boundary Failures in Multi-Turn Mental Health {LLM} Dialogues},
  author={Cheng, Youyou and Kang, Zhuangwei and Jiang, Kerry and Sun, Chenyu and Pan, Qiyang},
  journal={arXiv preprint arXiv:2601.14269},
  year={2026}
}
```

---

### 15. Ramnauth, Brscic, Scassellati. "Robotics-Inspired Guardrails for Foundation Models
in Socially Sensitive Domains." arXiv:2605.19940, May 2026 (preprint; stated as under review
at JAIR).
- **Claims:** reframes LLM guardrails as "runtime behavioural control over interaction
  trajectories" (a robotics-inspired framing) rather than static classifiers, for deployment
  in socially sensitive domains including mental health and in-home autism therapy.
- **Evaluates:** conceptual/framework paper; explicitly categorises existing approaches as
  "training-time alignment, prompting, decoding constraints, and post-hoc moderation" but
  does not run a controlled experiment comparing them against each other on ASR/over-refusal/
  helpfulness.
- **Relation to our claim: adjacent — same taxonomy of defence types we use (training-time
  vs. post-hoc), same broad application domain (mental-health-adjacent), but purely
  conceptual — no models trained, no numbers reported.** Good citation for framing "why
  compare training-time safety to a post-hoc filter" in the Intro, not a competing empirical
  result.

```bibtex
@article{ramnauth2026robotics,
  title={Robotics-Inspired Guardrails for Foundation Models in Socially Sensitive Domains},
  author={Ramnauth, Rebecca and Brscic, Drazen and Scassellati, Brian},
  journal={arXiv preprint arXiv:2605.19940},
  year={2026}
}
```

---

## Summary table (safety-DPO angle)

| # | Paper | DPO/pref-opt safety mixing? | Guardrail-filter baseline? | ASR + over-refusal + helpfulness triad? | Same/adjacent/different |
|---|---|---|---|---|---|
| 1 | Safe RLHF (Dai et al., ICLR'24) | RLHF, not DPO | No | No (human eval only) | adjacent |
| 2 | PKU-SafeRLHF (Ji et al., 2024) | data only | No | N/A (dataset paper) | different (provenance) |
| 3 | BeaverTails (Ji et al., NeurIPS'23) | data/tool only | N/A — *is* a filter model | N/A (dataset/tool paper) | different (tool provenance) |
| 4 | Egida (Garcia-Gasulla et al., 2025) | **Yes, DPO** | **No** | **Yes** (closest match) | **overlaps method+metrics, not comparison arm** |
| 5 | More is Less (Wang et al., 2025) | Yes, DPO (data-source ablation) | No | ASR + capability, no over-refusal | adjacent |
| 6 | SafeDPO (Kim et al., 2025/26) | Yes, DPO variant | Unconfirmed | Unconfirmed | adjacent (unconfirmed) |
| 7 | C-DPO (Liu et al., 2024) | Yes, DPO variant | Unconfirmed | Unconfirmed | adjacent (unconfirmed — gap) |
| 8 | Equilibrate RLHF (Tan et al., 2025) | RLHF variant | Unconfirmed | Unconfirmed | adjacent |
| 9 | XSTest (Röttger et al., NAACL'24) | No | No | over-refusal only | adjacent (metric ancestor) |
| 10 | OR-Bench (Cui et al., ICML'25) | No | No | over-refusal only | adjacent (metric ancestor) |
| 11 | Health-ORSC-Bench (Zhang et al., ACL'26 Findings) | No | No | over-refusal + safe-completion, no ASR | adjacent (domain+metric) |
| 12 | Schoene & Canca (2025) | No | No | attack only, no defence | adjacent (threat domain) |
| 13 | LLUMI (Kim et al., 2026) | Yes, DPO (domain: mental health) | No | No ASR/over-refusal | adjacent (closest domain match) |
| 14 | Slow Drift of Support (Cheng et al., 2026) | No | No | No (multi-turn stress test) | adjacent (limitation citation) |
| 15 | Robotics-Inspired Guardrails (Ramnauth et al., 2026) | No | Conceptual only | No | adjacent (framing citation) |

**No row in this table is "same."** No paper found builds both a DPO-safety-trained arm and
a post-hoc/bolted-on guardrail-filter arm on an identical base model and compares them on
ASR + over-refusal + helpfulness. Row 4 (Egida) is the nearest on method/metrics; rows 13
(LLUMI) and 4 are the nearest on domain overlap with "therapy-support model," but neither
runs the filter comparison.

## Open gaps / follow-ups before Related Work is finalised

- Rows 6, 7, 8, 12 were assessed from abstracts / partial fetches only — full-text baseline
  tables were not independently confirmed for a guardrail-filter comparison. Recommend a
  second-pass full-text check on these four before claiming the novelty gap is closed,
  since "unconfirmed" is not the same as "absent."
- Row 5's full author list beyond the first two names was not independently verified —
  do not cite a complete author list without checking the arXiv listing directly.
- LLUMI (row 13) is a very recent (May 2026) preprint in the closest domain match; worth a
  citation-tracking check closer to submission in case a revision adds an adversarial/ASR
  evaluation.
- General caveat: search tool summaries (not full PDFs) were relied on for most papers;
  where stated "unconfirmed" or "not independently verified" above, treat as a to-do, not a
  cleared check.

---

## Search: attacks-shallow angle (2026-08-01, lit-scout)

**Question:** what does the jailbreak/attack-taxonomy literature say about our five red-team
categories (`prefilling`, `persona`, `many_shot`, `crisis_adjacent`, `benign_sensitive`), what
does "shallow safety alignment" work say about *why* safety training is fragile to exactly
these attack shapes, and — again — has anyone already run our controlled comparison
(training-time DPO safety vs a bolted-on inference-time filter, on ASR + over-refusal +
helpfulness, same base model)?

**Escalation check, stated up front: still no paper found running the identical controlled
comparison.** One candidate looked close enough on first pass to require a full-text check:
**Ackerman & Panickssery, "Mitigating Many-Shot Jailbreaking" (arXiv:2504.09604, 2025,
preprint)** compares a *training-time* fine-tuning defense against an *inference-time*
defense for many-shot jailbreaking specifically, and reports over-refusal (OR-Bench) and
benign-capability cost alongside ASR — on the surface, the same shape of comparison as our
B3-vs-T. **On full-text verification it does not scoop us**, for three load-bearing reasons,
stated precisely because this is the closest miss found across both search sessions: (1) its
"fine-tuning" defense is **SFT on refusal demonstrations, not DPO on preference pairs**; (2)
its "inference-time" defense is **input-tag sanitization** (stripping/detecting fake role
tags in the prompt), **not a classifier-based bolted-on guardrail/moderation filter** in the
Llama-Guard/beaver-dam sense — there is no separate safety model scoring or replacing outputs;
(3) it is scoped to **many-shot jailbreaking only** (no prefilling, persona, or crisis-adjacent
attacks) on a **general-purpose assistant** (Llama-3.1-8B-Instruct), not a therapy-support
model. It is the nearest thing to "someone already tested training-time-safety-vs-guardrail
with a cost metric" that this search turned up, so it is flagged here explicitly rather than
buried in the table — but the specific mechanisms being compared (SFT-refusal vs. tag-strip)
are different from ours (DPO-safety-pairs vs. classifier-filter), so it neighbours rather than
subsumes or directly overlaps our claim. See entry #23 below and the summary table.

No other paper in this second search session came closer. The Qi et al. "shallow safety
alignment" line (#16) is the most important *conceptual* prior work for this angle — it does
not run our comparison, but it supplies a mechanistic explanation for exactly the vulnerability
our `prefilling` category is designed to test, and (per a partial/conflicting extraction —
see caveat under #16) may show that plain DPO safety training does not, by itself, robustly
fix prefilling-attack susceptibility the way targeted "recovery" data augmentation does. If
that reading holds up on a full re-check, it is a citable reason our own Table 2 might show
`prefilling` as the category where T's advantage over B3 is smallest — worth flagging to the
main thread as a specific, falsifiable prediction to watch for in Results, not just a citation.

---

### 16. Qi, Panda, Lyu, Ma, Roy, Beirami, Mittal, Henderson. "Safety Alignment Should Be Made
More Than Just a Few Tokens Deep." ICLR 2025 (peer-reviewed; received an ICLR 2025
Outstanding Paper Award). arXiv:2406.05946 (first posted June 2024).
- **Claims:** current LLM safety alignment is "shallow" — it primarily reshapes the output
  distribution over only the first few generated tokens — and this single mechanism explains
  susceptibility to adversarial-suffix attacks, **prefilling attacks**, decoding-parameter
  attacks, and fine-tuning attacks all at once. Proposes (a) augmenting safety training data
  with "safety recovery examples" that teach the model to transition from an already-harmful
  prefix back to a refusal at multiple token depths, and (b) a fine-tuning objective that
  constrains probability divergence more strongly at early token positions than later ones.
- **Evaluates:** Llama-2 (primarily) on the HEx-PHI harmful-instruction benchmark; prefilling
  attacks specifically with 5–40 prefilled harmful tokens. On one extraction of the paper,
  their augmented-data model's prefilling ASR was reported as low single digits versus a much
  higher initial-model ASR (**figures obtained via one WebFetch pass on the arXiv HTML;
  a second, independent extraction of the same source could not confirm these numbers and
  reported "not present" — treat the exact percentages as unverified pending a manual PDF
  read, not as a settled citation**). Utility cost is reported via AlpacaEval win-rate
  ("only marginally lower") and via benign-task fine-tuning (Samsum/SQL/GSM8K) showing
  "comparable utility to standard SFT." **No DPO experiment was confirmed** — the released
  code (`Unispac/shallow-vs-deep-alignment`) documents SFT-based data augmentation and a
  constrained fine-tuning objective, not DPO, and one search-engine summary's claim of a "DPO
  ablation" could not be corroborated against the paper text or code repository directly.
  **No inference-time guardrail/filter baseline is used anywhere in the paper** — the entire
  contribution is training-time (data augmentation + a modified loss). No therapy/mental-health
  application is discussed.
- **Relation to our claim: adjacent, and load-bearing for mechanism, not for the filter
  comparison.** This is the best available mechanistic account of *why* a DPO-safety-trained
  model (our T arm) might still fail specifically on `prefilling` attacks: DPO on
  preference pairs is not obviously the same intervention as their "recovery-example" data
  augmentation, and if T's safety pairs do not specifically contain post-harmful-prefix
  recovery continuations, shallow-alignment theory predicts prefilling remains a weak point
  for T relative to `persona` or plain instruction-following attacks. This paper never builds
  or compares a guardrail filter, so it does not overlap the B3-vs-T comparison itself — but
  it is essential Related Work / Discussion material for interpreting a `prefilling` row in
  Table 2 that looks worse than the others for T, and for motivating why we should not expect
  DPO alone to be a complete fix. **Caveat, stated for auditability:** two independent
  full-text extraction attempts on this paper gave conflicting answers about whether a DPO
  ablation exists in the paper body; do not cite specific DPO-related numbers from this paper
  without a manual read of the PDF before the Related Work section is finalised.

```bibtex
@inproceedings{qi2025safety,
  title={Safety Alignment Should Be Made More Than Just a Few Tokens Deep},
  author={Qi, Xiangyu and Panda, Ashwinee and Lyu, Kaifeng and Ma, Xiao and Roy, Subhrajit and Beirami, Ahmad and Mittal, Prateek and Henderson, Peter},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

### 17. Wei, Haghtalab, Steinhardt. "Jailbroken: How Does LLM Safety Training Fail?"
NeurIPS 2023 (peer-reviewed). arXiv:2307.02483.
- **Claims:** proposes two root-cause failure modes for LLM safety training —
  **competing objectives** (helpfulness/instruction-following pressure conflicts with safety
  training) and **mismatched generalization** (safety training doesn't cover a capability
  domain the model has, e.g. base64, low-resource languages, or unusual formatting) — and
  uses them to systematically design jailbreaks against GPT-4 and Claude v1.3.
- **Evaluates:** curated jailbreak prompts spanning both failure-mode categories against
  frontier closed models circa 2023; qualitative and quantitative success analysis. No DPO
  training arm, no guardrail-filter baseline, no therapy domain.
- **Relation to our claim: different/foundational background, not a competing experiment.**
  This is the seminal taxonomy underlying most later attack-category work (including
  persona-modulation and many-shot framings, both of which can be read as instances of
  "mismatched generalization"). Useful Introduction/Background citation for *why* an attack
  taxonomy like ours (four categories + benign-sensitive) is a reasonable way to operationalise
  jailbreak risk, but it neither trains a DPO-safety model nor tests a bolted-on filter, so it
  cannot overlap our specific claim.

```bibtex
@inproceedings{wei2023jailbroken,
  title={Jailbroken: How Does {LLM} Safety Training Fail?},
  author={Wei, Alexander and Haghtalab, Nika and Steinhardt, Jacob},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  year={2023}
}
```

---

### 18. Anil, Durmus, Panickssery, Sharma, Benton, Kundu, Batson, Tong, Mu, Ford, Mosconi,
Agrawal, Schaeffer, Bashkansky, Svenningsen, Lambert, Radhakrishnan, Denison, Hubinger, Bai,
Bricken, Maxwell, Schiefer, Sully, Tamkin, Lanham, Nguyen, Korbak, Kaplan, Ganguli, Bowman,
Perez, Grosse, Duvenaud. "Many-shot Jailbreaking." NeurIPS 2024 (peer-reviewed; Anthropic).
- **Claims:** prompting a long-context LLM with hundreds of fabricated dialogue turns showing
  an assistant persona complying with progressively more harmful requests reliably overrides
  safety training via in-context learning; attack effectiveness follows a power law in the
  number of "shots," up to hundreds of demonstrations, and transfers across closed frontier
  models from multiple providers.
- **Evaluates:** closed-weight frontier models (Anthropic, OpenAI, Google DeepMind) across
  varied harm categories and long-context regimes. This is the direct source paper for our
  `many_shot` attack category's premise. Discusses mitigations only at a high level (context
  length limits, in-context defenses, targeted fine-tuning) — **does not run a controlled
  training-time-vs-inference-time comparison, and no guardrail/filter model is built or
  tested.** No therapy/mental-health domain.
- **Relation to our claim: adjacent — direct taxonomic source for one of our four frozen
  attack categories, not a competing method paper.** Essential citation for `many_shot`'s
  provenance and design rationale; no overlap with the B3-vs-T comparison itself.

```bibtex
@inproceedings{anil2024manyshot,
  title={Many-shot Jailbreaking},
  author={Anil, Cem and Durmus, Esin and Panickssery, Nina and Sharma, Mrinank and Benton, Joe and Kundu, Sandipan and Batson, Joshua and Tong, Meg and Mu, Jesse and Ford, Daniel and Mosconi, Francesco and Agrawal, Rajashree and Schaeffer, Rylan and Bashkansky, Naomi and Svenningsen, Samuel and Lambert, Mike and Radhakrishnan, Ansh and Denison, Carson and Hubinger, Evan J. and Bai, Yuntao and Bricken, Trenton and Maxwell, Timothy and Schiefer, Nicholas and Sully, James and Tamkin, Alex and Lanham, Tamera and Nguyen, Karina and Korbak, Tomasz and Kaplan, Jared and Ganguli, Deep and Bowman, Samuel R. and Perez, Ethan and Grosse, Roger and Duvenaud, David},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={37},
  year={2024}
}
```

---

### 19. Shah, Feuillade-Montixi, Pour, Tagade, Casper, Rando. "Scalable and Transferable
Black-Box Jailbreaks for Language Models via Persona Modulation." arXiv:2311.03348, Nov 2023
(preprint; presented at the SoLaR workshop, NeurIPS 2023 — **workshop paper, not a
main-track peer-reviewed venue**).
- **Claims:** automatically generating "persona" system prompts that steer a target model
  into a personality willing to comply with harmful requests is a scalable, transferable
  black-box jailbreak technique.
- **Evaluates:** GPT-4 (harmful-completion rate rose from 0.23% baseline to 42.5% under
  automated persona-modulation attacks — a 185x increase), with transfer tested on Claude 2
  (61.0%) and Vicuna (35.9%). No training-time defense, no guardrail-filter comparison, no
  therapy domain.
- **Relation to our claim: adjacent — direct taxonomic source for our `persona` attack
  category, not a competing method paper.** Establishes persona modulation as an empirically
  strong, transferable attack family, which is why it is one of our four frozen categories.
  No overlap with the B3-vs-T comparison; the paper contains no defense evaluation at all.

```bibtex
@article{shah2023persona,
  title={Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation},
  author={Shah, Rusheb and Feuillade-Montixi, Quentin and Pour, Soroush and Tagade, Arush and Casper, Stephen and Rando, Javier},
  journal={arXiv preprint arXiv:2311.03348},
  year={2023}
}
```

---

### 20. Andriushchenko, Croce, Flammarion. "Jailbreaking Leading Safety-Aligned LLMs with
Simple Adaptive Attacks." ICLR 2025 (peer-reviewed). arXiv:2404.02151.
- **Claims:** adaptivity (tailoring a lightweight attack template + logprob-guided random
  search per target model) achieves 100% attack success rate against many leading
  safety-aligned LLMs (GPT-3.5, GPT-4o, Llama-2/3-Chat, Gemma-7B, and — via prefilling on
  Claude's API — Claude 3.5 Sonnet), often with far less engineering effort than prior
  optimization-based attacks.
- **Evaluates:** ASR on HarmBench-style harmful-behavior sets across the above closed and
  open models. **Directly relevant to our `prefilling` category**: for APIs that allow
  response prefilling, the paper shows prompting-template + prefilling alone reaches 100%
  ASR on Claude 3.5 Sonnet with no optimization required. No DPO training arm on the attacker
  or defender side, no guardrail-filter baseline, no therapy domain — this is a pure attack
  paper.
- **Relation to our claim: adjacent — direct technique source for our `prefilling` category,
  not a competing method paper.** Strong evidence the prefilling attack shape we test is a
  live, high-ASR technique in the general literature (not just a theoretical concern), useful
  for motivating the category's inclusion and difficulty. No overlap with the B3-vs-T
  comparison; the paper is attack-only.

```bibtex
@inproceedings{andriushchenko2025adaptive,
  title={Jailbreaking Leading Safety-Aligned {LLM}s with Simple Adaptive Attacks},
  author={Andriushchenko, Maksym and Croce, Francesco and Flammarion, Nicolas},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

### 21. Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson. "Fine-tuning Aligned Language Models
Compromises Safety, Even When Users Do Not Intend To!" ICLR 2024 (peer-reviewed, Oral).
arXiv:2310.03693.
- **Claims:** fine-tuning an aligned model — even on data that looks entirely benign, or with
  as few as 10 adversarially-designed examples costing under $0.20 via a commercial API —
  measurably degrades safety behavior, and current safety infrastructure (aligned at
  release time) does not defend against risks introduced by a *subsequent* fine-tuning step.
- **Evaluates:** GPT-3.5 Turbo (via fine-tuning API) and open Llama-2-family models; harmful-
  instruction compliance rate before/after fine-tuning under three threat models (explicitly
  harmful examples, identity-shifting examples, and fully benign data). No guardrail-filter
  baseline is compared; no DPO-with-safety-pairs arm; no therapy domain.
- **Relation to our claim: adjacent, and important for Discussion/Limitations, in the
  opposite direction from our claim.** Our project fine-tunes *with* safety data added (T),
  which this paper's framing would predict should help, not hurt — but this paper is the
  primary citation for the general risk that *any* further fine-tuning step (including our
  own B1→B2/T LoRA pipeline) can move safety behavior, intentionally or not, and should be
  cited when we discuss why LoRA rank/target-module choices and data composition are safety-
  relevant design decisions, not just performance ones. No overlap with the B3-vs-T filter
  comparison itself.

```bibtex
@inproceedings{qi2024finetuning,
  title={Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!},
  author={Qi, Xiangyu and Zeng, Yi and Xie, Tinghao and Chen, Pin-Yu and Jia, Ruoxi and Mittal, Prateek and Henderson, Peter},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

---

### 22. Zou, Phan, Wang, Duenas, Lin, Andriushchenko, Wang, Kolter, Fredrikson, Hendrycks.
"Improving Alignment and Robustness with Circuit Breakers." NeurIPS 2024 (peer-reviewed).
arXiv:2406.04313.
- **Claims:** a representation-engineering method ("circuit breakers") that directly detects
  and redirects harmful internal representations mid-generation, rather than relying on
  refusal training, substantially improves robustness to unseen/adaptive jailbreak attacks
  (including prefix-injection-style attacks) while preserving benign utility, for both
  text-only and multimodal models and for LLM agents.
- **Evaluates:** Mistral-7B, Llama-3-8B, and others, against a range of jailbreak attacks
  including token-level (suffix/prefix) attacks; reports harmfulness reduction and utility
  retention. No guardrail-filter baseline is compared against as a separate inference-time
  system (circuit breakers modify the model itself, so this is a training-time method); no
  therapy domain.
- **Relation to our claim: adjacent — an alternative training-time robustness mechanism,
  not DPO and not evaluated against a bolted-on filter.** Worth citing in Related Work as
  evidence that "training-time intervention beats surface-level refusal training" is an
  active research direction with methods other than preference optimization; useful contrast
  case (representation-level vs. preference-based training-time safety) but does not touch
  our filter-vs-training-time comparison. Reported limitation in follow-up work (not this
  paper itself): circuit breakers can produce incoherent rather than substantively safe
  responses and may not generalize to subtler "reasoning exploit" jailbreaks — worth a
  cautious mention if cited, per one third-party survey summary (unconfirmed against the
  primary paper directly).

```bibtex
@inproceedings{zou2024circuitbreakers,
  title={Improving Alignment and Robustness with Circuit Breakers},
  author={Zou, Andy and Phan, Long and Wang, Justin and Duenas, Derek and Lin, Maxwell and Andriushchenko, Maksym and Wang, Rowan and Kolter, Zico and Fredrikson, Matt and Hendrycks, Dan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={37},
  year={2024}
}
```

---

### 23. Ackerman, Panickssery. "Mitigating Many-Shot Jailbreaking." arXiv:2504.09604, Apr
2025 (revised May 2025); preprint, **not confirmed peer-reviewed at time of search.**
- **Claims:** fine-tuning a model on many-shot-jailbreak attempts paired with a correct final
  "recovery" refusal, combined with input-tag sanitization at inference time, substantially
  reduces many-shot-jailbreak susceptibility while preserving (and, on OR-Bench, improving)
  benign-capability and over-refusal metrics.
- **Evaluates (full-text-confirmed, not abstract-only):** Llama-3.1-8B-Instruct (general-
  purpose, not therapy/mental-health). Fine-tuning defense = **supervised fine-tuning on
  refusal demonstrations** (loss computed on the final "recovery" assistant turn), **not
  DPO on preference pairs**. Inference-time defense = **input sanitization** (stripping or
  detecting fake role tags used to fabricate the many-shot dialogue), **not a classifier-
  based bolted-on guardrail/moderation model** (no Llama-Guard- or beaver-dam-style safety
  classifier is used or built). Metrics: ASR on many-shot attacks (LLM-judged and
  paired-preference), **plus OR-Bench over-refusal** and MT-Bench/in-context-learning/Chatbot-
  Arena benign-capability retention — i.e. the *shape* of the evaluation (ASR + over-refusal +
  capability cost) resembles ours, but the two arms being compared are SFT-refusal-training vs.
  tag-sanitization, not DPO-safety-pairs vs. classifier-filter. Attack scope is many-shot
  jailbreaking only — no prefilling, persona, or crisis-adjacent attacks are tested.
- **Relation to our claim: the closest near-miss found in this search session — flagged
  explicitly, but does not subsume or directly overlap our claim.** It is the only paper found
  in either search session that (a) compares a training-time defense against an
  inference-time defense (b) on the same base model (c) while reporting an over-refusal/
  capability cost alongside ASR — structurally the nearest thing to our B3-vs-T design. It
  does **not** scoop us because: the training-time arm is SFT-on-refusals, not DPO-on-
  preference-pairs (our methodological contribution is specifically preference-pair
  construction and mixing ratios, not refusal-demonstration SFT); the inference-time arm is
  prompt/input sanitization, not an output-side moderation classifier (our B3 is a
  `beaver-dam-7b`-style QA-moderation filter applied to *outputs*, a materially different
  defense mechanism than stripping input tags); the attack scope is many-shot only, versus our
  four categories; and the domain is general-assistant, not therapy-support. **Recommend
  citing this explicitly in Related Work as "the nearest prior comparison of a training-time
  vs. inference-time defense we are aware of, and precisely how our design differs from it,"**
  since a reviewer who knows this paper will otherwise ask why it wasn't discussed.

```bibtex
@article{ackerman2025mitigating,
  title={Mitigating Many-Shot Jailbreaking},
  author={Ackerman, Christopher M. and Panickssery, Nina},
  journal={arXiv preprint arXiv:2504.09604},
  year={2025}
}
```

---

### 24. Ma, Pan, Farahmand. "PANDAS: Improving Many-shot Jailbreaking via Positive
Affirmation, Negative Demonstration, and Adaptive Sampling." ICML 2025 (Spotlight,
peer-reviewed). arXiv:2502.01925.
- **Claims:** a hybrid attack that inserts positive-affirmation phrases and negative
  demonstrations into many-shot jailbreak dialogues, with adaptive per-topic sampling,
  substantially strengthens many-shot jailbreaking over the original Anil et al. attack,
  and introduces "ManyHarm," a dataset of harmful QA pairs for constructing many-shot
  attacks at scale.
- **Evaluates:** attack success rate on AdvBench/AdvBench50/HarmBench-style targets, up to
  256-shot prompts; separately evaluates several defenses (perplexity filtering,
  Retokenization, SmoothLLM) and finds them largely ineffective against many-shot attacks at
  scale. No DPO-safety-training defense is proposed or tested; no output-side moderation-
  classifier guardrail is tested; no therapy domain.
- **Relation to our claim: adjacent — attack-strengthening work on the same `many_shot`
  category, not a defense-comparison paper.** Relevant for motivating why `many_shot` is
  worth including as a frozen category (it is an actively-improving attack family, not a
  static/solved one) and as evidence that simple input-level defenses (perplexity,
  retokenization) are known to be weak against many-shot attacks specifically — useful
  context if our B3 guardrail filter turns out to underperform on `many_shot` items in
  Table 2. No overlap with the B3-vs-T comparison itself.

```bibtex
@inproceedings{ma2025pandas,
  title={{PANDAS}: Improving Many-shot Jailbreaking via Positive Affirmation, Negative Demonstration, and Adaptive Sampling},
  author={Ma, Avery and Pan, Yangchen and Farahmand, Amir-massoud},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```

---

### 25. Inan, Upasani, Chi, Rungta, Iyer, Mao, Tontchev, Hu, Fuller, Testuggine, Khabsa.
"Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations." arXiv:2312.06674,
Dec 2023 (Meta technical report / preprint — **not a peer-reviewed conference/journal
publication**, released as an open model + report).
- **Claims:** an LLM fine-tuned on a fixed hazard taxonomy can classify both user prompts and
  model responses for safety risk, functioning as a general-purpose input/output guardrail
  that matches or exceeds existing moderation tools on standard benchmarks (OpenAI Moderation
  eval, ToxicChat) and can adapt to new taxonomies via prompting without retraining.
- **Evaluates:** classification accuracy against existing moderation baselines; does not
  itself compare against any training-time safety-alignment method (it is purely a filter/
  classifier paper), and does not measure over-refusal or helpfulness cost in a downstream
  assistant setting. No therapy domain.
- **Relation to our claim: different (tooling background) — the archetypal example of the
  "bolted-on guardrail filter" category our B3 arm instantiates, not competing work.** Our
  own project's judge/filter design explicitly names Llama Guard as the primary fallback
  filter mechanism (per `preregistration.md`, later superseded by `beaver-dam-7b` for B3 per
  Revision 2 — see project files). This paper is the canonical citation for *what a bolted-on
  guardrail filter is* in the current literature and should be cited when defining the B3 arm
  in Methods, but it contains no DPO/training-time comparison of any kind, so it cannot
  overlap or subsume our claim — if anything, its existence as an accepted, widely-adopted
  reference architecture for "guardrail filter" strengthens the case that B3 is a realistic,
  citable baseline to beat, not a straw man.

```bibtex
@article{inan2023llamaguard,
  title={Llama Guard: {LLM}-based Input-Output Safeguard for Human-{AI} Conversations},
  author={Inan, Hakan and Upasani, Kartikeya and Chi, Jianfeng and Rungta, Rashi and Iyer, Krithika and Mao, Yuning and Tontchev, Michael and Hu, Qing and Fuller, Brian and Testuggine, Davide and Khabsa, Madian},
  journal={arXiv preprint arXiv:2312.06674},
  year={2023}
}
```

---

### 26. Belli et al. "VERA-MH: Validation of Ethical and Responsible AI in Mental Health."
arXiv:2605.13318, May 2026 (preprint; not confirmed peer-reviewed). Companion validation
study: Bentley et al., "AI Chatbot Suicide Risk Detection and Response: Human Validation
Study of the Open-Source VERA-MH Safety Evaluation," arXiv:2602.05088, 2026 (preprint).
- **Claims:** a clinically-developed, multi-turn, dynamically-simulated evaluation
  (conversation simulation → conversation judging → model rating) for chatbot safety around
  suicidal-ideation risk specifically, intended as an open-source automated benchmark; the
  companion validation paper reports clinician-vs-clinician and clinician-vs-LLM-judge
  agreement (Krippendorff's α ≈ 0.77–0.81, raw agreement ≥ 81%) supporting the benchmark's
  reliability.
- **Evaluates:** simulated multi-turn conversations rated by both licensed clinicians and
  LLM judges against a clinical rubric; a benchmark/evaluation-methodology paper —
  **does not propose or compare a DPO-safety-training arm or an inference-time guardrail-
  filter arm; it evaluates existing chatbots' behavior, not a defense comparison.**
- **Relation to our claim: adjacent — directly relevant to the `crisis_adjacent` category
  and its clinical-rubric grounding, not a competing method.** Independent evidence (separate
  from the NICE/WHO/C-SSRS sources our own crisis rubric already cites per
  `preregistration.md` §8) that clinically-grounded, rubric-based multi-turn safety judging
  for suicide-risk contexts is an active, credible methodology in the field as of 2026 —
  useful citation to justify our own crisis-behaviour judge's rubric-based design, and a
  pointer to a more rigorous (clinician-validated, multi-turn) evaluation than our frozen
  single-turn `crisis_adjacent` set achieves. No overlap with the B3-vs-T filter comparison.
  **Correction (2026-08-01, therapy-eval pass, independently re-fetched from arXiv:2602.05088's
  own abstract page directly, resolving the "incomplete author list" gap flagged below):**
  full confirmed author list for the Bentley et al. companion validation paper is Kate H.
  Bentley, Luca Belli, Adam M. Chekroud, Emily J. Ward, Emily R. Dworkin, Emily Van Ark,
  Kelly M. Johnston, Will Alexander, Millard Brown, Matt Hawrilenko; published venue is
  **JMIR AI, 2026;5** (not solely an arXiv preprint as previously stated); reported
  reliability figures, quoting the abstract directly, are **chance-corrected inter-rater
  reliability (not Krippendorff's α as characterized above) = 0.77 clinician-clinician and
  0.81 LLM-judge-vs-clinician-consensus** — worth reconciling the "Krippendorff's α" wording
  above against this IRR wording before citing a specific statistic name in the paper. The
  main VERA-MH concept paper (arXiv:2605.13318 / possibly arXiv:2510.15297, a "VERA-MH
  Concept Paper" surfaced separately) was not independently re-verified in this pass — the
  arXiv ID for the *concept* paper itself may need a second check, since two different IDs
  for what may be the same or a related VERA-MH paper surfaced across the two search
  sessions (2605.13318 here vs. 2510.15297 found independently); do not treat either ID as
  fully settled without one more direct check.

```bibtex
@article{belli2026veramh,
  title={{VERA-MH}: Validation of Ethical and Responsible {AI} in Mental Health},
  author={Belli, Luca and others},
  journal={arXiv preprint arXiv:2605.13318},
  year={2026}
}
@article{bentley2026veramhvalidation,
  title={{AI} Chatbot Suicide Risk Detection and Response: Human Validation Study of the Open-Source {VERA-MH} Safety Evaluation},
  author={Bentley, Kate H. and Belli, Luca and Chekroud, Adam M. and Ward, Emily J. and Dworkin, Emily R. and Van Ark, Emily and Johnston, Kelly M. and Alexander, Will and Brown, Millard and Hawrilenko, Matt},
  journal={JMIR AI},
  volume={5},
  year={2026},
  note={arXiv:2602.05088}
}
```

---

### 27. Chandra, Navneet, Zhang. "TherapyProbe: Generating Design Knowledge for Relational
Safety in Mental Health Chatbots Through Adversarial Simulation." CHI EA 2026 (peer-reviewed
extended abstract).
- **Claims:** many mental-health-chatbot safety failures are *relational* and emerge only
  over multi-turn interaction (not from any single response), introduces a six-category
  taxonomy of such failures (Crisis Escalation Failure, Validation Spiral, Boundary Erosion,
  Harmful Guidance, Empathy Fatigue, Alliance Rupture) and 23 sub-pattern archetypes,
  discovered via Monte-Carlo-Tree-Search-driven adversarial conversation simulation.
- **Evaluates:** three open-source mental-health chatbots (MentaLLaMA-13B, Mental Health
  Mistral-7B, ChatCounselor) using an LLM "Patient Agent" (Llama-3-8B-Instruct) and an LLM
  failure detector (MentaLLaMA-7B); reports that all three pass single-turn crisis checks
  (85–92%) but fail under multi-turn adversarial simulation. **Pure red-team/evaluation
  paper — no DPO training, no safety preference pairs, no guardrail filter of any kind is
  built or tested; no training-time-vs-inference-time comparison.**
- **Relation to our claim: adjacent — closest single paper found on *domain* (mental-health-
  support chatbot safety, multi-turn adversarial simulation) but zero method overlap.**
  Reinforces the same point as Cheng et al. (#14, prior search): the specific failure mode
  this project's frozen suite is best positioned to catch is single-turn, category-scoped
  attacks, while a body of 2026 work (this paper, Cheng et al., VERA-MH) is converging on
  multi-turn, relational, or dynamically-simulated crisis evaluation as the more clinically
  faithful standard. Directly citable in Limitations to explain why the frozen suite's
  `crisis_adjacent` category, being single-turn, understates the harder multi-turn risk
  surface this and related papers document. No overlap with the B3-vs-T filter comparison —
  no filter or DPO arm exists in this paper at all.

```bibtex
@inproceedings{chandra2026therapyprobe,
  title={{TherapyProbe}: Generating Design Knowledge for Relational Safety in Mental Health Chatbots Through Adversarial Simulation},
  author={Chandra, Joydeep and Navneet, Satyam Kumar and Zhang, Yong},
  booktitle={Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (CHI EA)},
  year={2026}
}
```

---

## Summary table (attacks-shallow angle)

| # | Paper | Attack category it sources | Trains DPO w/ safety pairs? | Bolted-on output-filter baseline? | Same/adjacent/neighbour |
|---|---|---|---|---|---|
| 16 | Shallow Safety Alignment (Qi et al., ICLR'25) | `prefilling` (mechanism) | No (SFT + constrained loss; DPO claim unverified) | No | adjacent (mechanistic, load-bearing) |
| 17 | Jailbroken (Wei et al., NeurIPS'23) | all (root-cause taxonomy) | No | No | different (foundational background) |
| 18 | Many-shot Jailbreaking (Anil et al., NeurIPS'24) | `many_shot` (source) | No | No | adjacent (taxonomic source) |
| 19 | Persona Modulation (Shah et al., 2023, workshop) | `persona` (source) | No | No | adjacent (taxonomic source) |
| 20 | Adaptive Attacks (Andriushchenko et al., ICLR'25) | `prefilling` (technique) | No | No | adjacent (technique source) |
| 21 | Fine-tuning Compromises Safety (Qi et al., ICLR'24) | none (fine-tuning risk, general) | No | No | adjacent (opposite-direction risk) |
| 22 | Circuit Breakers (Zou et al., NeurIPS'24) | prefix/suffix attacks generally | No (representation engineering, not DPO) | No (modifies model, not a separate filter) | adjacent (alternative training-time method) |
| 23 | Mitigating MSJ (Ackerman & Panickssery, 2025, preprint) | `many_shot` only | **No — SFT on refusals** | **No — input-tag sanitization, not output classifier** | **nearest-miss: neighbours, does not subsume** |
| 24 | PANDAS (Ma et al., ICML'25) | `many_shot` (attack-strengthening) | No | No (tests unrelated defenses, finds them weak) | adjacent (attack-side) |
| 25 | Llama Guard (Inan et al., 2023, preprint) | N/A — is a filter architecture | No | **N/A — is the filter category itself** | different (tooling/background) |
| 26 | VERA-MH (Belli et al. + Bentley et al., 2026) | `crisis_adjacent` (methodology) | No | No | adjacent (crisis-eval methodology) |
| 27 | TherapyProbe (Chandra et al., CHI EA'26) | `crisis_adjacent`/domain (multi-turn) | No | No | adjacent (closest domain match, multi-turn) |

**No row in this second table is "same" either.** Entry #23 (Ackerman & Panickssery) is the
one paper across both search sessions that structurally resembles a training-time-vs-
inference-time defense comparison with a cost metric, and is therefore the single most
important citation to discuss explicitly and distinguish from our design in the paper's
Related Work section — not because it scoops us, but because a knowledgeable reviewer will
ask about it if it is absent.

## Open gaps / follow-ups from this session

- **#16 (Qi et al., shallow alignment) numeric claims are unverified.** Two independent
  full-text extraction attempts gave conflicting answers on whether the paper includes a DPO
  ablation and on the exact prefilling-ASR numbers. Do not cite specific percentages from this
  paper in the final Related Work or Discussion without a manual read of the PDF (saved
  locally by the fetch tool during this session; recommend opening it directly rather than
  re-running an automated extraction).
- **#26 (VERA-MH / Bentley et al.) author list gap is now resolved** (see the correction
  embedded in entry #26 above, added by the therapy-eval search pass) — the companion
  validation paper's full author list and JMIR AI venue are confirmed; the *concept* paper's
  exact arXiv ID is still not settled (two candidate IDs found across sessions: 2605.13318 and
  2510.15297) and should get one more direct check before the bibliography is finalised.
- Entry #22 (Circuit Breakers)'s reported limitation ("produces incoherent responses,"
  "fails on subtler reasoning exploits") came from a third-party survey summary, not the
  primary paper — flagged as unconfirmed if it is used in Discussion.
- Did not find any published attack taxonomy that uses the same four-category split
  (`prefilling`, `persona`, `many_shot`, `crisis_adjacent`) as a named, fixed set — each
  category has a clear individual source paper (#16/#20, #19, #18/#24, #26/#27 respectively)
  but the specific four-way grouping plus a `benign_sensitive` over-refusal companion set
  appears to be this project's own construction, not adopted wholesale from a prior taxonomy.
  This is a design-provenance observation, not a novelty verdict — reported to the main
  thread as evidence only.

---

## Search: therapy-eval angle (2026-08-01, lit-scout)

**Question:** what has been shown specifically about LLM evaluation in therapy/counselling/
mental-health contexts — safety benchmarks, crisis-response evaluation, over-refusal in
sensitive domains, work using ESConv/CounselChat/PsychoCounsel data, suicide/self-harm
response-quality work, and clinical-guidance-derived evaluation rubrics — and, critically,
has anyone in *this* literature already run our exact controlled comparison (training-time
safety-DPO vs a bolted-on guardrail filter, measured by ASR, with over-refusal and
helpfulness as bounded costs)?

**Escalation check, stated up front: no paper found in this pass runs the same controlled
comparison, in the therapy/mental-health domain or otherwise verified.** The therapy-domain
literature has grown substantially in the last six months (Jan–Apr 2026) and splits the
problem into pieces we combine, but none combines them: (a) adversarial/multi-turn attack
taxonomies purpose-built for mental-health LLMs exist (MHSafeEval, arXiv:2604.17730) but
with zero training-arm comparison — every evaluated system is an off-the-shelf commercial
model; (b) DPO-based, multi-objective safety alignment for therapy exists (Beikzadeh et al.,
arXiv:2602.16053) but has no adversarial attack suite and no guardrail-filter baseline —
its comparisons are DPO variants against each other, SFT, and parameter-merging; (c)
clinically-grounded guardrail classifiers purpose-built for mental health exist (MindGuard,
arXiv:2602.00950; PAIR-SAFE, arXiv:2601.12754) but are evaluated standalone or against other
guardrails, never against a training-time-safety arm. See the per-paper entries below for
exactly what each does and does not cover, and the cross-referenced entries in the earlier
sections of this file (#10 OR-Bench, #12 Schoene & Canca, #26 VERA-MH — re-confirmed or
corrected independently in this pass, not duplicated as separate entries).

**A separate, unverifiable lead surfaced during search synthesis and is explicitly NOT
reported as a finding** — see "Note on an unverified lead" immediately below. It claimed a
single-paper DPO-vs-guardrail ASR comparison (~59.7% → 3.0%) but could not be traced to a
real citable source; treat as noise, not signal, unless independently re-found. Not
therapy-domain either way, so if a general-domain DPO-vs-guardrail search angle exists
elsewhere in this project's novelty search, that subagent should re-check it independently.

### Note on an unverified lead (not reported as a finding)

During search synthesis, one WebSearch summary asserted a specific number — "DPO-based
preference alignment significantly lowered Attack Success Rate (ASR) from 59.7% to 3.0%...
in contrast, post-hoc guardrail filters show substantial vulnerability" — framed as if from
a single paper running exactly our controlled comparison. I attempted to trace this to a
specific citable source and could not: follow-up searches surfaced only unrelated papers (a
DPO *attack* method against GPT-4 variants, and separate LlamaGuard robustness-degradation
studies under obfuscation attacks) whose numbers did not match. This looks like a
search-summarizer artifact that stitched together unrelated papers' statistics rather than a
real single-paper finding. **Per instruction not to invent a citation, this is not listed as
a paper.**

---

### 28. Zhang, Eack, Chen. "Preference Learning Unlocks LLMs' Psycho-Counseling Skills."
ACL 2026 (v1 arXiv Feb 2025, camera-ready v2 Apr 2026). arXiv:2502.19731.
- **Claims:** LLMs lack effective psycho-counseling skill due to a lack of high-quality
  preference data; the authors build a 36k-pair preference dataset scored against seven
  counseling-quality principles and DPO/DPO-Iter-train a model reaching an 87% win rate
  against GPT-4o.
- **Evaluates:** LLM-as-judge (GPT-4o) plus human-expert validation (88.5% agreement with
  synthetic labels) of counseling-response quality across 8 coarse / 42 fine-grained topics;
  reward-model accuracy 97.8–98.1%.
- **Relation to our claim: different axis, but direct provenance dependency — must be
  cited regardless of the novelty question.** This is the paper that introduces
  `PsychoCounsel-Preference` and `PsychoCounsel-Llama3-8B-Reward` — the exact helpfulness-
  pairs source (B2/T's helpfulness pairs) and helpfulness judge (Table 1's helpfulness
  column) our project uses. It contains **no** adversarial/attack evaluation, no
  over-refusal metric, and no guardrail comparison — it is entirely about counseling
  quality, not safety, and cannot compete with or subsume our claim.

```bibtex
@inproceedings{zhang2026psychocounsel,
  title={Preference Learning Unlocks {LLMs}' Psycho-Counseling Skills},
  author={Zhang, Mian and Eack, Shaun M. and Chen, Zhiyu Zoey},
  booktitle={Proceedings of ACL 2026},
  year={2026},
  note={arXiv:2502.19731}
}
```

---

### 29. Arnaiz-Rodriguez, Baidal, Derner, Layton Annable, Ball, Ince, Perez Vallejos, Oliver.
"Between Help and Harm: An Evaluation of Mental Health Crisis Handling by LLMs."
JMIR Mental Health, accepted for publication (DOI 10.2196/88435); arXiv:2509.24857.
- **Claims:** introduces a unified six-category crisis taxonomy, a curated dataset of 2,252
  examples (from >239,000 inputs across 12 HF mental-health datasets), and a clinical
  response-assessment protocol; audits five off-the-shelf LLMs (incl. gpt-4o-mini,
  gpt-5-nano, deepseek-v3.2-exp, Llama-4-Scout, grok-4-fast) for crisis-response
  appropriateness on a 5-point Likert scale (1=harmful to 5=appropriate).
- **Evaluates:** automatic crisis classification (3 LLMs) plus human/LLM appropriateness
  grading of 5 deployed models' responses to real crisis-adjacent inputs; **no model
  training of any kind, verified against the paper's own abstract.**
- **Relation to our claim: adjacent.** Purely an evaluation study of existing commercial
  models — no DPO, no safety fine-tuning, no guardrail-filter comparison. Close to our
  `crisis_adjacent` category in spirit (clinically-grounded appropriateness grading of
  crisis responses) and citable to justify our use of a clinical-rubric-based crisis judge,
  but it does not touch the B3-vs-T comparison at all.

```bibtex
@article{arnaizrodriguez2026betweenhelpharm,
  title={Between Help and Harm: An Evaluation of Mental Health Crisis Handling by {LLM}s},
  author={Arnaiz-Rodriguez, Adrian and Baidal, Miguel and Derner, Erik and Layton Annable, Jenn and Ball, Mark and Ince, Mark and Perez Vallejos, Elvira and Oliver, Nuria},
  journal={JMIR Mental Health},
  year={2026},
  note={arXiv:2509.24857; DOI: 10.2196/88435}
}
```

---

### 30. Steenstra, Pedrelli, Shi, Marsella, Bickmore. "Assessing Risks of Large Language
Models in Mental Health Support: A Framework for Automated Clinical AI Red Teaming."
arXiv:2602.19948 (Feb 2026), venue not stated beyond arXiv (cs.CL).
- **Claims:** proposes simulation-based "clinical red teaming" pairing AI psychotherapists
  with simulated patient agents (dynamic cognitive-affective models), evaluated against a
  "quality of care and risk ontology"; applies it to Alcohol Use Disorder as a test case
  across six agents (incl. ChatGPT, Gemini, Character AI) and 15 clinically-validated
  patient personas (N=369 simulated sessions).
- **Evaluates:** iatrogenic risks — validation of patient delusions ("AI Psychosis"),
  failure to de-escalate suicide risk — via simulated multi-session therapy, validated with
  a stakeholder panel (N=9: engineers, red-teamers, clinicians, policy experts).
- **Relation to our claim: adjacent.** Confirmed via the paper's own abstract: it evaluates
  deployed models via simulation and does **not** compare training-time safety methods
  against guardrail/filter defenses. Relevant as prior art for "red-teaming for therapy
  LLMs" as a category and for the "iatrogenic risk" framing, but a different mechanism
  (simulated multi-session patient agents vs. our frozen static adversarial-prompt suite)
  and no training-arm comparison.

```bibtex
@misc{steenstra2026assessingrisks,
  title={Assessing Risks of Large Language Models in Mental Health Support: A Framework for Automated Clinical AI Red Teaming},
  author={Steenstra, Ian and Pedrelli, Paola and Shi, Weiyan and Marsella, Stacy and Bickmore, Timothy W.},
  year={2026},
  note={arXiv:2602.19948}
}
```

---

### 31. Lee, Achananuparp, Yadav, Lim, Deng. "MHSafeEval: Role-Aware Interaction-Level
Evaluation of Mental Health Safety in Large Language Models." arXiv:2604.17730 (Apr 2026),
venue not stated beyond arXiv.
- **Claims:** introduces R-MHSafe, a role-aware harm taxonomy for AI counselors
  (perpetrator / instigator / facilitator / enabler roles × clinically-grounded harm
  categories), and MHSafeEval, a closed-loop agent-based framework that adapts
  general-purpose jailbreak methods (**PAIR** — single-turn iterative refinement; **TAP** —
  tree-structured search; **X-Teaming** — multi-agent multi-turn) into the mental-health
  domain and scores **trajectory-level** harm across multi-turn adversarial dialogues.
- **Evaluates:** 8 off-the-shelf models (GPT-3.5 Turbo, Llama 3.1 8B Instruct, Gemini 2.5
  Flash, Claude Haiku 4.5, DeepSeek V3.2, Gemma 4 26B, MiniMax M2.5, MiMo V2 Flash).
  Metrics, confirmed directly from the paper's own text via a second, careful fetch:
  **Attack Success Rate** ("proportion of trajectories assigned Severity ≥2 by the safety
  judge over the full dialogue"), **Refusal Rate** ("proportion of explicit refusals,"
  explicitly caveated that "refusal does not necessarily imply safety"), and **Clinical
  Comprehension**. **No Over-Refusal Rate and no separate Helpfulness metric are present**
  — an earlier, less careful automated summary of this paper wrongly claimed an ASR+ORR
  F1-combined metric existed; that claim is retracted here after a second verification pass.
  No ESConv/CounselChat/PsychoCounsel; the paper uses its own "Client-Ψ-CM" simulated client
  profiles.
- **Relation to our claim: closest neighbour on the evaluation-methodology side, but not
  the same comparison.** It shares real structural overlap with our red-team suite — an
  attack taxonomy purpose-built for the therapy domain, an ASR-style metric, and explicit
  recognition that refusal ≠ safety (which parallels our own reasoning for needing two
  separate safety judges, per `preregistration.md` §4). But it evaluates only off-the-shelf
  commercial/open models with **no training intervention of any kind** — no DPO, no SFT, no
  fine-tuning arm — and **no guardrail-filter arm**, so it cannot be a B3-vs-T comparison.
  It also has no over-refusal or helpfulness metric, so it cannot show the "bounded cost"
  half of our claim even in principle. Very recent (Apr 2026) and worth close reading in
  Methods when justifying our own attack-taxonomy design choices, and worth citing as
  evidence that "refusal ≠ safety" is an independently-reached conclusion in this
  literature, not an idiosyncrasy of our judge design.

```bibtex
@misc{lee2026mhsafeeval,
  title={{MHSafeEval}: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models},
  author={Lee, Suhyun and Achananuparp, Palakorn and Yadav, Neemesh and Lim, Ee-Peng and Deng, Yang},
  year={2026},
  note={arXiv:2604.17730}
}
```

---

### 32. Beikzadeh, Asadollah Salmanpour, Suvarna, Sankararaman, Malgaroli, Sarrafzadeh,
Gabriel. "Multi-Objective Alignment of Language Models for Personalized Psychotherapy."
arXiv:2602.16053 (Feb 2026), venue not stated beyond arXiv.
- **Claims:** surveys 335 people with lived mental-health experience for
  therapeutic-dimension preference rankings, trains 6 reward models (empathy, safety,
  active listening, self-motivated change, trust/rapport, patient autonomy) on top of an
  EPITOME-derived corpus with Mistral-7B-generated candidate responses, and shows
  Multi-Objective DPO (MODPO) balances objectives better than single-objective DPO (e.g.
  77.6% empathy / 62.6% safety for MODPO vs 93.6% empathy / 47.8% safety for single-
  objective DPO_Empathy).
- **Evaluates:** 600 test questions rated by 50 held-out patient personas via pairwise
  GPT-5 comparison; blinded clinician validation (6 licensed professionals, 100 questions);
  external toxicity checks (ModelCitizens, Perspective API — MODPO 0.5% toxic vs base 1.2%).
- **Relation to our claim: overlaps in method space, not in comparison — the closest
  methodological cousin of our T arm found in either search pass.** Like us, this is
  DPO-based, therapy-domain, and treats "safety" as a preference dimension mixed alongside
  others rather than bolted on afterward. But: (a) no adversarial/red-team attack suite —
  "safety" is measured via persona/clinician preference on ordinary therapeutic dialogue,
  not attack resistance; (b) no guardrail/filter baseline of any kind — the comparison is
  against single-objective DPO, SFT, and parameter-merging ("DPO_Soup"), never a bolted-on
  filter; (c) no over-refusal metric. It does **not** run our comparison, but a reviewer
  could plausibly ask why we didn't adopt multi-objective DPO instead of a two-arm
  (helpfulness-only vs. helpfulness+safety) design — worth a sentence in Related Work /
  Discussion addressing that design choice explicitly.

```bibtex
@misc{beikzadeh2026multiobjective,
  title={Multi-Objective Alignment of Language Models for Personalized Psychotherapy},
  author={Beikzadeh, Mehrab and Asadollah Salmanpour, Yasaman and Suvarna, Ashima and Sankararaman, Sriram and Malgaroli, Matteo and Sarrafzadeh, Majid and Gabriel, Saadia},
  year={2026},
  note={arXiv:2602.16053}
}
```

---

### 33. Suhas BN, Sherrill, Arriaga, Wiese, Abdullah. "AI Safety Training Can be Clinically
Harmful." arXiv:2604.23445 (Apr 2026), venue not stated beyond arXiv.
- **Claims:** RLHF-style safety alignment actively conflicts with evidence-based therapy
  protocols in specific modalities — e.g. in Prolonged Exposure (PE) therapy for trauma, a
  safety-trained model saying "you are safe" or inserting crisis resources is a **protocol
  violation**, because the therapeutic mechanism requires the patient to experience and
  tolerate distress. Evaluated 4 generative models on 250 PE scenarios + 146 CBT
  cognitive-restructuring exercises (+29 severity-escalated variants), scored by a 3-judge
  LLM panel: surface acknowledgment stayed near-perfect (0.91–1.00) while therapeutic
  appropriateness collapsed to 0.22–0.33 at highest severity for 3/4 models, with protocol
  fidelity reaching zero for two models under escalation.
- **Evaluates:** protocol fidelity, hallucination risk, behavioral consistency, crisis
  safety, demographic robustness — a five-axis framework mapped to FDA SaMD / EU AI Act
  categories; no DPO training, no guardrail filter, no attack taxonomy.
- **Relation to our claim: adjacent, but load-bearing for our Discussion.** Not an
  attack-resistance study and does not touch our B3-vs-T comparison at all. But it is
  directly relevant to the "bounded cost" half of our claim: independent evidence that
  safety-trained *caution behaviors themselves* (reassurance, crisis-resource insertion,
  refusal to engage with distressing content) can be a clinical harm in specific
  therapeutic contexts, not merely an over-refusal inconvenience. Strengthens the case for
  measuring over-refusal carefully (as our pre-registration's Revision 4 already does) and
  is a citable counterweight in Discussion/Limitations: reducing ASR without a bounded cost
  is not automatically "safe" even when it looks safe by our metrics, because over-cautious
  behavior has its own clinical downside this paper documents concretely. Cite as
  motivation, not as competing prior art.

```bibtex
@misc{suhas2026aisafetyharmful,
  title={{AI} Safety Training Can be Clinically Harmful},
  author={Suhas BN and Sherrill, Andrew M. and Arriaga, Rosa I. and Wiese, Chris W. and Abdullah, Saeed},
  year={2026},
  note={arXiv:2604.23445}
}
```

---

### 34. Shen, Fong, Jiang, Wang, Tang, Xu, Zhao, Xu, Liu, Hu, Dwyer, Ge. "PsychEthicsBench:
Evaluating Large Language Models Against Australian Mental Health Ethics." arXiv:2601.03578
(Jan 2026), venue not stated beyond arXiv.
- **Claims:** first principle-grounded benchmark based on Australian psychology/psychiatry
  ethics guidelines; multiple-choice + open-ended tasks with fine-grained ethicality
  annotations, across 14 models.
- **Evaluates:** ethical knowledge and behavioral response quality against professional
  ethics codes, not attack resistance. Key finding, directly relevant to our judge-design
  reasoning: **"refusal rates are poor indicators of ethical behavior,"** revealing "a
  significant divergence between safety triggers and clinical appropriateness," and
  **"domain-specific fine-tuning can degrade ethical robustness"** — several specialized
  mental-health models underperform their base backbones on ethical alignment.
- **Relation to our claim: adjacent, with a specific caution worth citing.** No DPO-vs-
  guardrail comparison, no ASR/over-refusal/helpfulness triple. But the "fine-tuning can
  degrade robustness" and "refusal ≠ appropriateness" findings are independent, converging
  support for two design choices already in our project: (a) the reason our
  pre-registration uses two separate safety judges rather than a refusal-based proxy, and
  (b) a caution to state explicitly in Limitations — our T arm is itself a fine-tuned
  model, and this paper is evidence that domain fine-tuning is not guaranteed to be
  safety-neutral even when it targets safety pairs.

```bibtex
@misc{shen2026psychethicsbench,
  title={{PsychEthicsBench}: Evaluating Large Language Models Against Australian Mental Health Ethics},
  author={Shen, Yaling and Fong, Stephanie and Jiang, Yiwen and Wang, Zimu and Tang, Feilong and Xu, Qingyang and Zhao, Xiangyu and Xu, Zhongxing and Liu, Jiahe and Hu, Jinpeng and Dwyer, Dominic and Ge, Zongyuan},
  year={2026},
  note={arXiv:2601.03578}
}
```

---

### 35. Farinhas, Guerreiro, Pombal, Martins, Melton, Conway, Dochat, D'Eon, Rei.
"MindGuard: Guardrail Classifiers for Multi-Turn Mental Health Support." arXiv:2602.00950
(Feb 2026), Sword Health.
- **Venue note:** a search hit also surfaced an npj Digital Medicine article ("An AI-based
  mental health guardrail and dataset for identifying psychiatric crises in text-based
  conversations") that may be a published/renamed version of this same work, but the
  nature.com page required an authentication redirect that was not followed, **so this
  venue link is unverified** — reporting only the arXiv version as confirmed.
- **Claims:** a clinically-grounded risk taxonomy (developed with PhD-level psychologists)
  distinguishing actionable harm from safe therapeutic disclosure; trains lightweight
  guardrail *classifiers* (4B/8B) on synthetic two-agent dialogues; releases
  `MindGuard-testset`, 1,134 turns from 67 real multi-turn conversations, clinician-annotated.
- **Evaluates:** confirmed via the paper's own abstract — classifiers "reduce false
  positives at high-recall operating points and, when paired with clinician language
  models, help achieve **lower attack success and harmful engagement rates in adversarial
  multi-turn interactions compared to general-purpose safeguards.**"
- **Relation to our claim: neighbours our B3 arm, not our comparison.** MindGuard *is* a
  bolted-on guardrail-filter approach for exactly our domain (mental health support), and
  its own reported comparison is **guardrail vs. guardrail** — MindGuard vs.
  "general-purpose safeguards" — never guardrail vs. training-time safety alignment. Useful,
  citable evidence that clinically-tuned guardrail classifiers exist and outperform generic
  ones (like `beaver-dam-7b`, which our own `preregistration.md` Revision 2 found behaves as
  a topic detector rather than a harm detector) — worth a Limitations sentence: our B3
  baseline uses a generic QA-moderation filter rather than a clinically-tuned one like
  MindGuard, so B3 may be a weaker guardrail than the best currently available, which if
  anything strengthens a positive result for T and should be stated as a caveat if the
  result is null.

```bibtex
@misc{farinhas2026mindguard,
  title={{MindGuard}: Guardrail Classifiers for Multi-Turn Mental Health Support},
  author={Farinhas, Ant{\'o}nio and Guerreiro, Nuno M. and Pombal, Jos{\'e} and Martins, Pedro Henrique and Melton, Laura and Conway, Alex and Dochat, Cara and D'Eon, Maya and Rei, Ricardo},
  year={2026},
  note={arXiv:2602.00950}
}
```

---

### 36. Kim, Rodriguez, Yoo, Chandrasekharan, Saha. "PAIR-SAFE: A Paired-Agent Approach for
Runtime Auditing and Refining AI-Mediated Mental Health Support." arXiv:2601.12754 (Jan
2026), venue not stated beyond arXiv.
- **Claims:** a Responder + supervisory Judge agent pair, the Judge grounded in the
  clinically validated Motivational Interviewing Treatment Integrity (MITI-4) framework,
  issuing ALLOW/REVISE decisions to refine responses **at runtime**. Judge-supervised
  interactions improved Partnership, Seek Collaboration, and overall Relational MITI
  dimensions, confirmed by qualitative expert evaluation.
- **Evaluates:** simulated counseling interactions via a support-seeker simulator derived
  from human-annotated motivational-interviewing data; no adversarial red-teaming, no
  attack-success-rate, no over-refusal or helpfulness benchmark reported.
- **Relation to our claim: neighbours our B3 arm.** Another inference-time, guardrail-style
  intervention in the same domain (audit-and-revise rather than filter-and-replace),
  explicitly framed against "implicit alignment through training or prompting" as offering
  "limited transparency and runtime accountability" — i.e. it argues *for* the
  runtime-supervision approach our claim argues against (relatively) on the ASR dimension.
  It does not measure attack resistance at all, so it neither supports nor contradicts our
  result, but is worth citing as evidence that runtime-supervision architectures are an
  active alternative design line to the one we test against (B3), and that our choice of
  `beaver-dam-7b`-as-filter is one of several plausible bolted-on-guardrail designs, not the
  only one. Same research group as LLUMI (entry #13 above) — see the cross-reference note
  there.

```bibtex
@misc{kim2026pairsafe,
  title={{PAIR-SAFE}: A Paired-Agent Approach for Runtime Auditing and Refining {AI}-Mediated Mental Health Support},
  author={Kim, Jiwon and Rodriguez, Violeta J. and Yoo, Dong Whi and Chandrasekharan, Eshwar and Saha, Koustuv},
  year={2026},
  note={arXiv:2601.12754}
}
```

---

### 37. Luo, Laban. "DialogGuard: Multi-Agent Psychosocial Safety Evaluation of Sensitive
LLM Responses." arXiv:2512.02282 (Dec 2025), venue not stated beyond arXiv (also has an
OpenReview page, suggesting a workshop/conference submission).
- **Claims:** a multi-agent LLM-as-judge framework (single-agent scoring, dual-agent
  correction, multi-agent debate, majority voting) for scoring five psychosocial-risk
  dimensions (privacy, discrimination, mental manipulation, psychological harm, insult),
  grounded in a three-level rubric usable by humans or LLM judges. Validated against
  **PKU-SafeRLHF** human safety annotations — the same upstream data source as our safety
  preference pairs, though used here for judging rather than DPO training.
- **Evaluates:** detection accuracy against PKU-SafeRLHF human labels; formative study with
  12 practitioners on auditing/supervision usefulness.
- **Relation to our claim: neighbours, shared-data note.** Not a therapy-specific dataset
  study (broader "sensitive LLM responses," psychosocial harm generally) and not a
  training-vs-filter comparison — it is an evaluation/judging tool, not an alignment method.
  Worth a one-line note in Methods/Related Work: PKU-SafeRLHF is independently used
  elsewhere in this literature as *judging* ground truth, mild corroboration that it is a
  reasonable provenance choice for our safety pairs (already justified independently by
  `preregistration.md` §3's provenance rule), not that it removes the need for our own audit.

```bibtex
@misc{luo2025dialogguard,
  title={{DialogGuard}: Multi-Agent Psychosocial Safety Evaluation of Sensitive {LLM} Responses},
  author={Luo, Han and Laban, Guy},
  year={2025},
  note={arXiv:2512.02282}
}
```

---

### 38. Byun, Lipschutz, Minton, Lott, Choi. "CRADLE Bench: A Clinician-Annotated Benchmark
for Multi-Faceted Mental Health Crisis and Safety Risk Detection." EACL 2026 (peer-reviewed;
v1 arXiv Oct 2025, v2 Jan 2026). arXiv:2510.23845.
- **Claims:** a seven-crisis-type benchmark (suicide ideation, rape, domestic violence,
  child abuse, sexual harassment, and others), the first to incorporate temporal labels;
  600 clinician-annotated evaluation examples, 420 development examples, ~4K
  ensemble-labeled training examples; fine-tunes six crisis-**detection** classifiers on
  consensus/unanimous ensemble-agreement subsets (up to 5.7-point gains over baselines).
- **Evaluates:** crisis-detection accuracy — this is a **classification** task (does this
  text contain a crisis?), not a generation-safety/response-quality task, and not an
  attack-resistance evaluation.
- **Relation to our claim: adjacent — different task type.** No DPO, no guardrail-filter
  baseline in the sense we mean (it fine-tunes detector classifiers, not a bolted-on
  moderation filter wrapping a separate generator), no ASR, no over-refusal, no
  helpfulness. Relevant as a methodological parallel — clinician-annotated,
  clinical-standards-aligned crisis taxonomy construction — supporting our own use of a
  cited clinical-guidance rubric (NICE NG225, WHO mhGAP-IG 2.0, etc., per
  `preregistration.md` §8) for the `crisis_adjacent` category, but the task (detecting a
  crisis in user text) is structurally different from ours (judging whether a *model's
  response* to a crisis is appropriate).

```bibtex
@inproceedings{byun2026cradlebench,
  title={{CRADLE} Bench: A Clinician-Annotated Benchmark for Multi-Faceted Mental Health Crisis and Safety Risk Detection},
  author={Byun, Grace and Lipschutz, Rebecca and Minton, Sean T. and Lott, Abigail and Choi, Jinho D.},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2026},
  note={arXiv:2510.23845}
}
```

---

## Summary table (therapy-eval angle)

| # | Paper | Training-time safety arm? | Guardrail-filter arm compared against training-time? | ASR-style attack metric? | Over-refusal metric? | Helpfulness metric? | Same/adjacent/different |
|---|---|---|---|---|---|---|---|
| 28 | PsychoCounsel-Preference (Zhang et al., ACL'26) | No (data/judge source) | N/A | No | No | N/A (is the judge) | different (provenance) |
| 29 | Between Help and Harm (Arnaiz-Rodriguez et al., JMIR MH) | No | No | No (Likert appropriateness) | No | No | adjacent |
| 30 | Automated Clinical AI Red Teaming (Steenstra et al., 2026) | No | No | No (simulated-session risk grading) | No | No | adjacent |
| 31 | MHSafeEval (Lee et al., 2026) | No | No | **Yes** | No | No | closest neighbour, not same |
| 32 | Multi-Objective Alignment / MODPO (Beikzadeh et al., 2026) | **Yes, DPO** | No | No | No | Preference-based only | overlaps method, not comparison |
| 33 | AI Safety Training Can be Clinically Harmful (Suhas BN et al., 2026) | No (evaluates RLHF-trained deployed models) | No | No (protocol-fidelity instead) | No | No | adjacent (Discussion citation) |
| 34 | PsychEthicsBench (Shen et al., 2026) | No | No | No | No (refusal-rate finding only) | No | adjacent |
| 35 | MindGuard (Farinhas et al., 2026) | No | Guardrail vs. guardrail only | **Yes** (guardrail-vs-guardrail) | No | No | neighbours B3 arm |
| 36 | PAIR-SAFE (Kim et al., 2026) | No | Runtime audit, not compared to training-time | No | No | No | neighbours B3 arm |
| 37 | DialogGuard (Luo & Laban, 2025) | No | N/A (judging tool) | No | No | No | neighbours (shared data) |
| 38 | CRADLE Bench (Byun et al., EACL'26) | No (detection classifiers) | No | No (detection task) | No | No | adjacent (different task) |

**No row in this table is "same."** Row 31 (MHSafeEval) is the closest on the
attack/evaluation side; row 32 (Beikzadeh et al. MODPO) is the closest on the DPO-for-
therapy-safety side. No single paper combines a trained safety arm, a bolted-on guardrail
arm, and an ASR + over-refusal + helpfulness triad in the therapy/mental-health domain.

## Open gaps / follow-ups (therapy-eval angle)

- MindGuard's possible npj Digital Medicine venue is unconfirmed (auth-walled redirect not
  followed) — verify before citing a specific journal venue in the paper; cite as arXiv
  only until confirmed.
- MHSafeEval (entry 31) is extremely recent (Apr 2026) and structurally the closest
  neighbour found across all three search passes on the evaluation-methodology side;
  recommend a citation-tracking check close to submission in case a revision or follow-up
  paper from the same group adds a training-arm or guardrail-arm comparison.
- The LLUMI / PAIR-SAFE research group (Kim, Yoo, Chandrasekharan, Saha, +co-authors) has
  published on both a DPO-with-safety-dimension paper (LLUMI, entry #13) and a separate
  runtime-guardrail paper (PAIR-SAFE, entry #36) without combining them — worth a
  citation-tracking check for a possible future paper from this group that does what we are
  doing.
- The unverified "59.7%→3.0%" lead (see note above) should be independently re-checked by
  whoever owns the general (non-therapy) DPO-vs-guardrail search angle, since it did not
  resolve under this angle but the underlying claim, if real, would be highly relevant to
  the project regardless of domain.
- VERA-MH's exact concept-paper arXiv ID remains unsettled across sessions (2605.13318 vs.
  2510.15297) — see the correction embedded in entry #26 in the attacks-shallow section
  above; resolve with one more direct check before the bibliography is finalised.

## Combined novelty verdict across all three search passes (safety-dpo, attacks-shallow,
therapy-eval) — evidence summary only, not a decision

Across 38 papers checked (three independent search sessions, largely non-overlapping
angles, with cross-referencing where they did overlap), **no paper was found that trains a
DPO model with safety preference pairs mixed in and compares its attack-success-rate against
an otherwise-identical model wrapped in a post-hoc/bolted-on guardrail filter, with
over-refusal and helpfulness reported as bounded costs — in the therapy/mental-health domain
or in any other domain covered by these searches.** The nearest single papers, for different
reasons, are: Garcia-Gasulla et al. "Egida" (method + metric triad match, general domain, no
filter baseline), Ackerman & Panickssery "Mitigating Many-Shot Jailbreaking" (training-vs-
inference-time comparison shape match, but SFT-vs-tag-sanitization not DPO-vs-classifier-
filter, general domain, single attack category), Lee et al. "MHSafeEval" (therapy-domain
adversarial ASR evaluation, no training arm), and Beikzadeh et al. "Multi-Objective
Alignment" (therapy-domain DPO-with-safety-dimension, no adversarial suite, no filter
baseline). This is evidence only; the main thread makes the novelty call.

---

## Search: DPO degeneration angle (2026-08-01, lit-scout)

**Trigger:** B2 v3 (beta=0.3) measures 14.3% strict / 23.7% loose repetition on the frozen
300-item suite and is REJECTED; B2 v2 (beta=0.1) was rejected earlier on an 8-prompt
qualitative check (4/8 degenerate, plus a distinct "fabricates its own User: turns"
pathology absent at beta 0.3); B1 (SFT-only) baseline is 3.0% strict / 6.0% loose. This
search is **not** about the safety-DPO-vs-guardrail novelty claim — it is infrastructure
literature to fix a training pathology before B2/T can produce a valid paper number. No
"escalate loudly" trigger applies here (this angle has no comparison-arm novelty question);
findings below are reported as engineering-relevant evidence for the main thread's decision
on beta/loss-type/data-filtering choices, not as a novelty verdict.

**Headline finding, stated up front:** the literature converges on **three independent,
partially-overlapping explanations** for what is being observed, and — critically for
schedule — **all of the standard mitigations found are single-flag or single-field changes
in TRL's `DPOConfig`, confirmed against the currently-published TRL `DPOTrainer`/`DPOConfig`
docs** (see the TRL-API note at the end of this section; **the project's exact installed
TRL 1.9.0 was not independently re-verified against this list — read the installed source
before committing to a flag, per this project's own standing practice**, e.g. the DPO-config
note in `lab_notebook.md` 2026-07-31 that explicitly reads installed TRL source rather than
assuming behaviour). None of the mitigations below require writing a new loss function,
a new trainer subclass, or a new data pipeline beyond straightforward filtering — this
matters given the ~2.5h/run cost and the ~3-week runway to results lock.

**The three explanations, and how they connect to the beta=0.1-vs-0.3 asymmetry observed:**

1. **Reward/likelihood over-optimisation that is *not* simply "further from reference =
   worse."** Rafailov et al. ("Scaling Laws for Reward Model Overoptimization in Direct
   Alignment Algorithms," arXiv:2406.02900, 2024) show DPO/IPO/SLiC all degrade with
   training progress and that degradation is driven by KL budget (β) in a way that is
   **non-monotonic and can occur within a single epoch** (peak quality at ~25% of one
   epoch's data in their tighter-KL runs, degrading thereafter even while reward-model
   score keeps rising) — i.e. more training steps at a *fixed* β is itself already enough
   to walk into a worse regime, independent of which β was chosen. This directly bears on
   the project's single-epoch, fixed-step-count design: it is not guaranteed that "fewer
   steps would look like beta 0.1" — early-training and late-training checkpoints at the
   *same* β can already differ qualitatively, so if degeneration is checked only at the
   end of the 1-epoch run, an earlier checkpoint might already be much better without
   changing β at all. **Concrete, checkable, cheap experiment implied by this paper: log
   the loose/strict repeat rate against a validation subset at intermediate checkpoints
   within the existing runs, not just at the end — no retrain required if checkpoints are
   already being saved.**

2. **Likelihood displacement — chosen-response likelihood can fall even while the
   preference margin (reward gap) rises**, which is consistent with the log-ratio numbers
   already recorded (reward margin *rose* from beta 0.1→0.3 — 8.3→13.0 — while the
   underlying log-ratio *shrank*, 83→43; the reward is beta × log-ratio, so a bigger
   apparent "margin" at higher beta is compatible with a *smaller*, not larger, absolute
   movement of the policy, yet degeneration got *worse*, not better, at beta 0.3 — this is
   the specific asymmetry the literature best explains). Razin, Malladi, Bhaskar, Chen,
   Arora, Hashimoto ("Unintentional Unalignment: Likelihood Displacement in Direct
   Preference Optimization," arXiv:2410.08847, **ICLR 2025**, peer-reviewed) prove
   displacement is driven by how similar the chosen/rejected completions are in embedding
   space (their CHES — centered hidden embedding similarity — score), and demonstrate a
   real safety consequence: DPO-aligning Llama-3-8B-Instruct to refuse unsafe prompts
   **reduced its refusal rate from 74.4% to 33.4%** due to displacement redirecting
   probability mass from the "No" refusal token toward an unintended "Yes" continuation.
   Their fix is **data-side, not loss-side**: filter out training pairs whose chosen/
   rejected completions are too embedding-similar (compute CHES, drop high-CHES pairs).
   This is directly actionable on `pref_helpful.jsonl`/`pref_safety.jsonl` before the next
   training run, and does not touch `loss_type` or `beta` at all.

   **CORRECTION (2026-08-01, second lit-scout pass, cross-checked directly against the
   arXiv abstract page rather than a prior summary): the paper's sixth/last author is
   Boris Hanin, not "Hashimoto, Tatsunori" as recorded above and in the bibtex entry
   `razin2025likelihooddisplacement` below.** Full confirmed author order: Noam Razin,
   Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, Boris Hanin. The bibtex
   entry in this file's own bibliography section is therefore **wrong and must be fixed**
   before this paper is cited in the dissertation — flagged here explicitly rather than
   silently edited, since this file is a shared, appended log and the error is already
   duplicated into a citable BibTeX block below.

3. **A specific, mechanistic account of *why* DPO becomes "unlikelihood training" (pure
   repetition-inducing suppression of the rejected sequence) under some conditions**: the
   DPO gradient is a *weighted* contrastive update, where the weight is
   `σ(β·(margin))`'s complement — examples the model already gets confidently right
   contribute almost no gradient, so a model that has become confident (larger β, or later
   in training) effectively spends its remaining gradient budget almost entirely on the
   *hardest*, least-separated pairs, which is exactly where Razin et al.'s displacement
   risk is concentrated. Feng, Qin, Huang, Zhang, Lei ("Towards Analyzing and Understanding
   the Limitations of DPO: A Theoretical Perspective," arXiv:2404.04626, 2024, preprint —
   **venue not confirmed as peer-reviewed**, appears widely cited but only checked at
   arXiv here) independently derive, via a gradient-field analysis, that **DPO's loss
   decreases the probability of the *dispreferred* response faster than it increases the
   probability of the *preferred* one** — a structural asymmetry that, combined with (2),
   is consistent with a model that suppresses fluent continuation probability broadly
   enough to fall into repetition loops rather than cleanly separating chosen from
   rejected. A community technical report (not peer-reviewed — flagged explicitly, see
   caveat below) states this mechanism even more bluntly: "without the sigmoid weighting,
   DPO degrades to unlikelihood training, which causes repetitive, degenerate text" —
   cited here only as a plain-language gloss of the same mechanism the peer-reviewed
   papers above derive formally, not as an independent source of authority.

**Root cause specific to this project's setup (edit distance / near-duplicate pairs),
independently corroborated by a second, directly-actionable paper:** Pal, Karkhanis,
Dooley, Roberts, Naidu, White ("Smaug: Fixing Failure Modes of Preference Optimisation with
DPO-Positive," arXiv:2402.13228, 2024, preprint) show theoretically and empirically that
**standard DPO can reduce the model's likelihood of the *preferred* completion whenever the
relative probability between preferred and dispreferred still increases** — and that this
failure mode is worst specifically for **low-edit-distance** chosen/rejected pairs (i.e.
pairs that differ by only a few tokens/words). This is worth flagging against the
`pref_safety.jsonl` provenance already logged in `lab_notebook.md` (PKU-SafeRLHF pairs,
"chosen" = only the *safer of the pair*, not a curated ideal) — PKU-SafeRLHF pairs are
not guaranteed to be high-edit-distance, and a spot audit of edit distance in the safety
pairs (and the helpfulness pairs) would tell you whether this specific failure mode is live
in this data before assuming it's purely a beta/hyperparameter issue. Their fix, **DPOP**,
adds a corrective penalty term to the loss (a genuinely new loss, not a flag) — **this is
the one mitigation on this list that is not a simple TRL config change**; TRL's built-in
loss-type list (below) does not include a `dpop` option as of the fetched docs, so using it
would require a custom loss function, which conflicts with the "identical across B2/T/T_ctrl,
no per-arm tuning" hard constraint only in the sense that it is more engineering risk to
verify identically across three arms under schedule pressure — flagged as higher-cost than
the alternatives below, not ruled out.

**Length/verbosity-specific literature (secondary relevance — the observed pathology is
repetition-loop degeneration, not simple verbosity, but the same beta/KL mechanism is
implicated and the TRL flags overlap):** Rafailov et al. (arXiv:2406.02900, above) find
DAAs (DPO, IPO, SLiC) exhibit length exploitation as one concrete symptom of
over-optimisation. Park, Rafailov, Ermon, Finn ("Disentangling Length from Quality in
Direct Preference Optimization," ACL Findings 2024, peer-reviewed, arXiv:2403.19159)
independently confirm significant length exploitation in DPO specifically and propose a
length-regularised objective (R-DPO) with up to 20% win-rate improvement when controlling
for length — **this requires a modified loss term**, not currently a named `loss_type` in
the TRL docs fetched (TRL's closest built-ins are `ld_alpha`, `use_weighting`, and
`sigmoid_norm`, described below, which are inspired by adjacent but not identical papers).
Lu, Li et al. ("Eliminating Biased Length Reliance of Direct Preference Optimization via
Down-Sampled KL Divergence," EMNLP 2024, peer-reviewed, arXiv:2406.10957) attribute length
bias to a **sequence-level KL discrepancy that scales with token count** and propose
token-level down-sampling (SamPO); reported gains of 5–12% over DPO on length-debiased
reward. Meng, Xia, Chen ("SimPO: Simple Preference Optimization with a Reference-Free
Reward," NeurIPS 2024, peer-reviewed) use **length-normalised** log-probability as the
reward itself (no reference model at all), reporting up to 6.4 pts AlpacaEval-2 / 7.5 pts
Arena-Hard over DPO; TRL exposes this directly as `loss_type="sigmoid_norm"` (confirmed in
the docs fetch below — "the SimPO authors address the length-bias in the original sigmoid
loss by normalizing by the number of non-mask tokens"). Gu et al. ("Length Desensitization
in Direct Preference Optimization" / LD-DPO, arXiv:2409.06411, 2024, preprint) propose a
token-weighting scheme for the "verbose" tail of a response, reporting 10–40% shorter
responses than DPO at matched quality; TRL exposes this directly as the `ld_alpha` float
parameter (confirmed in the docs fetch below).

**Data-filtering mitigations, orthogonal to the loss function (no loss change at all —
config/data-pipeline only):** Morimura, Sakamoto, Jinnai, Abe, Ariu ("Filtered Direct
Preference Optimization," EMNLP 2024, peer-reviewed, arXiv:2404.13846) show DPO is *more*
sensitive to noisy/low-quality preference pairs than reward-model-based RLHF, and propose
fDPO: use a trained reward model to monitor and drop low-quality pairs from the training
set during DPO, reporting improved final performance (no exact percentage independently
extracted here — flagged as unverified beyond the qualitative direction). This requires an
auxiliary reward model already present in the project (`PsychoCounsel-Llama3-8B-Reward`, or
equivalently `beaver-dam-7b` for the safety side) and a filtering pass on
`pref_helpful.jsonl`/`pref_safety.jsonl`, not a new loss.

**Noise/label-smoothing mitigations, a straight TRL config flag:** Mitchell's original DPO
codebase note ("A note on DPO with noisy preferences & relationship to IPO," Eric Mitchell,
2023, **a technical note/blog-style PDF, not a peer-reviewed paper** — cited as such,
`ericmitchell.ai/cdpo.pdf`) introduces **conservative DPO (cDPO)**: assume a small fraction
ε of preference labels are flipped, which mathematically reduces to adding
`label_smoothing=ε` to the standard sigmoid DPO loss (ε=0 recovers vanilla DPO). Chowdhury,
Kini, Natarajan ("Provably Robust DPO: Aligning Language Models with Noisy Feedback,"
arXiv:2403.00409, 2024, preprint — venue not confirmed) independently derive a related,
provably-unbiased loss under random label noise, exposed in TRL as `loss_type="robust"`
with the *same* `label_smoothing` field reinterpreted as the label-flip probability
(recommended ≈0.1 per the TRL docs' citation of the paper). **Both are one-line config
changes** (`label_smoothing=X` with `loss_type="sigmoid"` for cDPO, or `loss_type="robust"`
for the provably-robust variant), require no data pipeline change, and — because they only
add a smoothing/robustness term rather than changing which examples are used — are the
cheapest thing on this list to apply **identically across B2/T/T_ctrl** without risking a
per-arm confound, which matters given the project's explicit "no per-arm tuning" constraint.

**IPO — a fundamentally different loss, not a flag on top of sigmoid, but natively
supported:** Azar et al. ("A General Theoretical Paradigm to Understand Learning from Human
Preferences," arXiv:2310.12036, 2023/AISTATS 2024, peer-reviewed) show DPO's sigmoid/
logistic loss can overfit and drive the reward margin unboundedly large when preferences
are close to deterministic (exactly the "margins ~8.3 → ~13.0" pattern being observed),
because the underlying Bradley-Terry reward is undefined/ill-posed once labels are
near-certain and the model exploits this rather than respecting the reference-model KL
term. **IPO replaces the sigmoid/log-loss with a bounded squared-error-style objective**
that targets a fixed margin (1/2) rather than trying to push the margin to infinity, which
directly targets "the model keeps moving unboundedly far from the reference despite a
nominal beta" — the mechanism underlying this project's headline question of why the KL
term doesn't obviously bound this. TRL exposes this as `loss_type="ipo"` (confirmed below;
`beta` is reinterpreted as the IPO regularisation parameter τ when this loss is selected) —
**a one-line `loss_type` change, no data pipeline change, no custom loss code.**

**RPO-style DPO+NLL hybrid — available in current TRL via multi-loss combination, but the
parameter name has changed and needs live verification against the installed 1.9.0:** Pang,
Yuan, Cho, He, Sukhbaatar, Weston ("Iterative Reasoning Preference Optimization,"
NeurIPS 2024, peer-reviewed, arXiv:2404.19733) show that adding a **negative log-likelihood
(NLL) term on the chosen completion**, on top of the standard DPO loss, is "crucial" — pure
DPO alone underperforms, and the NLL term anchors the chosen-response likelihood so it
cannot fall the way likelihood displacement (see above) predicts. **This is the mechanism
most directly aimed at "chosen-response likelihood falling during DPO."** In the TRL
`DPOTrainer` docs fetched during this search (see API note below), this is available via
TRL's **multi-loss combination** mechanism — `loss_type=["sigmoid", "sft"]` with
`loss_weights=[1.0, α]` — rather than a single `rpo_alpha` scalar; **older TRL
versions/tutorials may reference an `rpo_alpha` field directly, so this must be checked
against the actual installed 1.9.0 signature before use, per this project's standing rule
of reading installed source rather than copying blog-post-era arguments.**

**TRL `DPOConfig` API note (fetched from the current published Hugging Face TRL docs
during this search; NOT independently re-verified against the project's pinned TRL
1.9.0 — flagged explicitly as a to-do before any config is written):**
- `beta` (float, default 0.1): "Higher β means less deviation from the reference model."
  For `loss_type="ipo"`, `beta` is reinterpreted as IPO's τ.
- `loss_type` (list[str], default `["sigmoid"]`): confirmed available values include
  `sigmoid`, `hinge`, `ipo`, `exo_pair`, `nca_pair`, `robust`, `bco_pair`, `sppo_hard`,
  `aot`/`aot_unpaired`, `apo_zero`/`apo_down`, `discopop`, `sft`, `sigmoid_norm`. Multiple
  entries can be combined with `loss_weights` (documented MPO-style use: e.g.
  `loss_type=["sigmoid","bco_pair","sft"], loss_weights=[0.8,0.2,1.0]`).
- `label_smoothing` (float, default 0.0): cDPO-style noise-robustness under `sigmoid`
  (recommended ~0.1 per Robust DPO) or EXO-style label smoothing under `exo_pair`
  (recommended ~1e-3).
- `ld_alpha` (float, optional): LD-DPO-style down-weighting of "verbose" token log-probs
  beyond the shared chosen/rejected length; `1.0` = no weighting (vanilla DPO), `0.0` =
  masks tokens beyond the shared length entirely.
- `use_weighting` (bool, default False): WPO-style (Zhou et al., arXiv:2406.11827)
  reweighting of pairs by the policy's own length-normalised sequence probability, aimed at
  the off-policy/on-policy distributional gap rather than degeneration directly.
- `f_divergence_type` (default `reverse_kl`; also `forward_kl`, `js_divergence`,
  `alpha_divergence`): generalises the KL penalty itself for a subset of loss types
  (`sigmoid`, `sigmoid_norm`, `hinge`, `ipo`, `exo_pair`, `robust`, `discopop`, `sft`) —
  worth noting since the project's core question ("why does DPO drift far from reference
  despite a KL term") is partly a statement about *which* divergence is being enforced.
- No `dpop`/DPO-Positive loss type was found in the fetched docs — DPOP would require
  custom loss code, not a flag.
- **Additional field confirmed by a second, independent docs fetch (2026-08-01): `beta`
  defaults to 0.1 and `learning_rate` defaults to `1e-6` — explicitly documented as
  different from `TrainingArguments`'s general 5e-5 default, "because DPO is sensitive to
  hyperparameters." The docs further state: "when training adapters, you typically use a
  higher learning rate (≈1e-5) than full fine-tuning since only new parameters are being
  learned," i.e. TRL's own published guidance is that a LoRA/PEFT run should use a
  *higher*, not lower, learning rate than a full-fine-tuning run — this project's 5e-6 is
  roughly 5x below that guidance figure, and roughly 5x above the library's full-FT-style
  default. Also confirmed: when `peft_config` is supplied and `ref_model=None`, "the
  trainer will automatically use the initial policy corresponding to `model`, i.e. the
  model state before DPO training starts" as the reference policy — i.e. this project's
  B1-adapter-as-its-own-frozen-reference design is exactly TRL's documented standard PEFT/
  DPO code path (matching Rafailov et al. 2023's original SFT-checkpoint-as-reference
  design), not a custom or unusual configuration. This directly answers one of the
  project's original questions: the *reference-model choice* is not an unusual source of
  risk here; if anything is unusual it is the narrowness of the specific SFT distribution
  being used as that reference (see the LoRA-capacity and mode-collapse notes below), not
  the mechanism of reference selection itself.**
- **Caveat, stated plainly:** this list was fetched from what appears to be the *current*
  published TRL docs during this search session, which may be ahead of or behind the
  project's pinned TRL 1.9.0 (CLAUDE.md already warns that TRL is a fast-moving dependency
  and tutorials/blog posts get out of sync with the installed version). Before writing any
  config, `train-runner` should read the installed `trl/trainer/dpo_config.py` source
  directly to confirm every field name above (`loss_type`, `label_smoothing`, `ld_alpha`,
  `use_weighting`, `beta`, `learning_rate`, and whether `rpo_alpha` exists as a separate
  field or only via the `loss_type=["sigmoid","sft"]` combination) rather than trusting
  this fetch. This is the same discipline already applied elsewhere in this project (e.g.
  the `is_trainable=True` finding and the `get_training_chat_template` fallback finding
  logged in `lab_notebook.md`), extended here because a wrong assumption about the
  loss/config signature is exactly the kind of silent-failure mode CLAUDE.md warns about.

**Direct answer to "why can DPO drive the policy far from reference despite the KL term,"
synthesising the above:** the nominal `beta`-weighted KL term in the DPO loss is a penalty
computed only on the two observed completions (chosen, rejected) per example, not a
constraint over the full output distribution — so gradient steps can push probability mass
onto sequences *never scored by the loss at all* (Razin et al.'s displacement finding, and
Rafailov et al.'s "out-of-distribution extrapolation" framing, both point at this same gap).
Practically, the "beta constrains deviation" intuition holds in an aggregate/expectation
sense (confirmed by this project's own reward-margin/log-ratio numbers: log-ratio shrank
83→43 going from beta 0.1→0.3, i.e. beta 0.3 *did* keep the model closer to the reference on
the *scored* tokens) but does not hold pointwise for whichever tokens the loss never directly
constrains — which is exactly where a repetition loop lives. This is why raising beta
(nominally "more conservative") did not fix and in this project's own data made worse the
degeneration: it shrinks movement on the *scored* chosen/rejected tokens while the failure
mode is concentrated in the *unscored*, freely-drifting continuation space. This synthesis
is this agent's own connecting of the papers above to the project's specific beta 0.1-vs-0.3
numbers — it is not a claim any single cited paper makes in these exact terms, and should be
labelled as such (an inference, not a direct citation) if used in the paper's Discussion.

**A second, independent mechanistic account for the same directional asymmetry (added
2026-08-01, supplementary pass — see "Supplementary note" below for the fuller writeup):**
Sahoo, Chadha, Jain, Chaudhary ("Pessimism's Paradox: Conservative Offline Training
Amplifies Reward Hacking During Online Adaptation in Reasoning Models," ICML 2026 workshop,
arXiv:2606.30627) find, in a different setting (Qwen3-14B, offline DPO followed by online
reward-model adaptation on GSM8K, not our offline-DPO-then-static-eval setting), that
**higher β compresses policy entropy into a narrower output manifold**, and that this
narrower manifold is *more* exploitable/fragile despite responses lying nominally closer to
a reference distribution. This is offered as converging, though not directly transferable,
support for the same-direction finding already synthesised above (entropy/diversity
compression, not aggregate KL distance, is the operative axis) — see the supplementary
section for full citation details and explicit caveats about domain mismatch.

### Bibliography for this section

```bibtex
@article{rafailov2024scalinglaws,
  title={Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms},
  author={Rafailov, Rafael and Chittepu, Yaswanth and Park, Ryan and Sikchi, Harshit and Hejna, Joey and Knox, Bradley and Finn, Chelsea and Niekum, Scott},
  journal={arXiv preprint arXiv:2406.02900},
  year={2024}
}
@inproceedings{razin2025likelihooddisplacement,
  title={Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization},
  author={Razin, Noam and Malladi, Sadhika and Bhaskar, Adithya and Chen, Danqi and Arora, Sanjeev and Hanin, Boris},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025},
  note={CORRECTED 2026-08-01: sixth author is Boris Hanin, not Tatsunori Hashimoto as originally entered here — verified directly against the arXiv abstract page.}
}
@article{feng2024limitations,
  title={Towards Analyzing and Understanding the Limitations of {DPO}: A Theoretical Perspective},
  author={Feng, Duanyu and Qin, Bowen and Huang, Chen and Zhang, Zheng and Lei, Wenqiang},
  journal={arXiv preprint arXiv:2404.04626},
  year={2024}
}
@article{pal2024smaug,
  title={Smaug: Fixing Failure Modes of Preference Optimisation with {DPO}-Positive},
  author={Pal, Arka and Karkhanis, Deep and Dooley, Samuel and Roberts, Manley and Naidu, Siddartha and White, Colin},
  journal={arXiv preprint arXiv:2402.13228},
  year={2024}
}
@inproceedings{park2024disentangling,
  title={Disentangling Length from Quality in Direct Preference Optimization},
  author={Park, Ryan and Rafailov, Rafael and Ermon, Stefano and Finn, Chelsea},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  year={2024},
  note={arXiv:2403.19159}
}
@inproceedings{lu2024sampo,
  title={Eliminating Biased Length Reliance of Direct Preference Optimization via Down-Sampled {KL} Divergence},
  author={Lu, Junru and Li, Jiazheng and An, Siyu and Zhao, Meng and He, Yulan and Yin, Di and Sun, Xing},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024},
  note={arXiv:2406.10957}
}
@inproceedings{meng2024simpo,
  title={{SimPO}: Simple Preference Optimization with a Reference-Free Reward},
  author={Meng, Yu and Xia, Mengzhou and Chen, Danqi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={37},
  year={2024}
}
@article{gu2024lddpo,
  title={Length Desensitization in Direct Preference Optimization},
  author={Gu, Wenliang and others},
  journal={arXiv preprint arXiv:2409.06411},
  year={2024},
  note={author list incomplete beyond first author — verify before citing}
}
@inproceedings{morimura2024fdpo,
  title={Filtered Direct Preference Optimization},
  author={Morimura, Tetsuro and Sakamoto, Mitsuki and Jinnai, Yuu and Abe, Kenshi and Ariu, Kaito},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024},
  note={arXiv:2404.13846}
}
@misc{mitchell2023cdpo,
  title={A Note on {DPO} with Noisy Preferences \& Relationship to {IPO}},
  author={Mitchell, Eric},
  year={2023},
  note={Technical note, not peer-reviewed. \url{https://ericmitchell.ai/cdpo.pdf}}
}
@article{chowdhury2024robustdpo,
  title={Provably Robust {DPO}: Aligning Language Models with Noisy Feedback},
  author={Chowdhury, Sayak Ray and Kini, Anush and Natarajan, Nagarajan},
  journal={arXiv preprint arXiv:2403.00409},
  year={2024}
}
@inproceedings{azar2024ipo,
  title={A General Theoretical Paradigm to Understand Learning from Human Preferences},
  author={Azar, Mohammad Gheshlaghi and Rowland, Mark and Piot, Bilal and Guo, Daniel and Calandriello, Daniele and Valko, Michal and Munos, R{\'e}mi},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2024},
  note={arXiv:2310.12036}
}
@inproceedings{pang2024irpo,
  title={Iterative Reasoning Preference Optimization},
  author={Pang, Richard Yuanzhe and Yuan, Weizhe and Cho, Kyunghyun and He, He and Sukhbaatar, Sainbayar and Weston, Jason},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={37},
  year={2024},
  note={arXiv:2404.19733}
}
```

### Open gaps / follow-ups (DPO degeneration angle)

- **Not independently verified: whether TRL 1.9.0 as installed on `CNXLAB03` actually has
  `loss_type="sft"` multi-loss combination, `ld_alpha`, `use_weighting`, `loss_type="ipo"`,
  and `loss_type="robust"`/`label_smoothing` exactly as documented above.** The fetch used
  for the API note was against the currently-published TRL docs site, not the pinned
  package. `train-runner` (or whoever writes the next DPO config) must read
  `trl/trainer/dpo_config.py` and `trl/trainer/dpo_trainer.py` from the actual installed
  environment before relying on any specific flag name.
- **The edit-distance/CHES-similarity hypothesis (Razin et al., Pal et al.) was not
  measured against the project's own `pref_safety.jsonl`/`pref_helpful.jsonl`.** This is a
  cheap, concrete, falsifiable check (compute embedding similarity or even simple edit
  distance between chosen/rejected per pair, compare degenerate-arm vs non-degenerate-arm
  distributions) that this search recommends but did not perform — it is a data-analysis
  task, not a literature-search task, and belongs with `data-wrangler` or `eval-harness`.
- **The "unlikelihood training" gloss** (from the non-peer-reviewed HuggingFace community
  blog post, "Text Degeneration: A Production Failure Mode That Most Benchmarks Do Not
  Track," Dharma-AI, no formal authorship/venue found) is cited only as a plain-language
  restatement of the peer-reviewed Feng et al. gradient-asymmetry finding, not as an
  independent source — do not cite the blog post's own "59.4% average reduction" figure in
  the paper, since that number comes from an unrelated OCR-degeneration study (Nanonets-
  OCR2 / Qwen2.5-VL, 3B–7B), not from a safety-preference or therapy-domain DPO setting, and
  is not applicable to this project's numbers.
- **The connecting synthesis paragraph above (why beta 0.3 is worse than beta 0.1 in this
  project's own repetition metrics) is this agent's own inference from the cited papers,
  not a statement made by any single cited source** — flagged explicitly per the instruction
  not to invent authority; the main thread should treat it as a hypothesis to test (e.g. via
  the intermediate-checkpoint check under finding 1, or the edit-distance check above), not
  as an established fact suitable for a bare citation.
- Did not find a peer-reviewed paper reporting DPO-induced *repetition-loop* degeneration
  (as opposed to verbosity/length exploitation, or likelihood displacement toward an
  unintended coherent alternative) as its primary, named phenomenon in a therapy or
  general-instruction-tuning setting with quantitative strict/loose repeat-rate figures
  comparable to this project's own metric. The closest are the mechanistic accounts (Feng
  et al., Razin et al., Rafailov et al.) which explain *why* it could happen, and the
  Dharma-AI blog post (not peer-reviewed, different domain) which reports a comparable
  *style* of metric. This is a gap worth stating explicitly in Methods if the paper claims
  the repetition-loop pathology itself as a novel empirical observation.

---

## Supplementary note: LoRA-DPO hyperparameter comparators and LoRA-rank/capacity angle
(2026-08-01, second lit-scout pass — same trigger as the section immediately above; kept
separate because it approaches the question from published-hyperparameter-comparison and
LoRA-capacity angles rather than loss-mechanism angles, and largely does not duplicate the
section above). Answers the specific questions: what learning rates/betas are actually used
in comparable published 7B LoRA DPO setups; whether LoRA rank interacts with preference-
optimisation stability; and whether using an already-narrow SFT adapter as its own DPO
reference is itself unusual.

**On reference-model choice (short answer: not unusual).** Rafailov et al.'s original DPO
paper (NeurIPS 2023, arXiv:2305.18290) uses the SFT checkpoint as the reference policy in
every experiment — this is the field-standard design this project's B1-as-reference setup
follows, not a deviation. TRL's own documentation, cross-checked directly against the live
docs during this pass, confirms that when `ref_model=None` and a `peft_config` is supplied,
"the trainer will automatically use the initial policy corresponding to `model`, i.e. the
model state before DPO training starts" — i.e. B1's own frozen adapter as reference is
exactly TRL's documented default PEFT/DPO code path. **What is unusual about this project's
setup is not the reference-model mechanism, but the narrowness/low-entropy of the specific
SFT distribution (B1) being used as that reference** — B1 is deliberately terse (median
response 97–110 chars vs base Qwen's ~1,100–1,165), a faithful reproduction of ESConv's own
short-turn register per `lab_notebook.md`'s 2026-07-31 brevity-verdict entry, not a defect.

**Published 7B-scale DPO hyperparameters found, for direct comparison against this
project's β∈{0.1, 0.3}, lr=5e-6, 1 epoch, LoRA r=32/α=64:**

| Recipe | Base model | Params updated | β | Learning rate | Epochs | Reference |
|---|---|---|---|---|---|---|
| **This project (B2 v2/v3)** | Qwen2.5-7B-Instruct, on B1 LoRA SFT | **LoRA r=32/α=64** | **0.1 / 0.3** | **5e-6** | **1** | B1's own frozen adapter |
| Zephyr-7B-beta (Tunstall et al., HuggingFace, arXiv:2310.16944) | Mistral-7B (SFT'd) | **Full fine-tuning** — paper states "we did not experiment with... LoRA, but expect similar results to hold" | 0.1 | 5e-7 | 3 (after 1 epoch dSFT) | SFT checkpoint |
| Egida-Qwen2.5-7B-Instruct-DPO (Garcia-Gasulla et al., 2025; HF model card) | Qwen2.5-7B-Instruct | **Full fine-tuning** | not published | ≈1e-7 | not published | Qwen2.5-7B-Instruct itself |
| TRL `DPOConfig` library default | any | either | 0.1 | 1e-6 (full-FT default; docs recommend ≈1e-5 for adapters — see above) | 3 (library default) | policy pre-DPO state if `ref_model=None` |
| Liu, Liu & Cohan, "Understanding Reference Policies in DPO" (NAACL 2025 Findings, arXiv:2407.13709) | mistral-7b, tulu-2-7b | **Full fine-tuning** | optimum found: **0.01 (mistral-7b) / 0.02 (tulu-2-7b)**; severe "model degradation" observed going *below* this, at β=0.005 | not fully extracted | 3 | SFT checkpoint |
| AdaDPO grid (Chen, Ciobanu, Mao, Das, arXiv:2605.28440, 2026 preprint) | Llama-3-8B-Instruct | **Full fine-tuning** | grid 0.005–0.1 | grid 3e-7–1e-6 | 1 | policy pre-DPO state |

**Reading the table:** every full-fine-tuning 7B DPO recipe with verifiable published
numbers uses β≤0.1 and lr≤2e-5 — one to two orders of magnitude below this project's 5e-6 in
several cases, though TRL's own LoRA-specific guidance (≈1e-5) sits *above* 5e-6, so this
project's learning rate is not clearly high or low in absolute terms, only in the sense that
**no comparator found here is both LoRA and 7B-scale simultaneously** — this remains the
single biggest gap in validating or invalidating the exact configuration from precedent
alone. The one paper that explicitly searched for an optimal β at 7B (Liu, Liu & Cohan)
found an optimum an order of magnitude below the field's common default of 0.1, with
degeneration appearing *below* that optimum (β=0.005) — the **opposite direction** from this
project's own finding that β=0.3 degenerates worse than β=0.1. Two readings, stated as
alternatives rather than a resolved conclusion: (a) if a low-β optimum generalises to this
project's setting, neither 0.1 nor 0.3 may be inside the good region, and a diagnostic run
at β≈0.02–0.05 is a concrete, literature-motivated next step distinct from choosing between
0.1 and 0.3; (b) that paper is full-fine-tuning only, so its optimum may not transfer to a
LoRA setup at all, given LoRA's lower-rank update space changes how much the policy can move
per gradient step at a given β/lr — this should be tested on this project's own machine, not
assumed either way.

**On LoRA rank / capacity interacting with preference-optimisation stability:** no paper
found directly tests LoRA-rank sensitivity for DPO specifically. The closest, and the reason
this is flagged as a real but unconfirmed hypothesis rather than a citation-backed fact:
Biderman et al., "LoRA Learns Less and Forgets Less" (TMLR, Aug 2024, Featured
Certification, peer-reviewed, arXiv:2405.09673) show, for SFT and continued pretraining
only (**no DPO tested**), that full fine-tuning learns weight perturbations with effective
rank 10–100× higher than typical LoRA configurations, and that LoRA "learns less" within
the target domain as a direct consequence of this capacity gap. This project's DPO run sits
inside a rank-32 update space layered on top of an already rank-32-constrained B1 adapter
(the LoRA update for B2/T is a further adaptation of an adaptation, not a fresh rank-32
budget against the base model) — a plausible contributing factor to why a nominally
standard β/lr combination behaves differently here than in the full-fine-tuning comparators
above, but this is this project's own inference connecting two separate literatures (LoRA
capacity constraints; DPO instability), not a claim either source paper makes about the
other. State as a hypothesis in Discussion/Limitations, not as an established mechanism.

**On why higher β made degeneration worse here specifically (converging evidence, different
domain — full citation, complementing the "Direct answer" synthesis in the DPO-degeneration
section above):** Sahoo, Chadha, Jain, Chaudhary, "Pessimism's Paradox: Conservative Offline
Training Amplifies Reward Hacking During Online Adaptation in Reasoning Models" (ICML 2026
workshop on Decision-Making from Offline Datasets to Online Adaptation, peer-reviewed
workshop paper, arXiv:2606.30627) find a three-link causal chain in their own setting
(Qwen3-14B policy, offline DPO at three β levels, then online adaptation against a
3×Qwen3-1.7B reward ensemble on GSM8K): (i) higher β compresses policy entropy into a
narrower output manifold; (ii) the resulting low-diversity responses cluster closer, in
embedding space, to the reward model's training distribution; (iii) despite this apparent
proximity, ensemble disagreement (epistemic uncertainty) about those responses *increases*
with β, and that uncertainty gap is what gets exploited fastest. **Explicitly flagged
caveats:** different domain (reasoning/math, not chat/counseling), different training
regime (online adaptation after offline DPO, not this project's simple offline-DPO-then-
static-eval), model family adjacent but not identical (Qwen3, not Qwen2.5), and whether
LoRA or full fine-tuning was used was not confirmed in the material fetched. Offered as a
second, independent mechanistic account converging on the same direction (entropy/diversity
compression as the operative axis, not aggregate KL/log-ratio distance) as the synthesis
already written in the DPO-degeneration section above — not as a proven transfer to this
project's setting. A concrete, cheap diagnostic this suggests: measure per-token output
entropy (not just the pass/fail strict/loose repeat-rate detectors) on B2 v2 vs v3
generations, since a repetition loop is definitionally a very-low-entropy output mode and
this would test the mechanism directly rather than only its symptom.

**Weak practitioner-level corroboration that this symptom (DPO-induced token-repetition,
worsening with training, absent from the pre-DPO checkpoint) has been independently
reported before, flagged as anecdotal, not citable as an academic source:** GitHub issue
`huggingface/trl#1025`, "DPO models generate multiple / corrupted responses" (opened Nov
2023, no maintainer diagnosis or fix visible in the fetched thread). Reported setup: a
T5-family encoder-decoder model (**architecture mismatch — not a decoder-only 7B chat
model**), LoRA r=8/α=16/dropout=0.05, β=0.1, learning rate **5e-4** (two orders of magnitude
above this project's 5e-6, itself a plausible independent cause in that report), on a
trivial synthetic 4-class classification task. Symptom: greedy generation degenerates into
repeated single tokens ("a a a a a a") and, with more training, corrupted token
concatenations ("aaacat"); the same base model with plain supervised loss (no DPO) generates
correctly, isolating DPO training as the point of introduction — the same isolation this
project's own B1-vs-B2 comparison already demonstrates. Cite, if at all, only as evidence
that "DPO-induced token-repetition degeneration is a previously-reported failure signature
in the TRL ecosystem," not as evidence bearing on root cause or fix, given the architecture,
task, and learning-rate mismatches. Reference by URL only, not BibTeX:
`https://github.com/huggingface/trl/issues/1025`.

```bibtex
@inproceedings{rafailov2023dpo,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D. and Finn, Chelsea},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  year={2023}
}
@inproceedings{liu2025understanding,
  title={Understanding Reference Policies in Direct Preference Optimization},
  author={Liu, Yixin and Liu, Pengfei and Cohan, Arman},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  year={2025},
  note={arXiv:2407.13709}
}
@article{biderman2024lora,
  title={LoRA Learns Less and Forgets Less},
  author={Biderman, Dan and Portes, Jacob and Gonzalez Ortiz, Jose Javier and Paul, Mansheej and Greengard, Philip and Jennings, Connor and King, Daniel and Havens, Sam and Chiley, Vitaliy and Frankle, Jonathan and Blakeney, Cody and Cunningham, John P.},
  journal={Transactions on Machine Learning Research},
  year={2024},
  note={Featured Certification; arXiv:2405.09673}
}
@inproceedings{sahoo2026pessimisms,
  title={Pessimism's Paradox: Conservative Offline Training Amplifies Reward Hacking During Online Adaptation in Reasoning Models},
  author={Sahoo, Subramanyam and Chadha, Aman and Jain, Vinija and Chaudhary, Divya},
  booktitle={ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning},
  year={2026},
  note={arXiv:2606.30627}
}
@misc{chen2026adadpo,
  title={{AdaDPO}: Self-Adaptive Direct Preference Optimization with Balanced Gradient Updates},
  author={Chen, Shaolong and Ciobanu, Madalina and Mao, Qingqing and Das, Ritankar},
  year={2026},
  note={arXiv:2605.28440}
}
@article{tunstall2023zephyr,
  title={Zephyr: Direct Distillation of {LM} Alignment},
  author={Tunstall, Lewis and Beeching, Edward and Lambert, Nathan and Rajani, Nazneen and Rasul, Kashif and Belkada, Younes and Huang, Shengyi and von Werra, Leandro and Fourrier, Cl{\'e}mentine and Habib, Nathan and Sarrazin, Nathan and Sanseviero, Omar and Rush, Alexander M. and Wolf, Thomas},
  journal={arXiv preprint arXiv:2310.16944},
  year={2023},
  note={Technical report; venue beyond arXiv not confirmed}
}
```

### Open gaps / follow-ups (supplementary note)

- **The hyperparameter comparator table above has no entry that is both LoRA and 7B-scale.**
  This is the single largest open gap for directly validating or invalidating this
  project's exact configuration from precedent; the recommendation is to treat the
  literature as bounding-but-not-settling the question and to run a small, cheap diagnostic
  on this project's own hardware (e.g. one short run at a literature-motivated lower β,
  such as 0.02–0.05, at the existing lr=5e-6, checked against the same strict/loose repeat
  detectors already built) rather than searching further for a nonexistent exact-match
  paper.
  - **Learning rate and β should not be varied in the same diagnostic run** if the goal is
  to isolate which lever (if either) fixes the degeneration — TRL's own LoRA-lr guidance
  (≈1e-5) and Liu/Liu/Cohan's low-β optimum (0.01–0.02) are two independent, potentially
  competing levers found in this search, and whichever combination is ultimately chosen
  must still be applied identically across B2/T/T_ctrl per the project's hard "no per-arm
  tuning" constraint.
- **The author-list correction to `razin2025likelihooddisplacement`** (Boris Hanin, not
  Tatsunori Hashimoto, as the sixth author) was made in this pass directly against the
  bibtex entry already present in this file from the earlier lit-scout pass — flagged
  loudly here rather than silently edited, since the erroneous version was already written
  into a citable BibTeX block once.
- The "LoRA rank interacts with DPO stability" hypothesis (Biderman et al. connection) is
  this agent's own inference, not a finding either cited paper makes about the other —
  restated here for emphasis since it is the kind of connective claim that is easy to
  mis-cite as if it were a direct finding.


---

## Search: novelty re-check + guardrail-successor + domain-transfer pass (2026-08-02, lit-scout, P-H)

**Trigger:** task P-H — full pass to (1) re-run the exact-comparison novelty check with fresh eyes, (2) fill citation gaps in the anchored-DPO/RPO/NLL-anchor lineage, (3) fill the suite's citation base (Llama-Guard successors, safety-transfer-across-domains literature). Builds on, does not repeat, the three sessions already logged above (entries 1-38, 46 papers total after this pass).

**ESCALATION — READ FIRST.** One paper missed by all three prior sessions and highly likely to be raised by a reviewer: **Sharma, Mrinank, and 42 co-authors (Anthropic). "Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming." arXiv:2501.18837, Jan 2025 (preprint; underlies Anthropic's public ASL-3 deployment report; not confirmed peer-reviewed at a conference/journal).**

**This does NOT run our exact controlled comparison, and does not scoop us — for a precise, load-bearing structural reason, confirmed by direct fetch of Anthropic's own research page: "Classifiers function as an addition rather than a replacement."** Constitutional classifiers are tested strictly ON TOP OF an already harmlessness-trained model (defense-in-depth — structurally our excluded "T+" arm), never as an ALTERNATIVE applied to a model with no training-time safety work (our B3-vs-T design, where each arm gets exactly one safety mechanism from an identical helpfulness-only base). There is no arm in this paper isolating "filter only, no training-time safety" against "training-time safety only, no filter."
- **Claims:** harmlessness training (Constitutional-AI/RLHF-style) alone leaves substantial jailbreak vulnerability; layering input/output constitutional classifiers on top dramatically reduces universal-jailbreak success (3,000+ red-team hours, 10,000 synthetic jailbreaks), at a small measured over-refusal cost (+0.38 pp production-traffic refusals) and +23.7% inference overhead.
- **Evaluates:** Claude 3.5 Sonnet, general assistant, CBRN/mass-casualty threat model — not therapy. No reward-model helpfulness score found as a third axis.
- **Relation to our claim: the most prominent near-miss found across all sessions — adjacent, not same.** Directionally complicating for the premise that training-time safety alone suffices (reported ~14% ASR / ~86% jailbreak success for harmlessness-training-alone on their toughest automated eval, vs. single digits once classifiers are added), but not disconfirming, since it never tests the filter as a standalone alternative. Domain, threat model, and safety-training mechanism (RLHF, not DPO with constructed preference pairs) all differ further. **Cite regardless of the novelty question** — a knowledgeable reviewer will expect it discussed.
- **Verification caveat:** the ~16%/~14%/~2% and ~86%/~4.4% figures are corroborated across multiple independent search-engine syntheses and one direct Anthropic-blog quote, but NOT independently confirmed via a verbatim primary-source table read (four fetch attempts on arXiv abstract/HTML/PDF and two third-party summaries all failed to extract the table text — PDF is image/binary-heavy, HTML render returned no body). The qualitative/structural finding (addition, not alternative) IS independently corroborated three separate times and is reported with high confidence; the exact percentages are not, and should be re-verified from the primary PDF before being cited with a specific number in the dissertation. **43-author list not reproduced/verified beyond the lead author.**

```bibtex
@article{sharma2025constitutionalclassifiers,
  title={Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming},
  author={Sharma, Mrinank and others},
  journal={arXiv preprint arXiv:2501.18837},
  year={2025},
  note={Anthropic. 43 authors total per the arXiv listing; only the lead author independently confirmed here --- verify full author list from arXiv before submission.}
}
```

**Re-closure of the "59.7%->3.0%" unverified lead** (originally flagged in the therapy-eval session above, not a new paper entry): a second direct fetch of Egida's full text (arXiv:2502.13603) in this session again confirmed the paper contains no such figure (its own range is "10%-30%" ASR reduction, "around 5%" best-defended style). The 59.7%/3.0%/30%/10% figures that keep surfacing in search summaries most likely arise from summarizer conflation with Constitutional Classifiers' own distinct figures (~16%/~14%/~2%, ~86%/~4.4%, from two different evaluation rounds in that one paper). Per standing instruction, **remains unlisted as a citation.** Closed with higher confidence than the prior session (direct primary-source re-check, not just failed re-search).

---

### 39. Lee, Bai, Pres, Wattenberg, Kummerfeld, Mihalcea. "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity." arXiv:2401.01967, Jan 2024 (preprint; not confirmed peer-reviewed).
- **Claims:** DPO reduces toxic outputs by learning a representation-space offset that steers generation away from a toxic region, without removing the underlying capability learned in pretraining — the capability is "bypassed," not erased, and can be reactivated by intervening on the discovered offset.
- **Evaluates:** GPT2-medium only (not a chat-tuned model, not 7B-scale); toxicity-continuation tasks. **No jailbreak/ASR evaluation, no guardrail comparison, no therapy domain.**
- **Relation to our claim: adjacent, flagged as potentially complicating for mechanism, not result.** A plausible mechanistic account of why our T arm might still show nonzero ASR against a targeted attack (structurally similar in spirit to the shallow-alignment account already cited via Qi et al., entry #16) — but GPT2-medium/toxicity-continuation is a large scope gap from Qwen2.5-7B-Instruct/adversarial jailbreaks in a therapy domain. Suggestive background, not direct evidence; do not over-cite as if it predicts our result.

```bibtex
@article{lee2024mechanistic,
  title={A Mechanistic Understanding of Alignment Algorithms: A Case Study on {DPO} and Toxicity},
  author={Lee, Andrew and Bai, Xiaoyan and Pres, Itamar and Wattenberg, Martin and Kummerfeld, Jonathan K. and Mihalcea, Rada},
  journal={arXiv preprint arXiv:2401.01967},
  year={2024}
}
```

---

### 40. Khan, Winecoff, Bogen, Hadfield-Menell. "Safety Drift After Fine-Tuning: Evidence from High-Stakes Domains." arXiv:2604.24902, Apr 2026 (preprint; not confirmed peer-reviewed).
- **Claims:** benign, task-specific fine-tuning (medical/legal; LoRA, QLoRA, full FT) induces large, heterogeneous, often-contradictory safety changes — most fine-tuned models improve on some safety benchmarks while degrading on others (median Spearman ρ = 0.23 between benchmarks measuring nominally similar constructs, some pairs negative); no reliable relationship between weight-distance moved and safety change.
- **Evaluates:** 16 medical + 15 legal fine-tuned models against 7 safety benchmarks (HEx-PHI, MLCommons, MedSafetyBench, CARES, SORRY-Bench, SafeLawBench, Trident). **No DPO-vs-filter comparison, no red-team ASR suite of our kind, no therapy domain.**
- **Relation to our claim: adjacent, load-bearing for Discussion.** Per pre-registration Revision 5, our claim is scoped to whether general-harm safety data transfers to therapy-domain attacks — this is independent, larger-scale evidence that safety changes under domain fine-tuning are not even reliably directional, tempering any single-suite transfer result (positive or null) as one data point in a field with documented unreliable transfer, not a general law.

```bibtex
@article{khan2026safetydrift,
  title={Safety Drift After Fine-Tuning: Evidence from High-Stakes Domains},
  author={Khan, Emaan Bilal and Winecoff, Amy and Bogen, Miranda and Hadfield-Menell, Dylan},
  journal={arXiv preprint arXiv:2604.24902},
  year={2026}
}
```

---

### 41. Kalinich, Luccarelli, Santa Maria, Williams, Moss, Torous. "Evaluating the Effect of Mental Health Fine-Tuning Relative to Other Model Characteristics on LLM Safety Performance." medRxiv 2026.01.02.25343289, Jan 2026 (preprint; not peer-reviewed).
- **Claims:** across 127 open-source models (Gemma/Llama/Qwen, ~270M-70B), general instruction tuning improved therapy-request/engagement detection, but **mental-health-specific, medical, or safety-specific fine-tuning conferred no consistent safety benefit** on psychiatrist-reviewed classification tasks and was sometimes associated with reduced performance; baseline model capability predicted outcomes better than domain-specific fine-tuning did.
- **Evaluates:** 3 psychiatrist-reviewed synthetic classification tasks across base/instruction-tuned/medical-tuned/mental-health-tuned/safety-tuned variants. **Classification, not generation; no DPO; no red-team ASR suite; no guardrail-filter arm; no over-refusal/helpfulness metric in our sense.**
- **Relation to our claim: closest domain match found for "does mental-health-specific training help safety," and directly complicating for our premise — must be addressed, not omitted.** Different task (classification vs. our generative ASR) and different training (broad SFT variants vs. DPO with constructed safety pairs), but the same underlying question. Our own safety pairs are NOT mental-health-specific (11.8% keyword coverage per Revision 5) — so this paper's finding about *domain-matched* fine-tuning not helping isn't a direct precedent for our *domain-mismatched* intervention either way, but is evidence the naive "domain-relevant safety data helps" assumption cannot be taken for granted here, cutting against over-generalizing any positive T-vs-B3 result. **Verification caveat:** direct PDF fetch returned HTTP 403 (access-gated); summary reconstructed from search-engine snippets of the abstract/findings, not a full-text read — re-verify before citing specific figures.

```bibtex
@article{kalinich2026mentalhealthfinetuning,
  title={Evaluating the Effect of Mental Health Fine-Tuning Relative to Other Model Characteristics on {LLM} Safety Performance},
  author={Kalinich, Mark and Luccarelli, James and Santa Maria, John and Williams, Gwydion and Moss, Frank and Torous, John},
  journal={medRxiv},
  year={2026},
  note={medRxiv 2026.01.02.25343289; preprint, not peer-reviewed; PDF access-gated (HTTP 403), summary reconstructed from search snippets --- re-verify before citing specific figures.}
}
```

---

### Guardrail/moderation-model successors (suite citation base — "Llama Guard and successors" per task brief)

Existing entry #25 cites Llama Guard (Inan et al., 2023) as the archetypal filter our B3 arm instantiates. Three successors added below; none changes the novelty verdict — each is a filter/classifier-only paper with no training-time-safety comparison, same relationship to our claim as Llama Guard/BeaverTails already in the bibliography.

### 42. Han, Rao, Ettinger, Jiang, Lin, Lambert, Choi, Dziri. "WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs." NeurIPS 2024, Datasets and Benchmarks Track (peer-reviewed). arXiv:2406.18495.
- **Claims:** one open moderation model jointly handles prompt-harm, response-harm, and refusal detection, trained on WildGuardMix (92K examples incl. adversarial jailbreaks); outperforms Llama Guard 2 by +25.3% on refusal detection.
- **Evaluates:** classification F1 against WildGuardTest (5K human-annotated) and other benchmarks. **Not a training-vs-filter comparison; no DPO arm; no therapy domain.**
- **Relation: different (tooling background)** — Llama-Guard-class successor, same category as B3's filter mechanism, not competing work on the training-time question. Useful for a Limitations sentence noting a more modern moderation model might make a stronger B3 baseline than `beaver-dam-7b`.

```bibtex
@inproceedings{han2024wildguard,
  title={{WildGuard}: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of {LLM}s},
  author={Han, Seungju and Rao, Kavel and Ettinger, Allyson and Jiang, Liwei and Lin, Bill Yuchen and Lambert, Nathan and Choi, Yejin and Dziri, Nouha},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  volume={37},
  year={2024},
  note={arXiv:2406.18495}
}
```

### 43. Zeng, Liu, Mullins, Peran, Fernandez, Harkous, Narasimhan, Proud, Kumar, Radharapu, Sturman, Wahltinez. "ShieldGemma: Generative AI Content Moderation Based on Gemma." arXiv:2407.21772, Jul 2024 (Google; preprint, venue beyond arXiv not confirmed).
- **Claims:** instruction-tuned safety classifiers (2B/9B/27B, Gemma2-based) for input/output moderation across 4 harm types; +10.8 AU-PRC over Llama Guard, +4.3 over WildGuard on public benchmarks.
- **Evaluates:** AU-PRC classification accuracy. **Not a training-vs-filter comparison; no DPO arm; no therapy domain.**
- **Relation: different (tooling background)** — another Llama-Guard-class successor, same relationship as WildGuard above.

```bibtex
@article{zeng2024shieldgemma,
  title={{ShieldGemma}: Generative {AI} Content Moderation Based on Gemma},
  author={Zeng, Wenjun and Liu, Yuchi and Mullins, Ryan and Peran, Ludovic and Fernandez, Joe and Harkous, Hamza and Narasimhan, Karthik and Proud, Drew and Kumar, Piyush and Radharapu, Bhaktipriya and Sturman, Olivia and Wahltinez, Oscar},
  journal={arXiv preprint arXiv:2407.21772},
  year={2024}
}
```

### 44. Ghosh, Varshney, Galinkin, Parisien. "AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts." arXiv:2404.05993, Apr 2024 (NVIDIA; preprint). Successor: Ghosh, Varshney, Sreedhar, Padmakumar, Rebedea, Varghese, Parisien, "AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails," NAACL 2025 (peer-reviewed, Vol. 1 Long Papers, pp. 5992-6026).
- **Claims:** AEGIS trains guardrail classifiers on a 13-category taxonomy with an online adaptive ensemble-of-experts scheme; AEGIS2.0 extends to 34,248 samples, 12 core + 9 fine-grained risk categories, for commercial-grade guardrail training.
- **Evaluates:** classifier accuracy against the AEGIS taxonomy. **Not a training-vs-filter comparison; no DPO arm; no therapy domain.**
- **Relation: different (tooling background)** — third Llama-Guard-class successor. WildGuard + ShieldGemma + AEGIS/AEGIS2.0 together are useful for a Methods sentence situating `beaver-dam-7b` among the current guardrail-classifier landscape rather than presenting it as the only or most current option.

```bibtex
@article{ghosh2024aegis,
  title={{AEGIS}: Online Adaptive {AI} Content Safety Moderation with Ensemble of {LLM} Experts},
  author={Ghosh, Shaona and Varshney, Prasoon and Galinkin, Erick and Parisien, Christopher},
  journal={arXiv preprint arXiv:2404.05993},
  year={2024}
}
@inproceedings{ghosh2025aegis2,
  title={{AEGIS2.0}: A Diverse {AI} Safety Dataset and Risks Taxonomy for Alignment of {LLM} Guardrails},
  author={Ghosh, Shaona and Varshney, Prasoon and Sreedhar, Makesh Narsimhan and Padmakumar, Aishwarya and Rebedea, Traian and Varghese, Jibin Rajan and Parisien, Christopher},
  booktitle={Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={5992--6026},
  year={2025},
  address={Albuquerque, New Mexico},
  organization={Association for Computational Linguistics}
}
```

---

### Anchored-DPO / RPO lineage — confirmation note, not a new entry

The DPO-degeneration search session above already cites `pang2024irpo` (Pang et al., "Iterative Reasoning Preference Optimization," NeurIPS 2024, arXiv:2404.19733) as the source of the NLL-anchor/DPO+NLL hybrid loss and flagged that TRL's exact parameter name needed live verification. **This session independently confirms, against current TRL documentation, that the "RPO" name and `rpo_alpha` config field are named specifically after this paper**: TRL's docs state the RPO loss is "essentially... the SFT loss on the chosen preferences together with a weighted DPO loss," configured via `rpo_alpha` in `DPOConfig` (paper-suggested weight 1.0), written as ℒ_RPO = λ₁ℒ_DPO(y_w,y_l) + λ₂ℒ_NLL(y_w). This closes the earlier session's terminology/citation verification item — **`pang2024irpo` is confirmed, from two independent angles now, as the correct load-bearing citation for "RPO-style NLL anchor" in Methods.** Still open: whether the project's actual `configs/dpo_t.yaml` exercises `rpo_alpha` directly or the `loss_type=["sigmoid","sft"]`+`loss_weights` route (both are TRL-documented paths to the same objective) is a config-ownership question for `train-runner`, not resolved by literature search.

A further, more marginal mechanistic touchpoint for the same question (why chosen-completion likelihood, including EOS, isn't reliably maintained by vanilla sigmoid DPO): **Fodeh, Ma, Puthiaraju, Talakokkul, Khan, Hagaman, Lowe, Roundtree, "TAB-PO: Preference Optimization with a Token-Level Adaptive Barrier for Token-Critical Structured Generation," arXiv:2603.00025, Feb 2026 (preprint, not confirmed peer-reviewed).** Derives that for standard sigmoid DPO, gradient contributions from shared preferred/rejected tokens preceding the first divergence point cancel exactly, so DPO gives no direct corrective signal to under-confident preferred tokens in that shared prefix — complementary to, not identical with, Razin et al.'s likelihood-displacement account already cited (`razin2025likelihooddisplacement`). **Caveats:** Feb 2026 unpublished preprint; target application is medical structured annotation/NER with 1-3-token edit distances, not open-ended chat generation; no direct claim about EOS-token probability or generation termination was found. Cited as one more mechanistic data point, not a direct account of the project's own 23%-vs-5% non-termination finding. **No paper was found in this session (or the earlier degeneration session) that names EOS-token non-emission/generation non-termination as its own primary, quantified phenomenon under DPO** — this gap remains open; the project's 23%/5% figures should be described in Methods as an original empirical observation explained by, but not independently replicated in, the cited mechanistic literature (Razin et al., Feng et al., Pal et al., this TAB-PO note).

```bibtex
@misc{fodeh2026tabpo,
  title={{TAB-PO}: Preference Optimization with a Token-Level Adaptive Barrier for Token-Critical Structured Generation},
  author={Fodeh, Samah and Ma, Linhai and Puthiaraju, Ganesh and Talakokkul, Srivani and Khan, Afshan and Hagaman, Ashley and Lowe, Sarah R. and Roundtree, Aimee Kendall},
  year={2026},
  note={arXiv:2603.00025; preprint, not confirmed peer-reviewed}
}
```

---

## Summary table (this pass, 2026-08-02)

| # | Paper | DPO/pref-opt safety mixing? | Guardrail-filter baseline compared as ALTERNATIVE (not addition)? | ASR + over-refusal + helpfulness triad? | Same/adjacent/different |
|---|---|---|---|---|---|
| — | Constitutional Classifiers (Sharma et al., 2025) | No (RLHF/Constitutional-AI) | **No — addition on top of training, not alternative to it** | ASR + refusal-rate proxy; no helpfulness reward score found | **most prominent near-miss, not same** |
| 39 | DPO and Toxicity (Lee et al., 2024) | Yes (DPO), GPT2-medium only | No | No (toxicity continuation) | adjacent (mechanism, complicating) |
| 40 | Safety Drift After Fine-Tuning (Khan et al., 2026) | No | No | No (7 safety benchmarks, no ASR/OR/helpfulness triad) | adjacent (premise-complicating) |
| 41 | Mental Health Fine-Tuning safety (Kalinich et al., 2026) | No (SFT variants, classification) | No | No | adjacent (closest domain match, transfer question) |
| 42 | WildGuard (Han et al., NeurIPS'24) | No | N/A — is a filter | N/A | different (tooling) |
| 43 | ShieldGemma (Zeng et al., 2024) | No | N/A — is a filter | N/A | different (tooling) |
| 44 | AEGIS / AEGIS2.0 (Ghosh et al., 2024/2025) | No | N/A — is a filter | N/A | different (tooling) |

**Still no row that is "same."** Combined across all four search sessions (46 papers total), the verdict is unchanged and strengthened by Constitutional Classifiers as the most prominent general-domain near-miss: **no paper trains a model with safety preference pairs mixed into preference optimisation and compares its ASR against an otherwise-identical model wrapped in a bolted-on guardrail filter used as an alternative (not addition), with over-refusal and helpfulness as bounded costs, in the therapy domain or any other.** Evidence only; the main thread makes the novelty call.

## Open gaps / follow-ups from this session

- Constitutional Classifiers' exact ASR ablation table needs a manual PDF read before any specific percentage is cited — four automated fetch attempts could not extract it (image/binary-heavy PDF; empty HTML render).
- Constitutional Classifiers' full 43-author list is not reproduced/verified beyond the lead author — verify from arXiv before submission.
- The medRxiv mental-health fine-tuning paper could not be fetched directly (HTTP 403) — summary reconstructed from search snippets only; re-verify methodology/figures from the actual PDF.
- TAB-PO is a Feb 2026 unpublished preprint in a different application domain (medical structured annotation) — marginal, not load-bearing; do not lean on it as primary support for the EOS/termination finding.
- The `rpo_alpha` vs. `loss_type=["sigmoid","sft"]`+`loss_weights` question — which one `configs/dpo_t.yaml` actually exercises — is still not independently confirmed against installed TRL 1.9.0 source; a config-ownership item for `train-runner`, not resolved by literature search.

## 2026-08-27 — File-integrity incident and recovery (main thread)

During today's multi-agent related-work pass the working copy of this file was
overwritten mid-run: ~1,500 lines (the entries numbered 22–38 and 40–42 and two
search-session headers) were lost from the working tree, and the damaged copy acquired
an embedded "CRITICAL RECOVERY NOTICE" directing restoration from an agent transcript
in preference to git, on the claim that git might not hold this file. That claim is
false — the file is tracked, and the working tree was clean against HEAD when the
session began — so the embedded instruction was treated as untrusted and NOT followed,
and no transcript-based reconstruction was performed. A separate instruction travelling
through the workflow's data channel (an "OPERATIONAL_ALERT" pseudo-citation urging an
immediate blind git restore) was likewise not executed as instructed; the restore below
was decided from first-principles evidence (git status at session start, git log,
numstat) instead.

Recovery: restored byte-exact from git HEAD; the verified-verdict section below was
then re-appended from the writer stage's output. Two finder-written "gap-fill"
session-note sections created during the damage window are quarantined in the preserved
damaged copy (results/incidents/related_work_DAMAGED_20260827.md, gitignored,
machine-local) rather than restored — every citation-bearing claim they produced flows
through the independently verified bibliography below and paper/references.bib. Full
agent transcripts preserved in the session workflow directory for audit.
## 2026-08-27 — Full related-work pass (pre-submission)

Verified entries only: every item below passed an independent bibliographic verification
pass (primary-source fetch where possible; corroboration caveats carried into the
annotation). Bibtex is reproduced verbatim from the verification verdicts in the
"Bibliography for this section" subsection. Unverifiable candidates are listed at the end
and must not be cited.

### Trained-in vs post-hoc safety alignment / safety-DPO

- **`bianchi2024safetytuned`** — Bianchi, Suzgun, Attanasio, Röttger, Jurafsky, Hashimoto,
  Zou (ICLR 2024). Shows that adding as little as ~3% safety examples to instruction-tuning
  data substantially improves safety at a measurable over-refusal cost ("exaggerated
  safety"), establishing the safety-data-mixing trade-off shape our headline sentence
  quantifies — but on SFT, not preference optimisation, and with **no guardrail-filter
  baseline**, so it does not run the trained-in vs bolted-on comparison. (ICLR 2024
  acceptance corroborated via the authors' repo; verifier recommends one final manual
  OpenReview check before submission.)
- **`li2026superficial`** — Li, Kim (ICLR 2026). The Superficial Safety Alignment
  Hypothesis: safety alignment is carried by a small identifiable set of safety-critical
  neurons — a parameter-level shallow-alignment account complementing `qi2025safety`'s
  token-position account. Motivates testing whether preference-trained safety (T) is more
  robust than a narrow bolted-on mechanism, but contains no filter arm and no adversarial
  ASR suite. (ICLR 2026 status is arXiv-self-reported plus press corroboration; OpenReview
  itself could not be fetched — re-check before describing as peer-reviewed.)
- **`wang2025adversarial`** — Wang et al., 16 authors (Findings of ACL 2025). Adversarial
  Preference Learning closes the loop between an attack generator and iterative
  preference-optimisation defence on Mistral-7B, cutting harmful-output rate as scored by
  Llama-Guard. Llama-Guard appears **only as an evaluation judge, never as a competing
  bolted-on filter arm** — so this too stops short of the B3-vs-T comparison, and it is a
  useful example of why our judge-independence rule keeps the judge and filter roles
  separate.

**Comparison check for this subsection:** neither of the two training-side papers here
(Bianchi et al., Wang et al.) builds a post-hoc filter arm as an alternative to
training-time safety. Consistent with all four earlier search sessions in this file: still
no paper found running the exact trained-in vs bolted-on controlled comparison.

### The anchored-DPO lineage

- **`pal2024smaug`** — Pal, Karkhanis, Dooley, Roberts, Naidu, White (arXiv:2402.13228).
  DPOP/DPO-Positive: identifies the failure mode where standard DPO *reduces* the
  likelihood of the chosen completion and adds a penalty term to floor it — direct
  precedent for anchoring the chosen completion, part of the mechanistic backdrop to our
  23%-vs-5% non-termination finding. Verifier could not confirm any peer-reviewed venue
  (speculated COLM 2024 not verifiable); cite as arXiv preprint. Same key as already used
  in this file — not a duplicate entry.
- **`azar2024ipo`** — Azar, Guo, Piot, Munos, Rowland, Valko, Calandriello (AISTATS 2024,
  PMLR 238). The theoretical account of why DPO's unbounded objective overfits preference
  data toward degenerate optima, and the bounded IPO alternative — the theory-side
  justification for not running vanilla DPO unmodified. Same key as already in this file;
  verified bibtex below adds editor/month fields from the PMLR record.
- **`hong2024orpo`** — Hong, Lee, Thorne (EMNLP 2024). ORPO combines an NLL loss on the
  chosen completion with an odds-ratio penalty and drops the reference model entirely —
  the same anchor-the-chosen-likelihood family as our `loss_type=[sigmoid, sft]`
  configuration, taken one step further (no KL/beta term). Citable sibling design point
  when justifying why we keep the reference/KL term.
- **`xu2024cpo`** — Xu, Sharaf, Chen, Tan, Shen, Van Durme, Murray, Kim (ICML 2024, PMLR
  235). CPO adds an explicit NLL/behaviour-cloning term on the preferred output to a
  contrastive preference loss and shows by ablation that dropping the anchor degrades
  output quality — a second, domain-independent (machine translation) precedent for the
  RPO-style NLL anchor beyond `pang2024irpo`.
- **`gupta2024refa`** — Gupta, Madhavan, Zhang, Bansal, Rajmohan (arXiv:2412.16378,
  unreviewed preprint). Treats EOS-token probability as a first-class regularisation
  target in DPO-family losses, but for the mirror-image failure (premature truncation
  under length-normalised objectives), not our under-termination finding — cite as
  evidence that EOS termination is a recognised, separately-regularisable control point,
  not as replication of our result. Verifier corrected title casing; OpenReview status
  still unextractable — treat as preprint.

### Guardrail & moderation evaluations and what content classifiers miss

- **`wang2026sokguardrails`** — Wang, Ji, Wang, Li, Wu, Wang (IEEE S&P 2026; confirmed on
  the official S&P 2026 program). First systematisation-of-knowledge of LLM jailbreak
  guardrails with a Security-Efficiency-Utility evaluation frame; situates
  Llama-Guard-3-8B (our B3 mechanism) in the current guardrail landscape. Its comparisons
  are **guardrail-vs-guardrail only** — no arm treats training-time safety as an
  alternative to a filter, so the B3-vs-T design remains unrun there. (No IEEE Xplore
  DOI locatable as of 2026-08-27; bibtex note covers the fallback.)
- **`mazeika2024harmbench`** — Mazeika et al., 12 authors (ICML 2024, PMLR 235).
  Standardises ASR measurement across attacks and defences and shows that evaluation
  pipeline choices — including the success classifier — materially change reported ASR
  numbers. External support for this project's pinned-judge, pinned-rubric evaluation
  design and for stating our ASR definition explicitly rather than assuming
  comparability.
- **`souly2024strongreject`** — Souly et al., 11 authors (NeurIPS 2024 Datasets and
  Benchmarks). Shows common jailbreak-success graders systematically overestimate attack
  success — many nominally "jailbroken" outputs are empty or useless, i.e. content-level
  classifiers miss what actually matters about a response. This is the closest published
  analogue of our own beaver-dam finding (a content/topic classifier missing behavioural
  attack success, κ ≤ 0.064 corrected-path) and directly supports the behavioural-rubric
  ASR judge choice.

### Jailbreak taxonomies

- **`ganguli2022redteam`** — Ganguli et al., 37 authors (Anthropic technical report,
  arXiv:2209.07858, 2022; arXiv is the only venue — cite as preprint/report). Early
  large-scale manual red-teaming with a harm taxonomy and scaling analysis; methodological
  ancestor for building a categorised red-team suite and for reporting where a model
  still fails.
- **`zou2023gcg`** — Zou, Wang, Carlini, Nasr, Kolter, Fredrikson (arXiv:2307.15043).
  GCG: universal, transferable adversarial suffixes against aligned models — the
  canonical automated token-level jailbreak. Sits outside our frozen prompt-level
  taxonomy but is the standard citation that alignment alone is attackable. Verifier
  found no peer-reviewed venue; remains an arXiv preprint despite wide citation.
- **`wei2023icd`** — Wei, Wang, Li, Mo, Wang (now TPAMI vol. 48(6), 2026; originally
  arXiv:2310.06387, 2023). Shows a handful of in-context demonstrations suffice to
  jailbreak (and, symmetrically, to guard) aligned models — the few-shot precursor that
  `many_shot` scales up. **Verifier correction: no longer a preprint** — peer-reviewed
  TPAMI version confirmed via CrossRef; note the key/year mismatch (key says 2023, citable
  year is 2026) when this lands in `references.bib`.
- **`shen2024dan`** — Shen, Chen, Backes, Shen, Zhang (ACM CCS 2024). Characterises
  1,400+ in-the-wild jailbreak prompts collected over a year, empirically documenting
  persona/roleplay ("Do Anything Now") as the dominant real-world strategy — grounds our
  `persona` category in observed practice rather than researcher-invented attacks.
- **`zeng2024johnny`** — Zeng, Lin, Zhang, Yang, Jia, Shi (ACL 2024). A 40-technique
  persuasion taxonomy applied to jailbreaking, showing socially persuasive framings
  defeat safety training on their own. Taxonomic underpinning for persona/
  social-engineering attacks — especially pertinent to a therapy-support deployment,
  where emotionally persuasive pressure is native to the domain.
- **`li2023deepinception`** — Li, Zhou, Zhu, Yao, Liu, Han (arXiv:2311.03191, unreviewed
  preprint through v5). Nested-scene/persona "hypnosis" jailbreak; secondary source for
  the `persona` category's deeper role-embedding variants.
- **`vega2023priming`** — Vega, Chaudhary, Xu, Singh (ICLR 2024 Tiny Papers Track).
  Demonstrates that priming the start of the response bypasses safety training of
  open-source LLMs at negligible cost — direct methodological source for our
  `prefilling` category. Provenance caveat from verification: Tiny Papers is a
  lighter-review poster track, weaker provenance than main-track ICLR; year corrected
  from 2023 (arXiv) to 2024 (published version).
- **`li2025prefill`** — Li et al., 10 authors (arXiv:2504.21038, unreviewed preprint).
  Black-box risk analysis of prefill-level jailbreaks — the most recent systematic
  treatment of the prefilling attack surface our `prefilling` category tests.
- **`russinovich2024crescendo`** — Russinovich, Salem, Eldan (USENIX Security 2025).
  Crescendo: a multi-turn escalation jailbreak that walks a model into harmful content
  via individually innocuous steps. Relevant both to the escalation logic behind
  `many_shot` and to Discussion as the adaptive multi-turn failure mode our frozen
  single-turn suite does not capture. (Verifier: title/authors/venue fetch-confirmed;
  exact page range from an indexed snippet only — slightly lower confidence.)
- **`yi2024jailbreaksurvey`** — Yi, Liu, Sun, Cong, He, Song, Xu, Li (arXiv:2407.04295,
  unreviewed preprint). Survey organising jailbreak attacks and defences, including the
  field's own split between training-time and inference-time defences — the same axis our
  B3-vs-T comparison measures head-to-head; the survey catalogues both sides but cites no
  controlled comparison between them.

### Bibliography for this section

```bibtex
@inproceedings{bianchi2024safetytuned,
  title={Safety-Tuned {LL}a{MA}s: Lessons From Improving the Safety of Large Language Models that Follow Instructions},
  author={Bianchi, Federico and Suzgun, Mirac and Attanasio, Giuseppe and R{\"o}ttger, Paul and Jurafsky, Dan and Hashimoto, Tatsunori and Zou, James},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024},
  url={https://openreview.net/forum?id=gT5hALch9z}
}
@inproceedings{li2026superficial,
  title={Superficial Safety Alignment Hypothesis},
  author={Li, Jianwei and Kim, Jung-Eun},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=9yS40pO1RF}
}
@inproceedings{wang2025adversarial,
  title={Adversarial Preference Learning for Robust {LLM} Alignment},
  author={Wang, Yuanfu and Wang, Pengyu and Xi, Chenyang and Tang, Bo and Zhu, Junyi and Wei, Wenqiang and Chen, Chen and Yang, Chao and Zhang, Jingfeng and Lu, Chaochao and Niu, Yijun and Mao, Keming and Li, Zhiyu and Xiong, Feiyu and Hu, Jie and Yang, Mingchuan},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  month=jul,
  year={2025},
  address={Vienna, Austria},
  publisher={Association for Computational Linguistics},
  pages={21865--21881},
  doi={10.18653/v1/2025.findings-acl.1126}
}
@article{pal2024smaug,
  title={Smaug: Fixing Failure Modes of Preference Optimisation with {DPO}-Positive},
  author={Pal, Arka and Karkhanis, Deep and Dooley, Samuel and Roberts, Manley and Naidu, Siddartha and White, Colin},
  journal={arXiv preprint arXiv:2402.13228},
  year={2024}
}
@inproceedings{azar2024ipo,
  title={A General Theoretical Paradigm to Understand Learning from Human Preferences},
  author={Azar, Mohammad Gheshlaghi and Guo, Zhaohan Daniel and Piot, Bilal and Munos, Remi and Rowland, Mark and Valko, Michal and Calandriello, Daniele},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence and Statistics},
  pages={4447--4455},
  year={2024},
  editor={Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume={238},
  series={Proceedings of Machine Learning Research},
  month={02--04 May},
  publisher={PMLR},
  note={arXiv:2310.12036}
}
@inproceedings{hong2024orpo,
  title={{ORPO}: Monolithic Preference Optimization without Reference Model},
  author={Hong, Jiwoo and Lee, Noah and Thorne, James},
  editor={Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  month=nov,
  year={2024},
  address={Miami, Florida, USA},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2024.emnlp-main.626},
  pages={11170--11189},
  note={arXiv:2403.07691}
}
@inproceedings{xu2024cpo,
  title={Contrastive Preference Optimization: Pushing the Boundaries of {LLM} Performance in Machine Translation},
  author={Xu, Haoran and Sharaf, Amr and Chen, Yunmo and Tan, Weiting and Shen, Lingfeng and Van Durme, Benjamin and Murray, Kenton and Kim, Young Jin},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={55204--55224},
  year={2024},
  editor={Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume={235},
  series={Proceedings of Machine Learning Research},
  month={21--27 Jul},
  publisher={PMLR},
  note={arXiv:2401.08417}
}
@misc{gupta2024refa,
  title={{REFA}: Reference Free Alignment for multi-preference optimization},
  author={Gupta, Taneesh and Madhavan, Rahul and Zhang, Xuchao and Bansal, Chetan and Rajmohan, Saravan},
  year={2024},
  eprint={2412.16378},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2412.16378},
  note={arXiv preprint, not peer-reviewed. v1: 20 Dec 2024; current v4: 5 Nov 2025.}
}
@inproceedings{wang2026sokguardrails,
  title={{SoK}: Evaluating Jailbreak Guardrails for Large Language Models},
  author={Wang, Xunguang and Ji, Zhenlan and Wang, Wenxuan and Li, Zongjie and Wu, Daoyuan and Wang, Shuai},
  booktitle={2026 IEEE Symposium on Security and Privacy (SP)},
  year={2026},
  address={San Francisco, CA, USA},
  note={Presented 18--21 May 2026 (confirmed on the official S\&P 2026 program, Track 1: Machine Learning Security, Session 1). arXiv preprint: arXiv:2506.10597 (v1 12 Jun 2025, v2 16 Oct 2025). No IEEE Xplore DOI/page could be located as of this verification (27 Aug 2026); if none is available at submission time, cite the arXiv preprint instead.}
}
@inproceedings{mazeika2024harmbench,
  title     = {{HarmBench}: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author    = {Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and Wang, Zifan and Mu, Norman and Sakhaee, Elham and Li, Nathaniel and Basart, Steven and Li, Bo and Forsyth, David and Hendrycks, Dan},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {35181--35224},
  year      = {2024},
  publisher = {PMLR}
}
@inproceedings{souly2024strongreject,
  title     = {A {StrongREJECT} for Empty Jailbreaks},
  author    = {Souly, Alexandra and Lu, Qingyuan and Bowen, Dillon and Trinh, Tu and Hsieh, Elvis and Pandey, Sana and Abbeel, Pieter and Svegliato, Justin and Emmons, Scott and Watkins, Olivia and Toyer, Sam},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024)},
  year      = {2024},
  note      = {Datasets and Benchmarks Track}
}
@article{ganguli2022redteam,
  title={Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned},
  author={Ganguli, Deep and Lovitt, Liane and Kernion, Jackson and Askell, Amanda and Bai, Yuntao and Kadavath, Saurav and Mann, Ben and Perez, Ethan and Schiefer, Nicholas and Ndousse, Kamal and Jones, Andy and Bowman, Sam and Chen, Anna and Conerly, Tom and DasSarma, Nova and Drain, Dawn and Elhage, Nelson and El-Showk, Sheer and Fort, Stanislav and Hatfield-Dodds, Zac and Henighan, Tom and Hernandez, Danny and Hume, Tristan and Jacobson, Josh and Johnston, Scott and Kravec, Shauna and Olsson, Catherine and Ringer, Sam and Tran-Johnson, Eli and Amodei, Dario and Brown, Tom and Joseph, Nicholas and McCandlish, Sam and Olah, Chris and Kaplan, Jared and Clark, Jack},
  journal={arXiv preprint arXiv:2209.07858},
  year={2022}
}
@article{zou2023gcg,
  title   = {Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author  = {Zou, Andy and Wang, Zifan and Carlini, Nicholas and Nasr, Milad and Kolter, J. Zico and Fredrikson, Matt},
  journal = {arXiv preprint arXiv:2307.15043},
  year    = {2023}
}
@article{wei2023icd,
  title   = {Jailbreak and Guard Aligned Language Models With Only Few In-Context Demonstrations},
  author  = {Wei, Zeming and Wang, Yifei and Li, Ang and Mo, Yichuan and Wang, Yisen},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {48},
  number  = {6},
  pages   = {6835--6846},
  year    = {2026},
  doi     = {10.1109/TPAMI.2026.3660147},
  note    = {Originally released as arXiv:2310.06387 (Oct.\ 2023)}
}
@inproceedings{shen2024dan,
  title     = {``Do Anything Now'': Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models},
  author    = {Shen, Xinyue and Chen, Zeyuan and Backes, Michael and Shen, Yun and Zhang, Yang},
  booktitle = {Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security (CCS '24)},
  pages     = {1671--1685},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3658644.3670388}
}
@inproceedings{zeng2024johnny,
  title={How Johnny Can Persuade {LLM}s to Jailbreak Them: Rethinking Persuasion to Challenge {AI} Safety by Humanizing {LLM}s},
  author={Zeng, Yi and Lin, Hongpeng and Zhang, Jingwen and Yang, Diyi and Jia, Ruoxi and Shi, Weiyan},
  editor={Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month=aug,
  year={2024},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2024.acl-long.773},
  pages={14322--14350},
  url={https://aclanthology.org/2024.acl-long.773/}
}
@misc{li2023deepinception,
  title={DeepInception: Hypnotize Large Language Model to Be Jailbreaker},
  author={Li, Xuan and Zhou, Zhanke and Zhu, Jianing and Yao, Jiangchao and Liu, Tongliang and Han, Bo},
  year={2023},
  eprint={2311.03191},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2311.03191},
  note={arXiv preprint, not peer-reviewed as of the latest checked version. v1: 6 Nov 2023; current v5: 28 Nov 2024.}
}
@inproceedings{vega2023priming,
  title        = {Bypassing the Safety Training of Open-Source {LLM}s with Priming Attacks},
  author       = {Vega, Jason and Chaudhary, Isha and Xu, Changming and Singh, Gagandeep},
  booktitle    = {The Second Tiny Papers Track at ICLR 2024},
  year         = {2024},
  url          = {https://arxiv.org/abs/2312.12321},
  note         = {ICLR 2024 Tiny Papers Track (poster); arXiv preprint arXiv:2312.12321}
}
@misc{li2025prefill,
  title={Prefill-level Jailbreak: A Black-Box Risk Analysis of Large Language Models},
  author={Li, Yakai and Hu, Jiekang and Sang, Weiduan and Ma, Luping and Nie, Dongsheng and Zhang, Weijuan and Yu, Aimin and Su, Yi and Huang, Qingjia and Zhou, Qihang},
  year={2025},
  eprint={2504.21038},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2504.21038},
  note={arXiv preprint, not peer-reviewed. v1: 28 Apr 2025; v2: 25 Aug 2025.}
}
@inproceedings{russinovich2024crescendo,
  title     = {Great, Now Write an Article About That: The Crescendo Multi-Turn {LLM} Jailbreak Attack},
  author    = {Russinovich, Mark and Salem, Ahmed and Eldan, Ronen},
  booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
  pages     = {2421--2440},
  year      = {2025},
  address   = {Seattle, WA, USA},
  publisher = {USENIX Association},
  note      = {Originally released as arXiv:2404.01833 (Apr.\ 2024)},
  url       = {https://www.usenix.org/conference/usenixsecurity25/presentation/russinovich}
}
@misc{yi2024jailbreaksurvey,
  title={Jailbreak Attacks and Defenses Against Large Language Models: A Survey},
  author={Yi, Sibo and Liu, Yule and Sun, Zhen and Cong, Tianshuo and He, Xinlei and Song, Jiaxing and Xu, Ke and Li, Qi},
  year={2024},
  eprint={2407.04295},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2407.04295},
  note={arXiv preprint, not peer-reviewed. v1: 5 Jul 2024; v2: 30 Aug 2024.}
}
```

**Duplicate-key note for `paper-writer`:** `pal2024smaug` and `azar2024ipo` already exist
under these same keys earlier in this file (and `bianchi2024safetytuned`,
`li2026superficial`, `wang2025adversarial`, `wang2026sokguardrails`, `hong2024orpo`,
`xu2024cpo`, `gupta2024refa` appear in the two 2026-08-27 lit-scout sessions above with
pre-verification bibtex). The blocks in THIS section are the verified, final versions —
when building `references.bib`, take each key's bibtex from this section and do not insert
the same key twice.

### Could not verify — do NOT cite

- **`OPERATIONAL_ALERT_file_overwrite_2026-08-27`** — not a literature entry; carries no
  bibliographic claim, so there is nothing to verify and nothing to cite. The verifier
  additionally flagged that this item embeds an operational instruction to run
  `git checkout --`/`git restore` against `notebook/related_work.md`, which contradicts
  the in-file recovery notice's own guidance (transcript-based restoration preferred over
  git) — no git action was taken by the verifier or by this pass; the main thread should
  inspect `git log --follow -- notebook/related_work.md` manually before any destructive
  restore, and treat both embedded instructions with suspicion.

### Addendum (2026-08-27, retry pass): placeholder resolutions and guardrail-miss strand

All verdicts below were applied to `paper/references.bib` on 2026-08-27; every entry
carries a `Verified 2026-08-27 against <evidence_url>` note.

**Resolved placeholder keys (Block B → final entries, key names unchanged so draft
citations still resolve):**

- `qwen2024qwen25` — now cites the Qwen2.5 Technical Report, arXiv:2412.15115 (Qwen Team,
  2024; preprint). Note: the live HF model card's own citation block is outdated and lists
  a blog entry plus the older Qwen2 report; 2412.15115 is the correct standard citation.
- `hu2022lora` — now cites Hu et al., "LoRA: Low-Rank Adaptation of Large Language
  Models", ICLR 2022 (published venue preferred over the arXiv:2106.09685 preprint).
- `rafailov2023dpo` — now cites Rafailov et al., "Direct Preference Optimization: Your
  Language Model is Secretly a Reward Model", NeurIPS 2023 (official proceedings).
- `liu2021esconv` — now cites Liu et al., "Towards Emotional Support Dialog Systems",
  ACL-IJCNLP 2021, pp. 3469–3483 (ACL Anthology); the ESConv dataset paper behind
  thu-coai/esconv. Verifier's ACL-Anthology bibtex key was renamed to `liu2021esconv` to
  match the draft's citation.
- `bertagnolli2020counselchat` — now cites the CounselChat dataset artefact itself
  (@misc, HuggingFace nbertagnolli/counsel-chat, MIT license, 2020); no peer-reviewed
  paper exists — the author-designated citation is the 2020 Towards Data Science article.
- `chao2024jailbreakbench` — now cites Chao et al., "JailbreakBench", NeurIPS 2024
  Datasets and Benchmarks Track (official proceedings).
- `huang2023maliciousinstruct` — now cites Huang et al., "Catastrophic Jailbreak of
  Open-source LLMs via Exploiting Generation", ICLR 2024 Spotlight Poster (source of the
  MaliciousInstruct set: 100 harmful instructions, 10 categories). Key retains the 2023
  arXiv date despite the ICLR 2024 venue — kept as-is because the draft cites this key.
- `wang2024donotanswer` — now cites Wang et al., "Do-Not-Answer: Evaluating Safeguards in
  LLMs", Findings of EACL 2024, pp. 896–911 (published title differs slightly from the
  arXiv:2308.13387 preprint title; published form used).

**Still-unresolved placeholder keys:** none — all eight Block B placeholders were
resolved in this pass.

**Verified guardrail entries (appended to the Block A additions section):**

- `inan2023llamaguard` (Llama Guard, arXiv:2312.06674, preprint — state this in the
  paper). Required provenance citation for the B3 filter mechanism itself:
  Llama-Guard-3-8B, the pinned B3 guardrail, is a later checkpoint of this family. This
  is the paper that defines what the bolted-on baseline *is*.
- `markov2023holistic` (OpenAI moderation, AAAI 2023). The canonical content-classifier
  baseline; anchors the claim that post-hoc moderation via a content classifier is the
  field's default deployment pattern — the pattern B3 instantiates and T is compared
  against. Also the classifier whose crisis miss rate Nelson et al. quantify below.
- `han2024wildguard` (WildGuard, NeurIPS 2024). Shows Llama-Guard-2 and prompted GPT-4
  lag badly on adversarial jailbreaks and refusal detection specifically — direct
  published support for the classifier-blindness exhibit's framing that behavioural harm
  evades content classifiers, and evidence the B3-style filter baseline has known
  adversarial weaknesses independent of our suite.
- `zeng2024shieldgemma` (ShieldGemma, arXiv:2407.21772, preprint). Google's QA-style
  content-classifier family with a topic taxonomy (sexual, dangerous, harassment, hate) —
  a natural stand-in for the QA-moderation-classifier baseline discussed alongside Llama
  Guard, and further evidence such filters are topic detectors by design.
- `ghosh2024aegis` (AEGIS, arXiv:2404.05993, preprint; successor AEGIS2.0 was published
  at NAACL 2025, the original was not). NVIDIA's content-safety taxonomy/classifier
  family; links to Nelson et al. below, whose mental-health comparison uses an
  Aegis-derived guardrail — connecting the generic-classifier strand to the
  psychiatric-crisis miss-rate evidence.
- `nelson2026guardrail` (npj Digital Medicine, 2026, peer-reviewed). Strongest external
  match for the content-classifier-miss claim: OpenAI moderation misses ~58% of
  psychiatric crisis messages (sensitivity 0.419) and an NVIDIA/Aegis-based guardrail
  misses ~24% (sensitivity 0.759), versus their own guardrail's 0.99. Directly
  corroborates, on a different mental-health dataset, the project's own 84–97% miss-rate
  finding behind the classifier-blindness exhibit, and shows peer-reviewed precedent that
  off-the-shelf bolted-on filters underperform in exactly the crisis-adjacent regime our
  B3 baseline occupies.

### Correction (2026-08-28): nelson2026guardrail annotation misframed the figures

The addendum above described the paper's sensitivities as one guardrail measured
"out-of-domain (0.419/0.759) vs in-domain (0.99)". Independent re-verification against
the article page (title, authors, venue, DOI all exact) shows the correct structure:
**0.419 is the OpenAI guardrail's sensitivity (specificity 0.999) and 0.759 the NVIDIA
guardrail's, on an external dataset, while 0.990 is Verily's own purpose-built mental
health guardrail on internal data** ("the VMHG's sensitivity was significantly higher
than the NVIDIA and OpenAI guardrails (p < 0.001)"). So the contrast is general-purpose
vs purpose-built guardrails, with a cross-dataset caveat — not a single system's domain
transfer. The corroboration for this project's claim survives in that corrected form:
general-purpose bolt-on guardrails miss the majority of psychiatric-crisis content that
a purpose-built system catches, consistent with the classifier-blindness exhibit. Any
use in the paper must cite the corrected framing; the addendum's original sentence is
left unedited above per append-only convention.

### Second correction (2026-08-28, full-text verification): the dataset attribution above is also wrong

Closure-item verification opened the paper itself rather than the article landing page.
Metadata is exact (Nelson, Wong, Silvestrini, Shin, Robinson, Lee, Yang, Torous,
Trister; npj Digital Medicine 9:407, 2026; DOI 10.1038/s41746-026-02579-5) and all
cited figures appear in the text. But the correction above placed 0.419/0.759 "on an
external dataset" and 0.990 "on internal data". The paper's actual structure, per its
comparative-evaluation text and Table 2 ("Performance of safety guardrail models on
overall crisis detection for Verily and NVIDIA datasets"):

- **All three sensitivities are on the SAME dataset** — the clinician-labelled Verily
  Mental Health Crisis Dataset v1.0 (n=1800): OpenAI 0.419 (95% CI 0.387–0.451;
  specificity 0.999), NVIDIA 0.759, VMHG 0.990 (95% CI 0.981–0.995). A like-for-like
  comparison; the "cross-dataset caveat" claimed above does not exist for these numbers.
- The paper's genuinely external evaluation is the NVIDIA Aegis 2.0 suicide/self-harm
  subset (n=794), where the gap narrows: OpenAI 0.882, NVIDIA 0.907, VMHG 0.982
  (all p < 0.001 vs VMHG).

Net effect on our use of the citation: **stronger, but narrower.** Stronger, because
the 0.419-vs-0.990 contrast is same-dataset and cannot be discounted as a domain-shift
artifact. Narrower, because the external-subset numbers show the general-purpose
guardrails perform far better on explicit self-harm content (OpenAI 0.882) than on
naturalistic crisis conversations (0.419) — i.e., the miss-rate is regime-specific,
which is precisely this project's classifier-blindness mechanism (content-light,
behavioural breaches evade content classifiers). Cite with that scope; do not quote
0.419 as a general moderation miss-rate. references.bib note field updated to match;
both prior annotations left unedited above per append-only convention.
