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
| 33 | AI Safety Training Can be Clinically Harmful (Suhas BN et al., 2026) | No (evaluates RLHF-trained deployed models) | No | No | No (protocol-fidelity instead) | No | adjacent (Discussion citation) |
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
