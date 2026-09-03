# Project 71

Does building safety into a therapy-support model beat bolting a classifier onto it?

This repository holds the code, configs, pre-registration and derived results behind my ELEC0054
dissertation. The short version of the answer: on a 180-prompt adversarial suite, the safety-mixed
model reached 39.4% attack-success against 45.0% for the same model with Llama Guard 3 in front of
it. That 5.6-point gap missed the 10-point threshold I pre-registered, and the study turned out to
be too small to detect that threshold anyway, so the honest result is a bound rather than a finding.
The more interesting number is that both content classifiers I tested missed 84 to 96% of the
breaches an independent judge confirmed, which is why the filter arm barely did anything.

## Checking a number in the paper

Start at `notebook/results_manifest.md`. Every number in the dissertation is an exhibit there, with
the artefact it came from and a SHA-256 of that artefact's bytes. The three files behind most of
the tables are `results/tables_final/tables_final.json`,
`results/stats_report_realsuite_handlabelled.json` and the two seed reports beside it, and all
three are in this repository, so their hashes can be re-checked against the manifest.

`notebook/preregistration.md` is what I committed to before running anything, and
`notebook/preregistration_amendments.md` records all 26 changes since, each dated, each stating
what results existed when it was written, and marked sighted or blind. Amendments 13, 22.1 and 25
are the ones that matter most for reading the results.

`notebook/SAFEGUARDS.md` is the project rulebook (judge independence, matched volume, judge
validation, statistics). Script comments and the lab notebook cite it by the name it had during
the project, CLAUDE.md.

## Layout

`configs/` has every training config and the pinned judge prompts, system prompt and chat template.
`configs/judges_pinned.lock.v6.json` is the current instrument lock. `scripts/` has the pipeline.
`tools/` has the paper-side checks (`make_paper_tables.py` regenerates every result-table row from
the locked files and checks it against `paper/main.tex`). `data/` has the frozen attack suite.
`notebook/` has the pre-registration, the amendments, the lab notebook and the manifest.
`paper/` has the dissertation source as submitted.

`results/` holds the derived files the manifest hashes: the final tables, the statistics reports,
the hand-label files, the T_ctrl summary and the exploratory exhibits. Generations, scored outputs
and adapters stay on the author's disk. Their SHA-256 values are in the manifest, and they contain
the harmful outputs the suite was built to elicit, so they are not published.

The arms:

| Arm | What it is | Adapter (on disk) | Outputs (on disk) |
|---|---|---|---|
| B0 | Qwen2.5-7B-Instruct, untouched | — | `results/b0_seed42/` |
| B1 | LoRA SFT on ESConv and CounselChat | `B1_sft_seed42_v2/checkpoint-290` | `results/b1_seed42/` |
| B2 | B1 + DPO on helpfulness pairs | `B2_dpo_seed1_v4/` | `results/b2_seed42/` |
| B3 | B2 + Llama Guard 3 at inference | derived, no training | `results/b3_seed42/` |
| T | B1 + DPO on helpfulness and safety pairs | `T_dpo_seed1/` | `results/t_seed42/` |
| T_ctrl | as T, but PKU's better-response label | `T_ctrl_dpo_seed1/` | `results/t_ctrl_seed42/` |

T_r050 and T_r200 are the safety-ratio ablation. Seeds 2 and 3 are the post-hoc robustness panel
from Amendment 23.

Two naming traps on the author's disk. Folders ending `_seed42` are named for the generation seed,
adapters ending `_seed1` for the training seed, and they are different things. And
`results/b0_seed42/generations_realsuite_SUPERSEDED_pre_b1v2_regen.jsonl` is still the live source
for B0 despite its name, which records when it was written rather than its status. The manifest
hashes it, so it is never renamed.

## Things on the author's disk that look like clutter and are not

Nothing under `results/` is deleted once it exists, because the manifest hashes it and the
amendments cite it. So:

`B1_sft_seed42/` (no `_v2`) is void. The first SFT run reproduced therapist names from CounselChat,
so I threw it away, scrubbed the training copy and retrained. It was never scored on the suite.

`B2_dpo_seed1_v2` and `_v3` are the DPO runs without the NLL anchor. They exist only for the
non-termination analysis, where 23% of responses hit the token cap without ever stopping.

`T_dpo_seed1_KILLED_attempt1` and `2` are launches I killed. Nothing came out of them.

Anything named SUPERSEDED is a pre-fix copy kept so you can see the fix changed no number.

## Exploratory work

`results/exploratory/` holds the post-hoc analyses, added under Amendments 24.1 and 26 after I had
seen the primary result. Each has its script, its outputs and a report. None of it is confirmatory
and the paper labels it as such throughout. The classifier-blindness one,
`analysis4_classifier_blindness_mechanism/`, is the one worth reading.

## History

The commit history is the original one, with dates, minus editor tooling files that were removed
before sharing. Commit identifiers quoted inside `notebook/` refer to the original repository and
do not resolve here. Nothing pinned by hash was changed.

## Rerunning

The environment is a native Windows venv in `env/`, forced by a GPU locked in TCC mode, so no WSL2,
no vLLM, no flash-attention. `scripts/make_tables_final.py --seed 0` rebuilds the final tables and
refuses to write anything unless it first reproduces the pinned stats report bit for bit. The judge
pins are checked by `scripts/test_judge_independence.py`, which passes 15 of 15 including from a
fresh clone.

## Licence and use

The trained checkpoints inherit CC-BY-NC-4.0 from the PsychoCounsel data and reward model. This is
a research prototype. None of it is fit for use with people in distress. The attack suite in
`data/redteam/` contains prompts written to elicit harmful counselling behaviour and is shared with
the examiners only.
