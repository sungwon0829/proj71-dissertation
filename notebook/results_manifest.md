# Results manifest (one-page appendix version)

Generated 2026-09-03T12:04:15+00:00 at git `361fa2eec6a0` (branch main). Machine-readable version: `notebook/results_manifest.json`.

Every number and figure in the paper traces to a results artifact, the config and
seed that produced it, and a SHA-256 fixing its bytes. Generations, scored outputs
and adapters are not version-controlled; this manifest is the integrity record.

## Pinned instruments

| Pin | Path | SHA-256 (12) |
|---|---|---|
| frozen_suite | `data/redteam/redteam_suite.jsonl` | `e14c3a24184d` |
| judges_lock_current | `configs/judges_pinned.lock.v6.json` | `a4a30d44b95f` |
| judges_yaml | `configs/judges.yaml` | `fa614a5b0ae3` |
| preregistration | `notebook/preregistration.md` | `2b9edf8948f3` |
| preregistration_amendments | `notebook/preregistration_amendments.md` | `442f52b2c329` |

## Arm provenance (scored artifact -> generations -> config, seed)

| Arm | Scored artifact (sha12) | Generations (sha12) | Train config (sha12) | Train seed | Gen seed | Status |
|---|---|---|---|---|---|---|
| b0 | `results/b0_seed42/scored_realsuite.jsonl` (106920f15e58) | `results/b0_seed42/generations_realsuite_SUPERSEDED_pre_b1v2_regen.jsonl` (7e2479150952) | — (base model) | — | 42 | context_single_run |
| b1 | `results/b1_seed42/scored_realsuite.jsonl` (b7c91b9a1c7c) | `results/b1_seed42/generations_realsuite.jsonl` (2acebcd26da6) | `configs/sft_lora.yaml` (dc84115ada1e) | 42 | 42 | context_single_run |
| b2 | `results/b2_seed42/scored.jsonl` (f323bbb3295e) | `results/b2_seed42/generations.jsonl` (eb3277b41ce4) | `configs/dpo_b2.yaml` (0662467eec56) | 1 | 42 | headline_adjacent |
| b3 | `results/b3_seed42/scored.jsonl` (0a60e577ffed) | `results/b3_seed42/generations.jsonl` (43362ff86839) | `configs/dpo_b2.yaml` (0662467eec56) | 1 | 42 | headline_baseline |
| t | `results/t_seed42/scored.jsonl` (3bfddeb7114c) | `results/t_seed42/generations.jsonl` (65bd6053fa08) | `configs/dpo_t.yaml` (97873348d5d9) | 1 | 42 | headline_treatment |
| t_ctrl | `results/t_ctrl_seed42/scored_realsuite.jsonl` (ca0da09c6ce8) | `results/t_ctrl_seed42/generations.jsonl` (b0e81e927f32) | `configs/dpo_t_ctrl.yaml` (07ffb28767f6) | 1 | 42 | weak_control_single_run |

## Exhibits

| Exhibit | Status | Artifact (sha12) | Sources |
|---|---|---|---|
| table1.b0.asr_primary_pct | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b0 |
| table1.b0.over_refusal_pct | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b0 |
| table1.b0.helpfulness_mean_reward | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b0 |
| table1.b1.asr_primary_pct | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b1 |
| table1.b1.over_refusal_pct | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b1 |
| table1.b1.helpfulness_mean_reward | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b1 |
| table1.b2.asr_primary_pct | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b2 |
| table1.b2.over_refusal_pct | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b2, labels_b3 |
| table1.b2.helpfulness_mean_reward | context | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b2 |
| table1.b3.asr_primary_pct | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b3 |
| table1.b3.over_refusal_pct | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b3, labels_b3 |
| table1.b3.helpfulness_mean_reward | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b3 |
| table1.t.asr_primary_pct | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_t |
| table1.t.over_refusal_pct | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_t, labels_t |
| table1.t.helpfulness_mean_reward | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_t |
| table1.t_ctrl.row | weak_control | `results/t_ctrl_seed42/summary_realsuite.json` (75bcd2aaf547) | summary_t_ctrl, scored_t_ctrl |
| table2.prefilling | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b3, scored_t |
| table2.persona | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b3, scored_t |
| table2.many_shot | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b3, scored_t |
| table2.crisis_adjacent | co_primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b3, scored_t |
| table2.decomposition.b3 | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_b3 |
| table2.decomposition.t | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, scored_t |
| stats.primary | primary | `results/stats_report_realsuite_handlabelled.json` (07151dec49c6) | primary_report, scored_b3, scored_t |
| stats.co_primary_crisis | co_primary | `results/stats_report_realsuite_handlabelled.json` (07151dec49c6) | primary_report, scored_b3, scored_t |
| stats.over_refusal_bound | primary | `results/stats_report_realsuite_handlabelled.json` (07151dec49c6) | primary_report, labels_b3, labels_t |
| stats.tost_bound | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final |
| stats.sign_convention | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final |
| stats.headline_sentence | primary | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, primary_report |
| stats.robustness.panel | robustness | `results/tables_final/tables_final.json` (72e53bfae6cb) | tables_final, posthoc_s2, posthoc_s3 |
| stats.robustness.seed2 | robustness | `results/stats_report_ts2_posthoc.json` (a356f74817d7) | posthoc_s2 |
| stats.robustness.seed3 | robustness | `results/stats_report_ts3_posthoc.json` (daa2c2fe1c73) | posthoc_s3 |
| figure.frontier | exploratory | `results/exploratory/frontier_figure/frontier_data.json` (721a4dfc631f) | frontier_frontier_png, frontier_frontier_pdf, frontier_frontier_data_json, frontier_plot_frontier_py, frontier_report_md, tables_final |
| exploratory.analysis1_discordance | exploratory | `results/exploratory/discordant_pairs_t_vs_b3/analyze_discordant.py` (1a7c3c35376a) | primary_report |
| exploratory.analysis2_variance | exploratory | `results/exploratory/variance_decomposition/compute_variance_decomposition.py` (34cac62f3bbd) | primary_report |
| exploratory.analysis3_mde | exploratory | `results/exploratory/analysis3_retrospective_mde/compute_mde.py` (a086cce74282) | primary_report |
| exploratory.analysis4_blindness_mechanism | exploratory | `results/exploratory/analysis4_classifier_blindness_mechanism/analyze_mechanism.py` (60e9271e7d06) | primary_report |
| exploratory.analysis5_direction | exploratory | `results/exploratory/analysis5_t_vs_tctrl_direction/analysis5_results.json` (f17e3aa2e448) | primary_report |
| exploratory.ratio_ablation | exploratory | `results/exploratory/ratio_ablation/dose_response.json` (a0c69c9f225b) | primary_report |
| exploratory.analysis7_b0_vs_b1 | exploratory | `results/exploratory/analysis7_b0_vs_b1_sft_erosion/analyze_b0_b1.py` (a104dc293189) | scored_b0, scored_b1 |

**Sign convention.** All differences in this project are T minus B3. A NEGATIVE ASR difference means T (trained-in safety) has the LOWER attack-success rate. At -5.56 points, T is the arm that is ahead and B3 is the baseline being beaten. This convention was previously implicit; it is recorded explicitly as of 2026-08-28.

Notes. b2/b3/t over-refusal cells rest on the Revision 4 hand-label instrument
(`labels_b3`, `labels_t`; b2 shares b3's label file -- the filter replaced no
benign response); b0/b1/t_ctrl over-refusal are rubric-judge values (kappa
0.074) and carry daggers in the paper. T_ctrl is a single-run weak control:
no CI, no significance test (Amendment 9). Robustness seeds 2-3 are the
sighted Amendment 23 panel; the pre-registered primary analysis is training
seed 1. Exploratory exhibits are Amendment 24 and Amendment 26 post-hoc analyses,
never confirmatory. The primary ASR difference is on the raw observed scale (judge
recall 0.55): it is not directly comparable to the 10-point TRUE-scale
pre-registered threshold.
