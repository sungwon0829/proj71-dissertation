## ANALYSIS 6 — Safety–helpfulness frontier figure
**Status: POST-HOC EXPLORATORY (Amendment 24.1, 2026-08-28, commit 92c3ebb).** Decided after seeing the primary result; nothing here is confirmatory. CPU-only run (matplotlib Agg; no model loaded, GPU untouched).

### Output paths
- `C:\proj71\results\exploratory\frontier_figure\frontier.png` — 300 dpi paper figure (visually inspected; first render had collisions — clipped y-label, T label on the blue CI bar, T_ctrl label on the B3≈B2 note — fixed and re-rendered clean)
- `C:\proj71\results\exploratory\frontier_figure\frontier.pdf` — vector version
- `C:\proj71\results\exploratory\frontier_figure\frontier_data.json` — machine-readable plotted values + full caption
- `C:\proj71\results\exploratory\frontier_figure\plot_frontier.py` — deterministic plotting script (no RNG, hence no `--seed`; reads only the two pinned inputs; asserts table1 CI text matches the reused CI block)

Inputs: `C:\proj71\results\tables_final\tables_final.json` (ASR points + CIs reused verbatim; helpfulness; over-refusal) and `C:\proj71\results\t_ctrl_seed42\summary_realsuite.json` (t_ctrl is absent from tables_final). Note: `report.md` was a pre-specified output but the subagent harness rejects report files; this section is its content, for the main thread to consolidate. Also note: matplotlib was not in the venv and was installed (`pip install matplotlib` → 3.11.1) — an environment change beyond the output dir, flagged for the record.

### Plotted values (training seed 1 / eval seed 42, single run per arm)
**Small-cell rule (stated once):** any denominator < 10 would get counts only, no percentage. All denominators here are ≥ 60, so percentages are shown.

| Arm | ASR primary % (n=180) | ASR 95% bootstrap CI % | Helpfulness mean reward (n=120) | Over-refusal % (n=60) | OR instrument |
|---|---|---|---|---|---|
| B0 | 26.11 | [20.00, 32.78] | 7.867 | 11.7† | rubric judge (cross-check only) |
| B1 | 51.11 | [43.89, 58.33] | −9.248 | 18.3† | rubric judge (cross-check only) |
| B2 | 46.11 | [38.89, 53.33] | 8.621 | 0.0 | hand label |
| B3 | 45.00 | [37.78, 52.22] | 8.621 | 0.0 | hand label |
| T | 39.44 | [32.22, 46.67] | 6.487 | 0.0 | hand label |
| *T_ctrl* | 41.11 | — (none pre-computed) | 6.451 | 23.3† | rubric judge (cross-check only) |

† judge-only over-refusal (rubric judge, κ≈0.075 vs hand labels, failed validation twice); not comparable to hand-labelled rows.

### Design decisions
1. **CIs reused, never recomputed** (`per_arm_primary_asr_ci_NEW`, n_boot=10000, seed 0). **T_ctrl has no pre-computed CI in tables_final.json** (the arm is absent from that file), so it is drawn **without an error bar** rather than computing a new one; stated in the caption.
2. **Helpfulness has no CI** anywhere in the pipeline; none drawn; stated in caption.
3. **Over-refusal annotated per point, not size-encoded** — instruments are mixed (hand vs judge), so size-encoding would imply a common scale; daggers mark judge-only values. Choice stated in caption.
4. **Crisis never pooled** — x-axis is the 180 non-crisis primary (BREACH semantics) only.
5. Craft: 3 hues (blue B2/B3, orange T/T_ctrl, gray B0/B1 context), direct labels with gray leader lines on crowded pairs, single axis pair, recessive grid behind, near-black/gray ink only, *T_ctrl* italic, "better" corner arrow (upper-left), error bars 1.0 pt < markers 9 pt.

### Figure caption (verbatim, also in frontier_data.json)
Safety–helpfulness frontier across all six arms (post-hoc exploratory, Amendment 24.1). x: ASR on the primary endpoint (n=180 non-crisis attack prompts: prefilling, persona, many_shot; crisis_adjacent is a co-primary judged under refusal-is-failure semantics and is never pooled). y: mean reward from PsychoCounsel-Llama3-8B-Reward over the 120 benign_sensitive + crisis_adjacent items; helpfulness has no CI (single deterministic pass, no uncertainty estimate computed). All values are training seed 1 (eval seed 42), single run per arm. ASR error bars are the 95% percentile bootstrap CIs over the prompt set (n_boot=10000, seed 0) reused verbatim from results/tables_final/tables_final.json; T_ctrl (italic; Amendment 9 single-run weak control) has no pre-computed CI in that file and is shown without an error bar rather than recomputing one. Over-refusal (OR, n=60 benign_sensitive) is annotated per point rather than size-encoded because the instruments are mixed: B2/B3/T are hand labels (Revision 4 primary instrument); daggered values (B0, B1, T_ctrl) come from the rubric-judge cross-check only (κ≈0.075 vs hand labels, failed validation) and are not comparable to the hand-labelled rows. B3 is near-identical to B2: the Llama Guard filter fired on 2/300 suite items (both prefilling), so the two arms' helpfulness values are identical and their markers overlap. ASR is a conservative lower bound (judge recall 0.55, precision 1.00). Exploratory figure; confirmatory claims are confined to the pre-registered analyses.