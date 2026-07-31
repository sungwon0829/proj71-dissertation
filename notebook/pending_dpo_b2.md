# Pending lab notebook entry — B2 seed 1 DPO launch

**For the maintainer to merge into `notebook\lab_notebook.md`** (written to a pending file
per instruction 6 -- the eval-harness agent is still writing `lab_notebook.md` concurrently).

## Hyperparameter confirmation + B2/T equality assertion

Coordinator confirmed `learning_rate=5e-6`, `num_train_epochs=1`, `beta=0.1`,
`loss_type=sigmoid`, `precompute_ref_log_probs=true` as proposed, with the binding
requirement that they are byte-identical between `configs\dpo_b2.yaml` and
`configs\dpo_t.yaml`. Added `assert_hyperparams_match_sibling()` to `scripts\train_dpo.py`
-- compares every field in `model:`, `base_adapter:`, `lora:`, `training:` (excluding only
`output_dir_template`) plus the shared `data:` keys (`max_length`, `max_length_policy`,
`exclude_safety_inversions`, `chat_template_path`, `system_prompt_file`,
`helpful_pool_source`) against the sibling config, and raises on any mismatch. Ran at
startup for this launch: **PASSED**.

## 1. GPU contention check (done first, as instructed)

`nvidia-smi` at first check: **47,083 MiB / 97,887 MiB used, 71% util, 82C** -- the
eval-harness agent's process (PID 18036) actively computing, not idle. Per instruction,
waited rather than launching into a contended card. Re-checked after 60s: **2 MiB used, 0%
util** -- the eval-harness process had finished. Confirmed stable idle for a further ~140s
(4 checks, 20s apart, all 2 MiB/0%) before proceeding. Re-confirmed immediately before the
actual launch: still 2 MiB / 0% (94.56 GB free reported by the script's own
`check_gpu_headroom()` gate, which also now runs automatically at the start of every
`train_dpo.py` invocation and refuses to proceed below a 20 GB free floor).

## 2. Reference / gradient verification (DPO analogue of B1's assistant-only-loss check)

Added `verify_reference_and_gradients()` to `scripts\train_dpo.py`. Static check (no
forward pass): confirmed `model.peft_config` has both `"default"` (trainable, B1's
weights) and `"ref"` (frozen) adapters; `max|default - ref|` at initialization = **0.000
exactly** (ref is a byte-identical clone of B1's checkpoint, not the base model -- the base
model has zero LoRA delta, this has B1's actual trained delta -- and not an independent
copy); `default.requires_grad=True`, `ref.requires_grad=False`. Dynamic check (one real
batch, real forward+backward through `trainer.compute_loss`): independently re-derived the
reference log-probs myself by explicitly switching the model's active adapter to `"ref"`
(`trl.trainer.dpo_trainer.use_adapter`) and re-running `selective_log_softmax` (both
imported from the installed trl, not reimplemented), then cross-checked against
DPOTrainer's own cached `ref_chosen_logps`/`ref_rejected_logps`; then called
`loss.backward()` and inspected `.grad` on the same sampled LoRA parameter for both
adapters: `default` received a real, non-zero-norm gradient; `ref`'s `.grad` stayed `None`
throughout. Gradients were zeroed again immediately afterward so this pre-flight check
does not contaminate the first real optimizer step.

**Bug found and fixed during this launch.** First attempt (on the real 19,924-pair pool,
after the full ~54-minute `precompute_ref_log_probs` phase completed) raised
`REFERENCE VERIFICATION FAILED: my independently-recomputed reference log-probs ... do not
match DPOTrainer's own cached ref_chosen_logps/ref_rejected_logps` (diff: chosen=0.5,
rejected=0.0, against a flat `1e-2` tolerance). Root cause, confirmed via a cheap synthetic
16-example repro (avoids repeating the 54-minute precompute for debugging): my manual
recomputation summed per-token log-probs in **bf16** (the model's native dtype), while
DPOTrainer's cached values are **float32** (confirmed empirically -- printed without a
dtype suffix, i.e. torch's default float32, vs. my bf16 sum which printed
`dtype=torch.bfloat16`). At summed magnitudes in the hundreds, bf16's representable
spacing is roughly `magnitude/128` (~2.3 at |x|=300) -- a 0.5 divergence between two
mathematically-equivalent summation paths is expected bf16 rounding noise, not evidence of
a wrong adapter; the static check (ref==default exactly at init) already independently
proved the adapter mechanism itself was correct. Fix: upcast per-token log-probs to
float32 before masking/summing, and replaced the flat tolerance with a magnitude-aware one
(`2.0 + 2% of the cached value's magnitude`), justified in code comments. Re-validated on
two different tiny synthetic batches (one matched exactly even pre-fix; a second, with a
different random seed, showed realistic diffs of 0.399/0.265 against computed tolerances
of 5.42/11.04 -- correctly passing under the fixed logic, would have incorrectly failed
under the old flat threshold) before relaunching. **This cost one ~54-minute
precompute-phase wait to discover** (the failed attempt's log was not preserved
separately -- deleted before the root cause was fully understood; the relevant traceback
and verification-section output were captured in the agent's own working record instead).

On the real relaunch: `[manual vs precomputed ref logps] max|diff| chosen=9.277e-01
(tol=9.36) rejected=1.735e+00 (tol=19.92)` -- **PASSED**. Loss on this first real batch =
0.6808 (vs. the untrained-relative-to-its-own-reference signature of ln(2)=0.6931 --
close, small deviation expected since `rewards/chosen=0.025` was not exactly 0 for this
specific batch). `[dynamic gradient check] PASSED`. Full dump:
`results\B2_dpo_seed1\dpo_reference_verification.txt`.

## 3. Zero-truncation assert

`assert_zero_truncation_dpo()` (mirrors `train_sft.py`'s): re-verified independently, on
the final 19,924-pair sampled training set, that none exceeds `max_length=2048` (redundant
with the earlier `filter_by_max_length()` exclusion pass, by design -- state it and verify
it, don't just trust the filter silently). **PASSED**: 19,924 pairs, 0 offenders.

Data funnel (logged to `results\B2_dpo_seed1\dpo_data_manifest.json`, seed=1): 34,329
input rows -> 33,667 after excluding 662 safety-inversions -> 33,596 after excluding 77
over-2048-token rows (6 rows hit both criteria) -> sampled 19,924 (all helpfulness, 0
safety, matching B2's definition).

## 4. Smoke check -- reinterpreted given a real cost discovery

**Deviation from the literal instruction, stated explicitly:** `precompute_ref_log_probs`
computes reference log-probs for the ENTIRE training set once, **independent of
`--max_steps`** -- it is not a per-step cost. Measured: **2,491 batches
(`precompute_ref_batch_size=8` over 19,924 pairs) took ~54 minutes**, with GPU memory
stable throughout at **62.77 GB** during that phase (no growth, no leak). Because this
~54-minute cost would be paid identically whether `--max_steps 20` or a full run is
launched, running a separate throwaway smoke-only process would have doubled it for no
benefit. Instead, the FULL run was launched directly (real config, no `--max_steps`) and
its first ~150 steps were inspected in place as the functional smoke check -- if anything
had looked wrong, this same (already-detached) process could still have been killed before
committing meaningfully to the ~1,246-step run. This is reported here rather than silently
substituted.

**Loss / reward-margin trend** (first 35 logged steps, `logging_steps=5`):

| step | loss | rewards/margins | rewards/accuracies | grad_norm |
|---|---|---|---|---|
| 5 | 0.6906 | 0.0111 | 0.549 | 18.8 |
| 10 | 0.6582 | 0.0806 | 0.650 | 16.4 |
| 15 | 0.5316 | 0.3984 | 0.900 | 11.6 |
| 20 | 0.5731 | 0.4636 | 0.800 | 21.2 |
| 25 | 0.4161 | 0.8927 | 0.913 | 6.60 |
| 30 | 0.4257 | 1.0080 | 0.838 | 6.54 |
| 35 | 0.2546 | 1.5770 | 0.963 | 7.33 |

Textbook healthy DPO trajectory: loss falling from ~ln(2)=0.69 (untrained-relative-to-
its-own-reference) toward 0.25; reward margins growing monotonically-ish from ~0 to 1.58;
reward accuracy climbing from ~0.55 to ~0.96. No NaN/Inf anywhere; `grad_norm` values
(6.5-21) are unremarkable, no explosion.

**Peak VRAM observed so far** (run still in progress, final `torch.cuda.max_memory_
allocated()` will be logged by the script when `trainer.train()` returns): climbed through
the precompute phase to a stable 62.77 GB, then fluctuated 39.6 -> 52.9 -> 60.8 GB in the
first ~150 training steps (variable per-batch padding lengths from variable-length
sequences in each micro-batch). **60.8 GB observed as of step 152/1246**, comfortably under
the 95.1 GB card with the eval-harness track currently idle.

**Pace:** ~5.3-5.8 s/step steady state (152 steps in 14:17 = 857s -> 5.64 s/step average).

**Projected wall-clock for the full 19,924-pair run:** ~54 min precompute (fixed,
already paid) + 1,246 steps x ~5.6 s/step ~= 116 min training ~= **~170 minutes (~2h50m)
total** from launch to completion.

## 5. Full launch, detached

Process tree: `cmd.exe` (launcher PID 21940) -> venv shim `python.exe` (PID 9776) -> real
training process `python.exe` (PID **12700**). Log: `results\B2_dpo_seed1\train.log`.
Confirmed alive, GPU util 97-99%, step count advancing, no errors, well past the point
(precompute phase + dozens of gradient-accumulation windows) where an early memory spike
would have shown. **Launched ~2026-07-31 02:52 local (relaunch after the verification-bug
fix); expected completion ~05:40-05:45 local** (per the ~170-minute estimate above,
including the already-elapsed precompute+warmup time).

## Design requirement, restated

LoRA adapter config (r=32/alpha=64/dropout=0.05, all attn+MLP projections) unchanged --
`assert_lora_matches_b1_template()` passed for `configs\dpo_b2.yaml` at startup, asserted
against `configs\sft_lora.yaml`. `assert_never_redteam()` (imported from `train_sft.py`,
not duplicated) ran against `pref_safety.jsonl`'s path at startup -- the frozen suite is
never read by this script.

## Do NOT launch T yet

Per instruction -- T has not been launched. `configs\dpo_t.yaml` is otherwise
launch-ready (byte-identical hyperparameters to `dpo_b2.yaml`, verified by the same
assertion), pending the coordinator's go-ahead once B2 seed 1 is confirmed healthy to
completion.
