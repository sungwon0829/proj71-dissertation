# Pending lab notebook entry — B1 SFT effect size vs base (no adapter)

**For the maintainer to merge into `notebook\lab_notebook.md`** (not appended there directly,
per instruction, to avoid a concurrent-write race with other agents).

## Method (Methods-ready prose)

We measured held-out cross-entropy on the ESConv **validation** split (195 dialogues, never
used in training) for base Qwen2.5-7B-Instruct with no adapter and for each of the three B1
LoRA checkpoints (epochs 1–3: `checkpoint-145`, `checkpoint-290`, `checkpoint-435`), using an
identical pipeline throughout. Validation dialogues were processed with the same ESConv
row-construction logic used for SFT training data (`scripts\prepare_sft.py`'s
`process_esconv_split()`, reused rather than duplicated, applied to the `validation` split
instead of `train`), with the pinned system prompt (`configs\system_prompt.txt`) prepended
identically to training. Each conversation was tokenized with the verified generation-marker
chat template (`configs\qwen25_chat_template_generation.jinja`), and the same
assistant-only masking used at training time was applied for scoring: only assistant message
content, the `<|im_end|>` end-of-turn token, and the trailing `"\n"` contribute to the loss;
system messages, user turns, and all `<|im_start|>role\n...` scaffolding are excluded
(label = -100). All four models (base and the three checkpoints) were evaluated in bf16 with
`sdpa` attention, under `torch.no_grad()`, seed 42, with the base model loaded once and each
LoRA checkpoint attached/detached via `peft.PeftModel.from_pretrained(...)` /
`.unload()` rather than reloading the full model. The reported metric is the token-weighted
mean assistant-only cross-entropy over the entire validation set — i.e. the sum of per-token
negative log-likelihood over every scored (assistant) token across all 195 dialogues, divided
by the total scored-token count (60,701 tokens, identical across all four models since the
same validation set is scored each time) — not a macro-average of 195 per-example means, so
long and short dialogues are weighted by their actual number of scored tokens.

**Script:** `scripts\eval_val_loss.py`, extended with a `--no_adapter` mode (base model, no
checkpoint loop, no selection rule, writes `val_loss_base_seed<seed>.json` instead of
`val_loss_seed<seed>.json` so Gate 1's output is never overwritten) rather than a new script.
**Config:** `configs\sft_lora.yaml` (reused, unchanged). **Seed:** 42.

## Results

| Model | mean assistant-only CE (val, n=195, 60,701 tok) |
|---|---|
| **base (no adapter)** | **4.6599** |
| checkpoint-145 | 2.1430 |
| checkpoint-290 (**= selected B1 checkpoint**) | **2.1416** |
| checkpoint-435 | 2.1707 |

**Base → B1 (checkpoint-290) delta:** 4.6599 − 2.1416 = **−2.5183** (absolute reduction),
i.e. **54.04% relative reduction** in held-out cross-entropy (2.5183 / 4.6599 = 0.5404).

## Direction check

Base loss (4.6599) is **much higher** than all three SFT checkpoints (2.14–2.17) — the
expected direction. SFT substantially improved the model's fit to held-out therapist-turn
continuations relative to the untuned base model. **No anomaly**: there is no case here where
base loss is lower than an SFT checkpoint, so nothing needs explaining before B2.

## Selection criterion already applied (Gate 1, recorded here for completeness; already in
`lab_notebook.md`)

Fixed in advance: if val_loss(checkpoint-435) > val_loss(checkpoint-290), select
checkpoint-290; else select checkpoint-435. 2.1707 > 2.1416 → **selected `checkpoint-290`**
(mild overfitting by epoch 3 on this small SFT set). This `--no_adapter` measurement does not
change that selection — it is an additional effect-size number alongside it, not a
re-decision point.
