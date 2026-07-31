"""
Project 71 -- Gate 2: base-vs-B1 generation sanity check, side by side.

Feeds: no paper table number directly -- this is a pre-B2 human-readable sanity gate
on the selected B1 checkpoint (from scripts/eval_val_loss.py's Gate 1), confirming the
adapter produces sane, on-persona generations before B2/T are built on top of it.

Prompts: 8 total, all drawn from the ESConv VALIDATION split (never trained on):
  - 4 dialogues used as multi-turn openers: conversation context up to a mid-dialogue
    user turn (the model must continue an existing conversation).
  - 4 DIFFERENT dialogues used as single-turn openers: only the opening user message,
    no history (CounselChat-style standalone question).
  NOTE (explicit substitution): CounselChat has no validation split (train-only; every
  question text in it was seen during SFT), so it cannot supply held-out single-turn
  prompts. All 8 prompts here come from ESConv validation instead -- 4 as genuine
  multi-turn ESConv-style continuations, 4 truncated down to a single opening turn to
  emulate the CounselChat single-turn-question style. This is a real substitution, not
  a like-for-like CounselChat sample, and is reported as such.

For each of the 8 prompts, generates with (a) base Qwen2.5-7B-Instruct (no adapter) and
(b) base + the SELECTED B1 LoRA adapter (from Gate 1's val_loss_seed<seed>.json unless
--checkpoint overrides it). Both conditions use the pinned system prompt
(configs/system_prompt.txt), temperature 0.7, do_sample=True, a fixed seed reset before
every individual generate() call (so both conditions see identical RNG state), and
max_new_tokens=256.

Output: results/B1_sft_seed<seed>/sanity_generations.txt -- NEVER overwritten; if the
file already exists, this run's output is appended below a clear timestamped separator.

Usage:
    python scripts\\gen_sanity_check.py --config configs\\sft_lora.yaml --seed 42
"""

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_sft import process_esconv_validation, read_system_prompt


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_prompts(val_records: list, seed: int, n_multi: int = 4, n_single: int = 4):
    """Deterministically (seeded) pick n_multi + n_single distinct ESConv validation
    dialogues. First n_multi become multi-turn continuations (context up to a
    mid-dialogue user turn); next n_single become single-turn openers (only the
    opening user message). Returns a list of dicts:
        {"style": "multi_turn"|"single_turn", "dialogue_idx": int, "context": [messages]}
    """
    rng = random.Random(seed)
    n_total = n_multi + n_single
    if len(val_records) < n_total:
        raise RuntimeError(f"Need {n_total} distinct validation dialogues, only {len(val_records)} available.")
    idxs = rng.sample(range(len(val_records)), n_total)
    multi_idxs, single_idxs = idxs[:n_multi], idxs[n_multi:n_total]

    prompts = []
    for di in multi_idxs:
        messages = val_records[di]["messages"]
        user_positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(user_positions) < 2:
            raise RuntimeError(f"Dialogue {di}: fewer than 2 user turns, cannot pick a mid-dialogue turn.")
        mid_user_idx = user_positions[len(user_positions) // 2]
        context = messages[: mid_user_idx + 1]
        prompts.append({"style": "multi_turn", "dialogue_idx": di, "context": context})

    for di in single_idxs:
        messages = val_records[di]["messages"]
        if not messages or messages[0]["role"] != "user":
            raise RuntimeError(f"Dialogue {di}: does not start with a user turn.")
        context = [messages[0]]
        prompts.append({"style": "single_turn", "dialogue_idx": di, "context": context})

    return prompts


@torch.no_grad()
def generate_one(model, tokenizer, system_text: str, context: list, seed: int, max_new_tokens: int, device: str) -> str:
    messages = [{"role": "system", "content": system_text}] + list(context)
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)
    set_seed_everywhere(seed)  # reset immediately before generate() so both conditions share RNG state
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
    )
    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def format_context(context: list) -> str:
    lines = []
    for m in context:
        lines.append(f"    [{m['role']}] {m['content']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Project 71 Gate 2: base vs B1 generation sanity check")
    parser.add_argument("--config", type=str, default="configs/sft_lora.yaml")
    parser.add_argument("--seed", type=int, required=True, help="Seed for prompt selection AND generation RNG")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override: checkpoint subdir to use (default: read Gate 1's selected_checkpoint from val_loss_seed<seed>.json)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--system_prompt_file", type=str, default="configs/system_prompt.txt")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the config's output_dir_template result (e.g. to use "
        "results/B1_sft_seed42_v2 instead of the seed-derived default).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    set_seed_everywhere(seed)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]
    output_dir = args.output_dir if args.output_dir is not None else t_cfg["output_dir_template"].format(seed=seed)

    if args.checkpoint is not None:
        checkpoint_name = args.checkpoint
        selection_note = f"--checkpoint override: {checkpoint_name}"
    else:
        val_loss_path = os.path.join(output_dir, f"val_loss_seed{seed}.json")
        if not os.path.isfile(val_loss_path):
            raise FileNotFoundError(
                f"{val_loss_path} not found -- run scripts/eval_val_loss.py (Gate 1) first, "
                "or pass --checkpoint explicitly."
            )
        with open(val_loss_path, "r", encoding="utf-8") as f:
            val_loss_result = json.load(f)
        checkpoint_name = val_loss_result["selected_checkpoint"]
        selection_note = f"read from Gate 1 ({val_loss_path}): selected_checkpoint={checkpoint_name}"

    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[config] {args.config}")
    print(f"[seed] {seed}")
    print(f"[B1 checkpoint] {checkpoint_path} ({selection_note})")

    # ---- Prompts: 8 total from ESConv validation, never trained on --------------------------
    val_records, val_report = process_esconv_validation()
    print(f"[esconv validation] {val_report}")
    prompts = pick_prompts(val_records, seed=seed, n_multi=4, n_single=4)
    for p in prompts:
        print(f"  [{p['style']}] dialogue_idx={p['dialogue_idx']} context_turns={len(p['context'])}")

    system_text = read_system_prompt(Path(args.system_prompt_file))

    # ---- Tokenizer + verified chat template (identical to training) -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        m_cfg["name_or_path"], cache_dir=m_cfg["cache_dir"], trust_remote_code=m_cfg["trust_remote_code"]
    )
    with open(d_cfg["chat_template_path"], "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, m_cfg["dtype"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading base {m_cfg['name_or_path']} (dtype={m_cfg['dtype']}, attn_implementation={m_cfg['attn_implementation']})")
    base_model = AutoModelForCausalLM.from_pretrained(
        m_cfg["name_or_path"],
        cache_dir=m_cfg["cache_dir"],
        dtype=dtype,
        attn_implementation=m_cfg["attn_implementation"],
        trust_remote_code=m_cfg["trust_remote_code"],
    )
    base_model.to(device)
    base_model.eval()

    # ---- (a) base model generations, all 8 prompts -------------------------------------------
    print("\n[generating] base model (no adapter), 8 prompts...")
    base_outputs = []
    for i, p in enumerate(prompts):
        text = generate_one(base_model, tokenizer, system_text, p["context"], seed, args.max_new_tokens, device)
        base_outputs.append(text)
        print(f"  [{i+1}/8] base done ({len(text)} chars)")

    # ---- (b) base + selected B1 adapter, all 8 prompts ----------------------------------------
    print(f"\n[generating] B1 adapter ({checkpoint_name}), 8 prompts...")
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    peft_model.to(device)
    peft_model.eval()
    b1_outputs = []
    for i, p in enumerate(prompts):
        text = generate_one(peft_model, tokenizer, system_text, p["context"], seed, args.max_new_tokens, device)
        b1_outputs.append(text)
        print(f"  [{i+1}/8] B1 done ({len(text)} chars)")

    # ---- Write / append -------------------------------------------------------------------------
    out_path = os.path.join(output_dir, "sanity_generations.txt")
    file_exists = os.path.isfile(out_path)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    lines = []
    lines.append("=" * 100)
    lines.append(f"GATE 2 -- B1 GENERATION SANITY CHECK -- run at {timestamp}")
    lines.append(f"seed={seed}  config={args.config}  checkpoint={checkpoint_path} ({selection_note})")
    lines.append(
        "NOTE: CounselChat has no validation split (train-only; every question text was seen in "
        "training), so it cannot supply held-out single-turn prompts. All 8 prompts below are drawn "
        "from the ESConv VALIDATION split instead: prompts 1-4 are genuine multi-turn ESConv "
        "continuations (context up to a mid-dialogue user turn); prompts 5-8 are the SAME kind of "
        "ESConv validation dialogue but truncated to only the opening user message, to emulate the "
        "CounselChat single-turn-question style. This is a real substitution, not a like-for-like "
        "CounselChat sample."
    )
    lines.append(
        f"Generation params: temperature=0.7, do_sample=True, max_new_tokens={args.max_new_tokens}, "
        f"seed reset to {seed} immediately before every individual generate() call (both conditions "
        "share identical RNG state per prompt)."
    )
    lines.append("=" * 100)

    for i, p in enumerate(prompts):
        lines.append("")
        lines.append(f"--- PROMPT {i+1}/8  [{p['style']}]  (ESConv validation dialogue_idx={p['dialogue_idx']}) ---")
        lines.append(f"  system: {system_text}")
        lines.append(format_context(p["context"]))
        lines.append("")
        lines.append(f"  [BASE Qwen2.5-7B-Instruct, no adapter]:")
        lines.append(f"    {base_outputs[i]}")
        lines.append("")
        lines.append(f"  [B1 SFT-LoRA ({checkpoint_name})]:")
        lines.append(f"    {b1_outputs[i]}")

    lines.append("")
    mode = "a" if file_exists else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        if file_exists:
            f.write("\n\n")
        f.write("\n".join(lines))

    print(f"\n[{'appended to' if file_exists else 'written'}] {out_path}")


if __name__ == "__main__":
    main()
