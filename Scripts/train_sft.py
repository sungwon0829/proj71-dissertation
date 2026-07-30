"""
Project 71 -- B1 SFT training script.

LoRA fine-tune of Qwen2.5-7B-Instruct on data/processed/sft_train.jsonl.
This LoRA adapter config is the FIXED TEMPLATE for arms B1, B2, and T --
if you must change an adapter hyperparameter here, it must change for every
arm, and that must be recorded explicitly in notebook/lab_notebook.md.

All hyperparameters live in the YAML config passed via --config. This script
takes only --config and --seed (plus an optional --max_steps override used
for the smoke check). Never hardcode a hyperparameter here.

Usage:
    python scripts\\train_sft.py --config configs\\sft_lora.yaml --seed 42
    python scripts\\train_sft.py --config configs\\sft_lora.yaml --seed 42 --max_steps 20   # smoke check
"""

import argparse
import os
import random
import sys

# Must be set before importing transformers/datasets/huggingface_hub, and the model
# is already cached locally -- never allow an accidental network fetch.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def assert_never_redteam(train_file: str) -> None:
    norm = os.path.normpath(train_file).replace("\\", "/").lower()
    if "redteam" in norm or "data/redteam" in norm:
        raise RuntimeError(
            f"REFUSING TO TRAIN: '{train_file}' looks like it is under data/redteam/, "
            "which is the frozen held-out evaluation suite and must never be trained on."
        )


def assert_zero_truncation(tokenizer, examples: list, max_seq_length: int) -> tuple[int, list]:
    """Tokenize every example with the chat template and fail hard if any would truncate.

    Returns (max_observed token length, per-example token length list) for reporting and,
    optionally, worst-case-ordering smoke checks.
    """
    max_observed = 0
    offenders = []
    lengths = []
    for i, ex in enumerate(examples):
        messages = ex["messages"]
        # transformers 5.x: apply_chat_template returns a dict, not a bare tensor/list --
        # must pass return_dict=True and index into ['input_ids'] (see CLAUDE.md).
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True
        )
        n = len(out["input_ids"])
        lengths.append(n)
        max_observed = max(max_observed, n)
        if n > max_seq_length:
            offenders.append((i, n))
    if offenders:
        detail = ", ".join(f"row {i} ({n} tok)" for i, n in offenders[:20])
        raise RuntimeError(
            f"ZERO-TRUNCATION ASSERT FAILED: {len(offenders)} example(s) exceed "
            f"max_seq_length={max_seq_length}. First offenders: {detail}. "
            "Refusing to train -- silently truncating would train on incomplete assistant turns "
            "and corrupt the EOS-in-loss guarantee."
        )
    print(
        f"[zero-truncation assert] PASSED: {len(examples)} examples, "
        f"max observed length = {max_observed} tok (limit {max_seq_length})."
    )
    return max_observed, lengths


def dump_masking_verification(tokenizer, batch: dict, out_path: str, max_examples: int = 1) -> str:
    """Take one real collated batch from the trainer's dataloader, decode token-by-token,
    and produce a readable span-level dump of MASKED vs TRAINED tokens. Returns the dump text.
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    n = min(max_examples, input_ids.shape[0])
    lines = []
    for row in range(n):
        ids_row = input_ids[row].tolist()
        lab_row = labels[row].tolist()
        lines.append(f"=== Example {row} ({len(ids_row)} tokens) ===")
        # Build span-level summary: consecutive runs of MASKED (-100) vs TRAINED.
        spans = []
        cur_state = None
        cur_tokens = []
        for tid, lab in zip(ids_row, lab_row):
            state = "MASKED" if lab == -100 else "TRAINED"
            if state != cur_state:
                if cur_state is not None:
                    spans.append((cur_state, cur_tokens))
                cur_state = state
                cur_tokens = [tid]
            else:
                cur_tokens.append(tid)
        if cur_state is not None:
            spans.append((cur_state, cur_tokens))

        for state, toks in spans:
            text = tokenizer.decode(toks)
            preview = text.replace("\n", "\\n")
            if len(preview) > 160:
                preview = preview[:157] + "..."
            lines.append(f"[{state} {len(toks):4d} tok]: {preview!r}")
        lines.append("")

    dump = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(dump)
    return dump


def main():
    parser = argparse.ArgumentParser(description="Project 71 B1 SFT (LoRA) training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max optimizer steps (for smoke checks only). Leave unset for a full run.",
    )
    parser.add_argument(
        "--worst_case_smoke",
        action="store_true",
        help=(
            "Smoke-check only: reorder training examples by descending tokenized length so the "
            "first batches contain the longest sequences (worst case for activation memory), "
            "instead of the file's normal (already-shuffled-at-data-prep-time) order. Never use "
            "this for a real launch -- it changes example order and is only meant to stress-test "
            "peak VRAM before committing to a multi-hour run."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    set_seed_everywhere(seed)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    l_cfg = cfg["lora"]
    t_cfg = cfg["training"]

    train_file = d_cfg["train_file"]
    assert_never_redteam(train_file)
    if not os.path.isfile(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file} (fail loudly, not silently)")

    output_dir = t_cfg["output_dir_template"].format(seed=seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[config] {args.config}")
    print(f"[seed] {seed}")
    print(f"[output_dir] {output_dir}")
    print(f"[train_file] {train_file}")

    # ---- Tokenizer + verified chat template -------------------------------------------------
    dtype = getattr(torch, m_cfg["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(
        m_cfg["name_or_path"], cache_dir=m_cfg["cache_dir"], trust_remote_code=m_cfg["trust_remote_code"]
    )
    chat_template_path = d_cfg["chat_template_path"]
    if not os.path.isfile(chat_template_path):
        raise FileNotFoundError(f"Chat template not found: {chat_template_path}")
    with open(chat_template_path, "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load + validate dataset --------------------------------------------------------------
    raw = load_dataset("json", data_files=train_file, split="train")
    examples = list(raw)
    if len(examples) == 0:
        raise RuntimeError(f"Loaded 0 examples from {train_file} -- fail loudly, not silently.")
    for i, ex in enumerate(examples):
        if "messages" not in ex or not isinstance(ex["messages"], list) or len(ex["messages"]) == 0:
            raise RuntimeError(f"Row {i} in {train_file} is malformed (missing/empty 'messages'). Fail loudly.")

    max_seq_length = t_cfg["max_seq_length"]
    max_observed, lengths = assert_zero_truncation(tokenizer, examples, max_seq_length)

    if args.worst_case_smoke:
        if args.max_steps is None:
            raise RuntimeError(
                "--worst_case_smoke changes example order and must only be used together with "
                "--max_steps (a smoke check), never for a real launch. Refusing to proceed."
            )
        order = sorted(range(len(examples)), key=lambda i: lengths[i], reverse=True)
        examples = [examples[i] for i in order]
        raw = raw.select(order)
        print(
            f"[worst_case_smoke] reordered {len(examples)} examples by descending length; "
            f"first batch will contain the {t_cfg['per_device_train_batch_size']} longest "
            f"examples (top length = {lengths[order[0]]} tok)."
        )

    # ---- Model --------------------------------------------------------------------------------
    print(f"[model] loading {m_cfg['name_or_path']} from {m_cfg['cache_dir']} (dtype={m_cfg['dtype']}, "
          f"attn_implementation={m_cfg['attn_implementation']})")
    model = AutoModelForCausalLM.from_pretrained(
        m_cfg["name_or_path"],
        cache_dir=m_cfg["cache_dir"],
        dtype=dtype,
        attn_implementation=m_cfg["attn_implementation"],
        trust_remote_code=m_cfg["trust_remote_code"],
    )

    lora_config = LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["alpha"],
        lora_dropout=l_cfg["dropout"],
        target_modules=l_cfg["target_modules"],
        bias=l_cfg["bias"],
        task_type=l_cfg["task_type"],
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        optim=t_cfg["optim"],
        weight_decay=t_cfg["weight_decay"],
        max_grad_norm=t_cfg["max_grad_norm"],
        seed=seed,
        data_seed=seed,
        bf16=t_cfg["bf16"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs=t_cfg.get("gradient_checkpointing_kwargs"),
        assistant_only_loss=t_cfg["assistant_only_loss"],
        packing=t_cfg["packing"],
        dataloader_num_workers=t_cfg["dataloader_num_workers"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        report_to=t_cfg["report_to"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        # --worst_case_smoke forces sequential sampling (over the descending-length-sorted
        # dataset above) so the first batches are a genuine worst case for activation memory.
        # The real run always uses the yaml's own value (default "random", TRL's own default).
        train_sampling_strategy=("sequential" if args.worst_case_smoke else t_cfg.get("train_sampling_strategy", "random")),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=raw,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ---- B. Empirical masking verification (critical gate) ------------------------------------
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    verify_path = os.path.join(output_dir, "masking_verification.txt")
    dump = dump_masking_verification(tokenizer, batch, verify_path, max_examples=1)
    print("\n" + "=" * 88)
    print("ASSISTANT-ONLY LOSS -- EMPIRICAL MASKING VERIFICATION (one real collated batch)")
    print("=" * 88)
    print(dump)
    print("=" * 88)
    print(f"[masking verification dump written to] {verify_path}\n")

    # Hard gate: at least one TRAINED span must exist, and the FIRST span must be MASKED
    # (system/user prefix), otherwise abort before spending any compute.
    labels_row0 = batch["labels"][0].tolist()
    has_trained = any(l != -100 for l in labels_row0)
    first_is_masked = labels_row0[0] == -100
    if not has_trained:
        raise RuntimeError(
            "MASKING VERIFICATION FAILED: no TRAINED tokens found in the first batch. "
            "assistant_only_loss is masking everything. STOPPING -- do not launch training."
        )
    if not first_is_masked:
        raise RuntimeError(
            "MASKING VERIFICATION FAILED: the first token of the batch is TRAINED, not MASKED. "
            "This means the system/user prefix is being trained on. STOPPING -- do not launch training."
        )
    print("[masking gate] PASSED: TRAINED spans present; leading system/user span is MASKED.\n")

    # ---- C/D. Train -----------------------------------------------------------------------------
    print(f"[max_observed_seq_len] {max_observed}")
    print(f"[trainable params]")
    trainer.model.print_trainable_parameters()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    import time as _time
    t0 = _time.time()
    result = trainer.train()
    wall = _time.time() - t0
    print(f"[train result] {result}")
    print(f"[wall_clock_sec] {wall:.1f}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[peak_vram_gb] {peak_gb:.2f}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[done] model + tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
