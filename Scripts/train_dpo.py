"""
Project 71 -- DPO training script, shared by B2 (helpfulness-only) and T (helpfulness +
safety). Which arm runs is entirely determined by --config (configs/dpo_b2.yaml or
configs/dpo_t.yaml); this script contains no arm-specific branching.

Produces: the B2 row (config: dpo_b2.yaml) or T row (config: dpo_t.yaml) in Table 1, and
(via T) the ASR-per-category numbers in Table 2. B3 = B2 + guardrail filter at inference,
so this script also produces the B3 baseline's underlying model.

DRAFT ONLY as of the commit that introduces this file -- do not launch until the
coordinator confirms the PROPOSED (not yet owner-confirmed) hyperparameters flagged in
configs/dpo_b2.yaml / configs/dpo_t.yaml (learning_rate, num_train_epochs, beta, loss_type,
precompute_ref_log_probs).

Continuation semantics (the correctness-critical part): B2/T are "B1 + DPO", not "base +
fresh-DPO-LoRA". The B1 LoRA checkpoint is loaded via
`PeftModel.from_pretrained(base_model, b1_checkpoint_path, is_trainable=True)` --
`is_trainable` defaults to False in the installed peft (0.19.1); omitting it would silently
load a frozen adapter and DPO would train nothing. No `peft_config` is passed to
DPOTrainer (passing one alongside an already-a-PeftModel model raises in the installed trl
1.9.0 source -- verified by reading dpo_trainer.py, not assumed). With `peft_config=None`,
`ref_model=None`, and the model already a PeftModel with a pretrained "default" adapter,
DPOTrainer's own __init__ clones the current adapter weights into a second, frozen "ref"
adapter within the SAME PeftModel (verified in the installed trl 1.9.0 source,
dpo_trainer.py ~line 649-670) -- so the DPO reference policy is exactly B1's trained
distribution, and the "default" adapter (still B1's weights, now trainable) is what DPO
updates. This is the correct, TRL-native mechanism for LoRA-continuation DPO; it is not a
custom hack.

LoRA adapter config is asserted (not just documented) to be byte-identical to
configs/sft_lora.yaml's `lora:` block at startup -- the fixed-template requirement is a
hard gate, not a convention.

assert_never_redteam() is reused (imported, not duplicated) from scripts/train_sft.py.

Usage (NOT to be run until the coordinator confirms the proposed hyperparameters):
    python scripts\\train_dpo.py --config configs\\dpo_b2.yaml --seed 42
    python scripts\\train_dpo.py --config configs\\dpo_t.yaml --seed 42
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pref import build_helpful_pairs
from prepare_sft import read_system_prompt
from train_sft import assert_never_redteam  # reused, not duplicated (per Task 2 instruction)


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


def assert_lora_matches_b1_template(this_lora_cfg: dict, sft_config_path: str = "configs/sft_lora.yaml") -> None:
    """The LoRA adapter config is a hard, asserted fixed template across B1/B2/T -- not a
    convention that can silently drift. Compares every field, including target_modules as
    an order-independent set, against configs/sft_lora.yaml's `lora:` block."""
    if not os.path.isfile(sft_config_path):
        raise FileNotFoundError(
            f"Cannot verify the fixed LoRA template: {sft_config_path} not found. Refusing to proceed."
        )
    with open(sft_config_path, "r", encoding="utf-8") as f:
        b1_lora_cfg = yaml.safe_load(f)["lora"]

    mismatches = []
    for key in ("r", "alpha", "dropout", "bias", "task_type"):
        if this_lora_cfg.get(key) != b1_lora_cfg.get(key):
            mismatches.append(f"{key}: this={this_lora_cfg.get(key)!r} vs B1={b1_lora_cfg.get(key)!r}")
    if set(this_lora_cfg.get("target_modules", [])) != set(b1_lora_cfg.get("target_modules", [])):
        mismatches.append(
            f"target_modules: this={sorted(this_lora_cfg.get('target_modules', []))} vs "
            f"B1={sorted(b1_lora_cfg.get('target_modules', []))}"
        )
    if mismatches:
        raise RuntimeError(
            "LORA FIXED-TEMPLATE ASSERT FAILED: this config's `lora:` block does not match "
            f"{sft_config_path}'s (the B1/fixed template). Mismatches: {'; '.join(mismatches)}. "
            "Per CLAUDE.md's non-negotiables, an adapter hyperparameter change must be applied "
            "identically to B1, B2 and T, and stated explicitly in notebook/lab_notebook.md -- "
            "it cannot drift silently between configs. Refusing to proceed."
        )
    print("[lora fixed-template assert] PASSED: this config's lora: block matches configs/sft_lora.yaml exactly.")


def pair_id(pair: dict) -> str:
    """Stable content-hash identifier for a preference pair, independent of file order --
    used so the sampled pair ids logged per run are reproducible from (source data + seed +
    exclusion policy) alone, per the coordinator's Task 2 requirement."""
    blob = json.dumps(pair, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def tokenized_len(tokenizer, messages: list) -> int:
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
    return len(out["input_ids"])


def filter_by_max_length(pairs: list, tokenizer, max_length: int, pool_name: str) -> tuple:
    """EXCLUDE (never truncate) any pair whose tokenize(prompt+chosen) or
    tokenize(prompt+rejected) exceeds max_length. See configs/dpo_*.yaml `max_length_policy`
    for the reasoning: truncating a completion would corrupt the DPO chosen-vs-rejected
    comparison, so this project excludes rather than lets DPOConfig silently truncate.
    Returns (kept_pairs, n_excluded)."""
    kept = []
    n_excluded = 0
    for p in pairs:
        len_chosen = tokenized_len(tokenizer, p["prompt"] + p["chosen"])
        len_rejected = tokenized_len(tokenizer, p["prompt"] + p["rejected"])
        if len_chosen > max_length or len_rejected > max_length:
            n_excluded += 1
            continue
        kept.append(p)
    print(f"[max-length filter, {pool_name}] max_length={max_length}: {n_excluded} excluded, {len(kept)} kept "
          f"(of {len(pairs)} input)")
    return kept, n_excluded


def load_jsonl(path: str) -> list:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Loaded 0 rows from {path} -- fail loudly, not silently.")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Project 71 DPO training (shared by B2 and T)")
    parser.add_argument("--config", type=str, required=True, help="configs/dpo_b2.yaml or configs/dpo_t.yaml")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    set_seed_everywhere(seed)

    m_cfg = cfg["model"]
    ba_cfg = cfg["base_adapter"]
    d_cfg = cfg["data"]
    l_cfg = cfg["lora"]
    t_cfg = cfg["training"]

    arm_name = "T" if d_cfg["n_safety_sample"] > 0 else "B2"
    output_dir = t_cfg["output_dir_template"].format(seed=seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[config] {args.config}")
    print(f"[arm] {arm_name}")
    print(f"[seed] {seed}")
    print(f"[output_dir] {output_dir}")

    # ---- Hard gates -----------------------------------------------------------------------
    assert_never_redteam(d_cfg["safety_pairs_file"])
    assert_lora_matches_b1_template(l_cfg)

    # ---- Tokenizer + verified chat template (same one used for B1) -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        m_cfg["name_or_path"], cache_dir=m_cfg["cache_dir"], trust_remote_code=m_cfg["trust_remote_code"]
    )
    with open(d_cfg["chat_template_path"], "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = read_system_prompt(Path(d_cfg["system_prompt_file"]))

    # ---- Helpfulness pool: built in-memory from the raw dataset, never reading/overwriting
    #      data/processed/pref_helpful.jsonl (see prepare_pref.build_helpful_pairs docstring) --
    helpful_pool, helpful_funnel = build_helpful_pairs(
        seed=seed, system_prompt=system_prompt, exclude_safety_inversions=d_cfg["exclude_safety_inversions"]
    )
    print(f"[helpful pool] {helpful_funnel}")
    helpful_pool, n_excl_length_helpful = filter_by_max_length(
        helpful_pool, tokenizer, d_cfg["max_length"], "helpful"
    )

    n_helpful_needed = d_cfg["n_helpful_sample"]
    if len(helpful_pool) < n_helpful_needed:
        raise RuntimeError(
            f"Helpfulness pool after exclusion+length-filter has only {len(helpful_pool)} rows, "
            f"need {n_helpful_needed}. Fail loudly rather than silently sampling with replacement "
            "or shrinking N."
        )
    rng = random.Random(seed)
    sampled_helpful = rng.sample(helpful_pool, n_helpful_needed)
    print(f"[sampled helpful] n={len(sampled_helpful)} (seed={seed}, from pool of {len(helpful_pool)})")

    # ---- Safety pool (T only) ----------------------------------------------------------------
    n_safety_needed = d_cfg["n_safety_sample"]
    sampled_safety = []
    n_excl_length_safety = 0
    if n_safety_needed > 0:
        safety_pool = load_jsonl(d_cfg["safety_pairs_file"])
        print(f"[safety pool] loaded {len(safety_pool)} rows from {d_cfg['safety_pairs_file']}")
        safety_pool, n_excl_length_safety = filter_by_max_length(
            safety_pool, tokenizer, d_cfg["max_length"], "safety"
        )
        if len(safety_pool) < n_safety_needed:
            raise RuntimeError(
                f"Safety pool after length-filter has only {len(safety_pool)} rows, need {n_safety_needed}."
            )
        if n_safety_needed == len(safety_pool):
            sampled_safety = list(safety_pool)  # use all, still copy for a stable list
        else:
            sampled_safety = rng.sample(safety_pool, n_safety_needed)
        print(f"[sampled safety] n={len(sampled_safety)} (seed={seed}, from pool of {len(safety_pool)})")

    # ---- Matched-volume assertion (Methodological Safeguards v2, rule 2) ---------------------
    total_pairs = len(sampled_helpful) + len(sampled_safety)
    print(f"[matched volume] helpful={len(sampled_helpful)} safety={len(sampled_safety)} total={total_pairs}")

    # ---- Manifest: log both counts + sampled pair ids, so a seed is reproducible from its
    #      config alone (Task 2 requirement) -----------------------------------------------
    manifest = {
        "arm": arm_name,
        "config": args.config,
        "seed": seed,
        "n_helpful_sampled": len(sampled_helpful),
        "n_safety_sampled": len(sampled_safety),
        "total_pairs": total_pairs,
        "helpful_funnel": helpful_funnel,
        "n_excluded_helpful_by_max_length": n_excl_length_helpful,
        "n_excluded_safety_by_max_length": n_excl_length_safety,
        "max_length": d_cfg["max_length"],
        "sampled_helpful_pair_ids": [pair_id(p) for p in sampled_helpful],
        "sampled_safety_pair_ids": [pair_id(p) for p in sampled_safety],
    }
    manifest_path = os.path.join(output_dir, "dpo_data_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[written] {manifest_path}")

    train_pairs = sampled_helpful + sampled_safety
    train_dataset = Dataset.from_list(train_pairs)

    # ---- Model: base + B1's TRAINED LoRA adapter, loaded is_trainable=True so DPO continues
    #      from B1's weights rather than a frozen copy or a fresh adapter -----------------------
    dtype = getattr(torch, m_cfg["dtype"])
    print(f"[model] loading base {m_cfg['name_or_path']} (dtype={m_cfg['dtype']}, "
          f"attn_implementation={m_cfg['attn_implementation']})")
    base_model = AutoModelForCausalLM.from_pretrained(
        m_cfg["name_or_path"],
        cache_dir=m_cfg["cache_dir"],
        dtype=dtype,
        attn_implementation=m_cfg["attn_implementation"],
        trust_remote_code=m_cfg["trust_remote_code"],
    )
    b1_checkpoint_path = os.path.join(ba_cfg["b1_output_dir"], ba_cfg["b1_checkpoint"])
    if not os.path.isdir(b1_checkpoint_path):
        raise FileNotFoundError(f"B1 checkpoint not found: {b1_checkpoint_path}")
    print(f"[model] loading B1 adapter from {b1_checkpoint_path} (is_trainable=True -- continuation, not a fresh adapter)")
    model = PeftModel.from_pretrained(base_model, b1_checkpoint_path, is_trainable=True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise RuntimeError(
            "0 trainable parameters after loading the B1 adapter with is_trainable=True -- "
            "DPO would train nothing. Refusing to proceed."
        )
    print(f"[trainable params after loading B1 adapter] {n_trainable:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        max_length=d_cfg["max_length"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        optim=t_cfg["optim"],
        weight_decay=t_cfg["weight_decay"],
        max_grad_norm=t_cfg["max_grad_norm"],
        beta=t_cfg["beta"],
        loss_type=t_cfg["loss_type"],
        label_smoothing=t_cfg["label_smoothing"],
        disable_dropout=t_cfg["disable_dropout"],
        precompute_ref_log_probs=t_cfg["precompute_ref_log_probs"],
        precompute_ref_batch_size=t_cfg["precompute_ref_batch_size"],
        seed=seed,
        data_seed=seed,
        bf16=t_cfg["bf16"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs=t_cfg.get("gradient_checkpointing_kwargs"),
        dataloader_num_workers=t_cfg["dataloader_num_workers"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        report_to=t_cfg["report_to"],
        train_sampling_strategy=t_cfg.get("train_sampling_strategy", "random"),
    )

    # model is already a PeftModel with a pretrained "default" adapter; do NOT pass
    # peft_config (would raise -- verified in the installed trl 1.9.0 source) and leave
    # ref_model=None so DPOTrainer clones the current "default" adapter into a frozen "ref"
    # adapter (the reference policy = B1's trained distribution) -- see module docstring.
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print(f"[max_length_policy] {d_cfg['max_length_policy']} (exclude, not truncate -- see config comments)")
    print("\n*** DO NOT LAUNCH: this script is a draft pending the coordinator's confirmation of "
          "proposed hyperparameters (learning_rate, num_train_epochs, beta, loss_type, "
          "precompute_ref_log_probs). trainer.train() is intentionally not being called by "
          "any automated harness until that confirmation is recorded in notebook/lab_notebook.md. ***\n")

    # trainer.train()  # INTENTIONALLY COMMENTED OUT -- draft config, not yet launched.
    # trainer.save_model(output_dir)
    # model.save_pretrained(output_dir, selected_adapters=["default"])  # save ONLY the
    #     trained adapter, not the frozen "ref" clone TRL added alongside it.
    # tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
