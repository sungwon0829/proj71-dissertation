"""
Project 71 -- Gate 1: checkpoint selection for B1 via held-out validation loss.

Feeds: no paper table directly, but this decides WHICH LoRA checkpoint becomes
"the B1 checkpoint" that B2 and T are both built on top of -- so it gates every
downstream number in Table 1/2.

Data: ESConv VALIDATION split (195 dialogues), processed through the exact same
pipeline as training -- reuses process_esconv_validation() / read_system_prompt() /
add_system_prompt() from scripts/prepare_sft.py rather than duplicating that logic.
Never used for training; read-only here.

Masking: identical mechanism to training -- the verified chat template at
configs/qwen25_chat_template_generation.jinja (assistant content + <|im_end|> +
trailing "\n" trained/scored; everything else -100), computed via
return_assistant_tokens_mask=True exactly as scripts/train_sft.py does.

Metric: token-weighted mean assistant-only cross-entropy over the ENTIRE validation
set (sum of per-token NLL over all assistant tokens across all 195 dialogues, divided
by the total assistant-token count) -- not a macro-average of per-example means, so
long and short dialogues are weighted by their actual number of scored tokens.

Selection rule (fixed in advance, per the coordinator's instruction):
    if val_loss(checkpoint-435) > val_loss(checkpoint-290): select checkpoint-290
    else: select checkpoint-435
checkpoint-145 is also reported, for the full curve, but is never selected by this rule.

--no_adapter mode: scores BASE Qwen2.5-7B-Instruct with NO LoRA adapter attached, on the
exact same 195 validation dialogues, same chat template, same masking, same
token-weighting -- for a directly comparable SFT-effect-size number against the three
checkpoints above. This mode skips the checkpoint loop and the selection rule entirely
and writes to a separate file (val_loss_base_seed<seed>.json) so it never overwrites
Gate 1's val_loss_seed<seed>.json.

Usage:
    python scripts\\eval_val_loss.py --config configs\\sft_lora.yaml --seed 42
    python scripts\\eval_val_loss.py --config configs\\sft_lora.yaml --seed 42 --no_adapter
"""

import argparse
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
from prepare_sft import process_esconv_validation, read_system_prompt, add_system_prompt


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


def tokenize_with_assistant_mask(tokenizer, messages, max_seq_length: int, row_ctx: str):
    """Tokenize one conversation with the verified generation-marker template and
    return (input_ids [1D long tensor], assistant_mask [1D long tensor, 1=scored]).
    Fails loudly if the conversation would exceed max_seq_length (zero-truncation
    policy, same as training)."""
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = out["input_ids"]
    assistant_mask = out["assistant_masks"]
    n = len(input_ids)
    if n > max_seq_length:
        raise RuntimeError(
            f"ZERO-TRUNCATION ASSERT FAILED (validation, {row_ctx}): {n} tokens > "
            f"max_seq_length={max_seq_length}. Refusing to silently truncate eval data."
        )
    if not any(assistant_mask):
        raise RuntimeError(
            f"MASKING FAILURE (validation, {row_ctx}): no assistant-masked tokens found "
            "-- refusing to score a dialogue with zero trainable/scorable tokens."
        )
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(assistant_mask, dtype=torch.long),
    )


@torch.no_grad()
def token_weighted_mean_ce(model, tokenizer, records, max_seq_length: int, device: str) -> dict:
    """Return {"mean_ce": float, "total_tokens": int, "n_examples": int} -- the
    token-weighted mean assistant-only cross-entropy over every record in `records`.

    Relies on the model's own internal labels-based loss (HF's CrossEntropyLoss with
    ignore_index=-100, mean reduction over non-ignored positions, standard shift-by-one
    causal-LM alignment) for the per-example mean, rather than reimplementing the shift
    by hand -- then de-averages by that example's scored-token count and re-aggregates
    as a running sum, to get a token-weighted mean across the whole validation set
    (long and short dialogues weighted by their actual number of scored tokens, not
    treated as equal-weight examples).
    """
    model.eval()
    total_nll_sum = 0.0
    total_tokens = 0

    for i, rec in enumerate(records):
        input_ids, assistant_mask = tokenize_with_assistant_mask(
            tokenizer, rec["messages"], max_seq_length, row_ctx=f"row {i}"
        )
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[0, assistant_mask == 0] = -100

        # Number of tokens the model's internal loss will actually average over: the
        # shifted labels (labels[:, 1:]) that are not -100. Position 0 can never be a
        # prediction target (there is no preceding token), so it is excluded either way.
        n_scored = int((labels[0, 1:] != -100).sum().item())
        if n_scored == 0:
            raise RuntimeError(f"Row {i}: 0 scored tokens after shift -- refusing to silently skip.")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        per_example_mean_ce = float(out.loss.item())
        total_nll_sum += per_example_mean_ce * n_scored
        total_tokens += n_scored

    if total_tokens == 0:
        raise RuntimeError("0 total scored tokens across the entire validation set -- fail loudly.")

    return {
        "mean_ce": total_nll_sum / total_tokens,
        "total_tokens": total_tokens,
        "n_examples": len(records),
    }


def main():
    parser = argparse.ArgumentParser(description="Project 71 Gate 1: B1 checkpoint selection via val loss")
    parser.add_argument("--config", type=str, default="configs/sft_lora.yaml", help="Path to YAML config (reuses model/data/training fields from B1's SFT config)")
    parser.add_argument("--seed", type=int, required=True, help="Seed identifying the trained B1 run (results/B1_sft_seed<seed>)")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=["checkpoint-145", "checkpoint-290", "checkpoint-435"],
        help="Checkpoint subdirectory names under the B1 output dir to evaluate",
    )
    parser.add_argument("--system_prompt_file", type=str, default="configs/system_prompt.txt")
    parser.add_argument(
        "--no_adapter",
        action="store_true",
        help="Score BASE Qwen2.5-7B-Instruct with no LoRA adapter attached (skips the "
        "checkpoint loop and selection rule; writes val_loss_base_seed<seed>.json instead).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the config's output_dir_template result (e.g. to evaluate "
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
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"B1 output dir not found: {output_dir} -- train B1 first.")

    max_seq_length = t_cfg["max_seq_length"]

    print(f"[config] {args.config}")
    print(f"[seed] {seed}")
    print(f"[B1 output_dir] {output_dir}")

    # ---- Validation data: exact same pipeline as training, reused not duplicated -------------
    val_records, val_report = process_esconv_validation()
    print(f"[esconv validation] {val_report}")

    system_text = read_system_prompt(Path(args.system_prompt_file))
    val_records = add_system_prompt(val_records, system_text)
    print(f"[system prompt] loaded from {args.system_prompt_file} ({len(system_text)} chars), prepended to all {len(val_records)} validation dialogues")

    # ---- Tokenizer + verified chat template (identical to training) --------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        m_cfg["name_or_path"], cache_dir=m_cfg["cache_dir"], trust_remote_code=m_cfg["trust_remote_code"]
    )
    chat_template_path = d_cfg["chat_template_path"]
    with open(chat_template_path, "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Base model (bf16, sdpa) ---------------------------------------------------------------
    dtype = getattr(torch, m_cfg["dtype"])
    print(f"[model] loading {m_cfg['name_or_path']} from {m_cfg['cache_dir']} (dtype={m_cfg['dtype']}, attn_implementation={m_cfg['attn_implementation']})")
    base_model = AutoModelForCausalLM.from_pretrained(
        m_cfg["name_or_path"],
        cache_dir=m_cfg["cache_dir"],
        dtype=dtype,
        attn_implementation=m_cfg["attn_implementation"],
        trust_remote_code=m_cfg["trust_remote_code"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)

    # ---- --no_adapter mode: score BASE model only, no checkpoint loop, no selection rule -------
    if args.no_adapter:
        print("\n[eval] BASE Qwen2.5-7B-Instruct, NO ADAPTER")
        stats = token_weighted_mean_ce(base_model, tokenizer, val_records, max_seq_length, device)
        print(f"  mean_ce={stats['mean_ce']:.4f}  total_scored_tokens={stats['total_tokens']}  n_examples={stats['n_examples']}")

        out_path = os.path.join(output_dir, f"val_loss_base_seed{seed}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": seed,
                    "output_dir": output_dir,
                    "n_validation_dialogues": len(val_records),
                    "model": "base_no_adapter",
                    "stats": stats,
                },
                f,
                indent=2,
            )
        print(f"[written] {out_path}")
        return

    # ---- Evaluate each checkpoint --------------------------------------------------------------
    results = {}
    for ckpt_name in args.checkpoints:
        ckpt_path = os.path.join(output_dir, ckpt_name)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"\n[eval] {ckpt_name} ({ckpt_path})")
        peft_model = PeftModel.from_pretrained(base_model, ckpt_path)
        peft_model.to(device)
        stats = token_weighted_mean_ce(peft_model, tokenizer, val_records, max_seq_length, device)
        results[ckpt_name] = stats
        print(f"  mean_ce={stats['mean_ce']:.4f}  total_scored_tokens={stats['total_tokens']}  n_examples={stats['n_examples']}")
        base_model = peft_model.unload()  # detach adapter, get bare base model back for the next checkpoint

    # ---- Selection rule (fixed in advance) ------------------------------------------------------
    def loss_of(name):
        return results[name]["mean_ce"]

    print("\n" + "=" * 88)
    print("VALIDATION LOSS CURVE (token-weighted mean assistant-only cross-entropy)")
    print("=" * 88)
    for ckpt_name in args.checkpoints:
        print(f"  {ckpt_name:>16s}: {loss_of(ckpt_name):.4f}")

    if "checkpoint-435" not in results or "checkpoint-290" not in results:
        raise RuntimeError("Selection rule requires both checkpoint-290 and checkpoint-435 to be evaluated.")

    if loss_of("checkpoint-435") > loss_of("checkpoint-290"):
        selected = "checkpoint-290"
        reason = (
            f"val loss at checkpoint-435 ({loss_of('checkpoint-435'):.4f}) is HIGHER than at "
            f"checkpoint-290 ({loss_of('checkpoint-290'):.4f}) -- selecting checkpoint-290 per the fixed rule."
        )
    else:
        selected = "checkpoint-435"
        reason = (
            f"val loss at checkpoint-435 ({loss_of('checkpoint-435'):.4f}) is NOT higher than at "
            f"checkpoint-290 ({loss_of('checkpoint-290'):.4f}) -- selecting checkpoint-435 per the fixed rule."
        )

    print(f"\n[SELECTED CHECKPOINT] {selected}")
    print(f"[reason] {reason}")

    out_path = os.path.join(output_dir, f"val_loss_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "output_dir": output_dir,
                "n_validation_dialogues": len(val_records),
                "checkpoints": results,
                "selected_checkpoint": selected,
                "selection_reason": reason,
            },
            f,
            indent=2,
        )
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
