"""
Project 71 -- DPO training script, shared by B2 (helpfulness-only) and T (helpfulness +
safety). Which arm runs is entirely determined by --config (configs/dpo_b2.yaml or
configs/dpo_t.yaml); this script contains no arm-specific branching.

Produces: the B2 row (config: dpo_b2.yaml) or T row (config: dpo_t.yaml) in Table 1, and
(via T) the ASR-per-category numbers in Table 2. B3 = B2 + guardrail filter at inference,
so this script also produces the B3 baseline's underlying model.

Hyperparameters (learning_rate, num_train_epochs, beta, loss_type, precompute_ref_log_probs,
etc.) are CONFIRMED as of 2026-07-31, with one binding requirement: B2 and T must be
byte-identical on every training/model/lora hyperparameter and never differ except in data
composition (n_helpful_sample, n_safety_sample, output_dir_template). This is asserted at
startup (assert_hyperparams_match_sibling), not just documented -- since B2-vs-T is the
whole experiment, any other difference between the two configs is a confound.

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
from copy import deepcopy

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
from prepare_pref import build_helpful_pairs, build_safety_pairs
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


# Keys allowed to differ across dpo_b2.yaml / dpo_t.yaml / dpo_t_ctrl.yaml -- everything
# else in these sections must be byte-identical, or a B2-vs-T-vs-T_ctrl comparison is
# confounded by something other than data composition.
_SIBLING_ALLOWED_TO_DIFFER = {
    ("training", "output_dir_template"),
    ("data", "n_helpful_sample"),
    ("data", "n_safety_sample"),
    ("data", "arm"),
    ("data", "safety_direction"),
}
_SIBLING_DATA_KEYS_MUST_MATCH = [
    "max_length", "max_length_policy", "exclude_safety_inversions",
    "chat_template_path", "system_prompt_file", "helpful_pool_source",
]
_ALL_DPO_CONFIG_BASENAMES = ("dpo_b2.yaml", "dpo_t.yaml", "dpo_t_ctrl.yaml")


def _other_config_paths(this_config_path: str) -> list:
    """Returns the paths of the other two configs in the {B2, T, T_ctrl} trio, in the same
    directory as this_config_path."""
    directory = os.path.dirname(this_config_path)
    this_base = os.path.basename(this_config_path).lower()
    if this_base not in _ALL_DPO_CONFIG_BASENAMES:
        raise RuntimeError(
            f"Cannot determine sibling configs for {this_config_path!r} -- expected one of "
            f"{_ALL_DPO_CONFIG_BASENAMES}. Refusing to skip the hyperparameter-match "
            "assertion silently."
        )
    return [os.path.join(directory, b) for b in _ALL_DPO_CONFIG_BASENAMES if b != this_base]


def assert_hyperparams_match_sibling(this_config_path: str, this_cfg: dict) -> None:
    """B2 vs T (vs T_ctrl) is the whole experiment (CLAUDE.md's ONE CLAIM; T_ctrl added by
    notebook/preregistration.md Revision 5). Any hyperparameter difference between the
    three configs other than data composition is a confound, not a tuning choice -- so
    this is a hard startup gate, exactly like the LoRA fixed-template assert, not a
    documentation convention that can silently drift. Compares this config against BOTH
    other configs in the {B2, T, T_ctrl} trio, so no pairwise drift (e.g. T_ctrl diverging
    from B2 while still matching T) can slip through."""
    other_paths = _other_config_paths(this_config_path)
    all_mismatches = []
    checked = []
    for other_path in other_paths:
        if not os.path.isfile(other_path):
            raise FileNotFoundError(
                f"Cannot verify hyperparameter equality: {other_path} not found. Refusing to "
                "proceed without all three configs present to compare."
            )
        other_cfg = load_config(other_path)

        mismatches = []
        for section in ("model", "base_adapter", "lora", "training"):
            this_section = this_cfg.get(section, {}) or {}
            other_section = other_cfg.get(section, {}) or {}
            for key in set(this_section) | set(other_section):
                if (section, key) in _SIBLING_ALLOWED_TO_DIFFER:
                    continue
                if this_section.get(key) != other_section.get(key):
                    mismatches.append(
                        f"{section}.{key}: this={this_section.get(key)!r} vs {other_path}={other_section.get(key)!r}"
                    )
        this_data = this_cfg.get("data", {}) or {}
        other_data = other_cfg.get("data", {}) or {}
        for key in _SIBLING_DATA_KEYS_MUST_MATCH:
            if this_data.get(key) != other_data.get(key):
                mismatches.append(
                    f"data.{key}: this={this_data.get(key)!r} vs {other_path}={other_data.get(key)!r}"
                )
        all_mismatches.extend(mismatches)
        checked.append(other_path)

    if all_mismatches:
        raise RuntimeError(
            "HYPERPARAMETER-MATCH ASSERT FAILED across the {B2, T, T_ctrl} trio: "
            f"{'; '.join(all_mismatches)}. Per the coordinator's binding requirement, any "
            "hyperparameter difference between these configs (other than arm / "
            "safety_direction / n_helpful_sample / n_safety_sample / output_dir_template) "
            "is a confound. Refusing to proceed."
        )
    print(f"[hyperparameter-match assert] PASSED: {this_config_path} matches {checked} "
          "on every field except data composition.")


def check_gpu_headroom(min_free_gb: float = 20.0) -> None:
    """Operational safety check, not a correctness gate: print current GPU memory state and
    warn/raise if headroom looks thin. B1's full run already OOM'd once on a card that
    looked fine at a glance; this is deliberately loud rather than silent about it. This is
    informational only for concurrent contention -- the actual go/no-go decision (wait and
    re-check vs. launch) is made by the operator using this output, per the coordinator's
    explicit instruction."""
    if not torch.cuda.is_available():
        print("[gpu headroom] CUDA not available -- skipping (CPU run?).")
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    print(f"[gpu headroom] {free_gb:.2f} GB free / {total_gb:.2f} GB total")
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"GPU headroom check FAILED: only {free_gb:.2f} GB free (< {min_free_gb} GB floor). "
            "Refusing to launch into a contended card -- wait and re-check with nvidia-smi."
        )


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


def assert_zero_truncation_dpo(train_pairs: list, tokenizer, max_length: int) -> None:
    """Explicit, redundant final gate (mirrors train_sft.py's zero-truncation assert):
    even though filter_by_max_length() already excluded every over-length row before
    sampling, re-verify independently on the FINAL sampled training set that nothing here
    would be truncated by DPOConfig's max_length -- state it and verify it, don't just
    trust the earlier filter silently."""
    offenders = []
    for i, p in enumerate(train_pairs):
        len_chosen = tokenized_len(tokenizer, p["prompt"] + p["chosen"])
        len_rejected = tokenized_len(tokenizer, p["prompt"] + p["rejected"])
        if len_chosen > max_length or len_rejected > max_length:
            offenders.append((i, len_chosen, len_rejected))
    if offenders:
        detail = ", ".join(f"row {i} (chosen={lc}, rejected={lr})" for i, lc, lr in offenders[:20])
        raise RuntimeError(
            f"ZERO-TRUNCATION ASSERT FAILED (DPO, final sampled set): {len(offenders)} pair(s) "
            f"exceed max_length={max_length}. First offenders: {detail}. This should be "
            "impossible given filter_by_max_length() already ran -- something upstream is "
            "wrong. Refusing to train."
        )
    print(f"[zero-truncation assert, DPO] PASSED: {len(train_pairs)} pairs, none exceed "
          f"max_length={max_length}.")


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


def verify_reference_and_gradients(trainer, model, tokenizer, output_dir: str) -> None:
    """The DPO analogue of B1's empirical assistant-only-loss masking verification.
    assistant-only-loss has no analogue here, but the equivalent question is: is the
    reference policy really B1's FROZEN adapter (not the base model, not a trainable
    copy), and does the trainable ("default") adapter actually receive gradients? This is
    checked directly against real model state and one real collated batch, not assumed
    from `is_trainable=True` alone.

    Static check (no forward pass needed): peft_config must contain both "default" and
    "ref" adapters; at this point (before any optimizer step) their weights must be
    numerically identical (the "ref" adapter was cloned from "default"/B1's checkpoint at
    DPOTrainer construction time) and their requires_grad flags must differ (default=True,
    ref=False) -- this alone is direct proof the reference is B1's own trained weights
    (non-zero LoRA delta, unlike "the base model") and is frozen (unlike a trainable copy).

    Dynamic check (one real batch, real forward+backward through the trainer's own loss):
    manually re-derives the reference log-probs by switching the active adapter to "ref"
    myself (reusing trl's own selective_log_softmax/use_adapter helpers, not reimplementing
    the math) and cross-checks them against DPOTrainer's own cached/computed values: this
    independently confirms the cached reference numbers really did come from the "ref"
    adapter. Then runs the real trainer.compute_loss() (with the "default" adapter active,
    gradients enabled) and calls .backward(), and inspects .grad on both adapters: default
    must have a non-zero-norm gradient, ref's grad must stay None throughout. Gradients are
    zeroed afterwards so this pre-flight check does not contaminate the first real training
    step.

    Writes a readable dump to <output_dir>/dpo_reference_verification.txt and raises if any
    check fails (do not launch training on an unverified reference).
    """
    from trl.trainer.dpo_trainer import selective_log_softmax, use_adapter

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log("=" * 88)
    log("DPO REFERENCE / GRADIENT VERIFICATION (one real batch)")
    log("=" * 88)

    # ---- Static check: adapters present, ref==default at init, requires_grad correct -----
    adapter_names = set(model.peft_config.keys())
    log(f"[adapters present] {sorted(adapter_names)}")
    if "default" not in adapter_names or "ref" not in adapter_names:
        raise RuntimeError(
            f"REFERENCE VERIFICATION FAILED: expected adapters 'default' and 'ref', found {adapter_names}. "
            "This means DPOTrainer did not create the expected frozen reference clone. Do not launch training."
        )

    default_named = {n: p for n, p in model.named_parameters() if ".default." in n}
    ref_named = {n: p for n, p in model.named_parameters() if ".ref." in n}
    if not default_named or not ref_named:
        raise RuntimeError("REFERENCE VERIFICATION FAILED: found no per-adapter named parameters to compare.")

    # Pick one concrete LoRA matrix to report in detail (readable evidence, not just a pass/fail).
    sample_name = sorted(n for n in default_named if "lora_A" in n and "q_proj" in n)[0]
    sample_ref_name = sample_name.replace(".default.", ".ref.")
    default_sample = default_named[sample_name]
    ref_sample = ref_named[sample_ref_name]

    max_abs_diff = (default_sample.detach().float() - ref_sample.detach().float()).abs().max().item()
    log(f"[sample param] {sample_name}")
    log(f"  default.requires_grad={default_sample.requires_grad}  ref.requires_grad={ref_sample.requires_grad}")
    log(f"  max|default - ref| at init = {max_abs_diff:.3e}  (expect ~0.0 -- ref is a clone of B1's default)")

    if ref_sample.requires_grad:
        raise RuntimeError("REFERENCE VERIFICATION FAILED: 'ref' adapter parameter has requires_grad=True "
                            "(it should be frozen). Do not launch training.")
    if not default_sample.requires_grad:
        raise RuntimeError("REFERENCE VERIFICATION FAILED: 'default' adapter parameter has requires_grad=False "
                            "(DPO would train nothing). Do not launch training.")
    if max_abs_diff > 1e-4:
        raise RuntimeError(
            f"REFERENCE VERIFICATION FAILED: 'ref' adapter differs from 'default' by {max_abs_diff:.3e} at "
            "initialization -- the reference clone should be numerically identical to B1's checkpoint at this "
            "point (before any optimizer step). Do not launch training."
        )
    log("[static check] PASSED: ref is a frozen, numerically-identical clone of B1's trained (trainable) "
        "'default' adapter -- not the base model (non-zero LoRA delta) and not a trainable copy.")

    # ---- Dynamic check: one real batch, manual reference recomputation + real backward ----
    batch = next(iter(trainer.get_train_dataloader()))
    batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    model.eval()
    with torch.no_grad(), use_adapter(model, adapter_name="ref"):
        assert model.active_adapters == ["ref"], f"active adapter did not switch to 'ref': {model.active_adapters}"
        ref_out = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
        )
        ref_shift_logits = ref_out.logits[..., :-1, :]
        shift_labels = batch["input_ids"][..., 1:]
        shift_completion_mask = batch["completion_mask"][..., 1:]
        # Upcast to float32 BEFORE masking/summing. The model forward itself still runs in
        # bf16 (weights are bf16), but summing ~100-300 per-token log-probs (each already
        # bf16-rounded) in bf16 accumulates real rounding error: at magnitudes of a few
        # hundred, bf16's representable spacing is ~magnitude/128, e.g. ~2.3 at |x|=300 --
        # comfortably able to produce a 0.5 divergence between two mathematically-equivalent
        # summation orders with no bug involved. DPOTrainer's own cached ref_chosen_logps /
        # ref_rejected_logps are float32 (confirmed empirically: printed without a dtype
        # suffix, i.e. torch's default float32, vs. my first draft's uncast bf16 sum, which
        # did print `dtype=torch.bfloat16`) -- so comparing a bf16 accumulation against a
        # float32 one was never an apples-to-apples comparison. Fixed here.
        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels).float()
        ref_per_token_logps[shift_completion_mask == 0] = 0.0
        my_ref_logps = ref_per_token_logps.sum(dim=1)
        my_ref_chosen, my_ref_rejected = my_ref_logps.chunk(2, dim=0)
    log(f"[active adapters back to default after context exit] {model.active_adapters}")
    assert model.active_adapters == ["default"], f"adapter did not restore to 'default': {model.active_adapters}"

    if "ref_chosen_logps" in batch:
        cached_chosen = batch["ref_chosen_logps"].float()
        cached_rejected = batch["ref_rejected_logps"].float()
        diff_chosen = (my_ref_chosen - cached_chosen).abs().max().item()
        diff_rejected = (my_ref_rejected - cached_rejected).abs().max().item()
        # Tolerance is relative-plus-absolute, not a flat epsilon: the underlying model
        # forward pass runs in bf16 regardless of the float32 accumulation above, so a few
        # bf16 quantization steps of residual difference (~1-2 at magnitudes of a few
        # hundred) is expected numerical noise, not evidence of a wrong adapter. This was
        # verified empirically on a tiny (16-example) synthetic batch, where the manual and
        # cached values matched exactly once both were compared in float32.
        tol_chosen = 2.0 + 0.02 * cached_chosen.abs().max().item()
        tol_rejected = 2.0 + 0.02 * cached_rejected.abs().max().item()
        log(f"[manual vs precomputed ref logps] max|diff| chosen={diff_chosen:.3e} (tol={tol_chosen:.2f}) "
            f"rejected={diff_rejected:.3e} (tol={tol_rejected:.2f}) -- both should be small relative to bf16 "
            "forward-pass precision at these magnitudes, not exactly 0")
        if diff_chosen > tol_chosen or diff_rejected > tol_rejected:
            raise RuntimeError(
                "REFERENCE VERIFICATION FAILED: my independently-recomputed reference log-probs (via the "
                "'ref' adapter) do not match DPOTrainer's own cached ref_chosen_logps/ref_rejected_logps, "
                "beyond what bf16 forward-pass precision can explain. The precomputed reference may not "
                "actually be B1's adapter. Do not launch training."
            )
        log("[dynamic reference check] PASSED: independently-recomputed 'ref'-adapter log-probs match "
            "DPOTrainer's own cached reference log-probs within bf16-forward-pass precision.")
    else:
        log("[note] precompute_ref_log_probs=False for this config -- no cached values to cross-check; "
            "the manual 'ref'-adapter forward pass above is itself the evidence that a real, distinct "
            "frozen adapter was used.")

    # ---- Real forward+backward through the trainer's own loss, gradients inspected --------
    model.train()
    for p in model.parameters():
        p.grad = None
    loss = trainer.compute_loss(model, batch, return_outputs=False)
    loss.backward()

    default_grad_norm = default_sample.grad.detach().float().norm().item() if default_sample.grad is not None else None
    ref_grad = ref_sample.grad
    log(f"[after one real backward()] loss={loss.item():.4f}")
    log(f"  default adapter grad norm (sample param) = {default_grad_norm}")
    log(f"  ref adapter grad (sample param) = {'None' if ref_grad is None else ref_grad.norm().item()}")

    metrics = trainer._metrics.get("train", {})
    for key in ("rewards/chosen", "rewards/rejected", "rewards/margins", "logps/chosen", "logps/rejected"):
        if key in metrics and metrics[key]:
            log(f"  {key} = {metrics[key][-1]:.4f}")
    log("  (at initialization, default==ref exactly, so rewards/chosen and rewards/rejected are expected "
        "~0 and loss ~ln(2)=0.6931 -- the standard DPO 'untrained-relative-to-its-own-reference' signature)")

    # Clear gradients again before real training starts, so this pre-flight check does not
    # contaminate the first real optimizer step.
    for p in model.parameters():
        p.grad = None

    if default_grad_norm is None or default_grad_norm == 0.0:
        raise RuntimeError(
            "GRADIENT VERIFICATION FAILED: the trainable 'default' adapter received no gradient (or an "
            "exactly-zero-norm gradient) from one real DPO loss.backward(). Do not launch training."
        )
    if ref_grad is not None:
        raise RuntimeError(
            "GRADIENT VERIFICATION FAILED: the frozen 'ref' adapter received a non-None gradient -- it "
            "should never be touched by autograd. Do not launch training."
        )
    log("[dynamic gradient check] PASSED: 'default' adapter receives real, non-zero gradients; "
        "'ref' adapter's grad stayed None throughout.")

    dump_path = os.path.join(output_dir, "dpo_reference_verification.txt")
    with open(dump_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[reference verification dump written to] {dump_path}")


def main():
    parser = argparse.ArgumentParser(description="Project 71 DPO training (shared by B2 and T)")
    parser.add_argument("--config", type=str, required=True, help="configs/dpo_b2.yaml or configs/dpo_t.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max optimizer steps (for smoke checks only). Leave unset for a full run.",
    )
    parser.add_argument(
        "--min_free_gpu_gb",
        type=float,
        default=20.0,
        help="GPU headroom floor (GB); the script refuses to proceed below this.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the config's output_dir_template result (e.g. to write "
        "results/B2_dpo_seed1_v2 for a rerun on a retrained base adapter, without editing "
        "the canonical config or overwriting the prior run's directory).",
    )
    parser.add_argument(
        "--t_output_dir",
        type=str,
        default=None,
        help="T_ctrl only: override where to look for T's dpo_data_manifest.json (default: "
        "configs/dpo_t.yaml's own output_dir_template formatted with --seed). Needed only if "
        "T was itself launched with a non-default --output_dir.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    set_seed_everywhere(seed)

    check_gpu_headroom(args.min_free_gpu_gb)

    m_cfg = cfg["model"]
    ba_cfg = cfg["base_adapter"]
    d_cfg = cfg["data"]
    l_cfg = cfg["lora"]
    t_cfg = cfg["training"]

    arm_name = d_cfg.get("arm")
    if arm_name not in ("B2", "T", "T_ctrl"):
        raise RuntimeError(
            f"data.arm must be one of B2/T/T_ctrl in {args.config}, got {arm_name!r}. "
            "Refusing to infer the arm implicitly."
        )
    output_dir = args.output_dir if args.output_dir is not None else t_cfg["output_dir_template"].format(seed=seed)
    if os.path.isdir(output_dir):
        pre_existing = [f for f in os.listdir(output_dir) if not f.endswith(".log")]
        if pre_existing:
            raise RuntimeError(
                f"Output dir {output_dir} already exists and is non-empty ({pre_existing}). "
                "Refusing to overwrite an existing run's directory -- pass a fresh --output_dir "
                "or clear it explicitly first."
            )
    os.makedirs(output_dir, exist_ok=True)

    print(f"[config] {args.config}")
    print(f"[arm] {arm_name}")
    print(f"[seed] {seed}")
    print(f"[output_dir] {output_dir}")

    # ---- Hard gates -----------------------------------------------------------------------
    assert_never_redteam(d_cfg["safety_pairs_file"])
    assert_lora_matches_b1_template(l_cfg)
    assert_hyperparams_match_sibling(args.config, cfg)

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

    # ---- Safety pool (T / T_ctrl only): built in-memory from the raw PKU-SafeRLHF dataset,
    #      never reading data/processed/pref_safety.jsonl -- see prepare_pref.build_safety_pairs
    #      docstring. Row selection is IDENTICAL regardless of `safety_direction`, so T and
    #      T_ctrl structurally draw from the same rows; only chosen/rejected differs. --------
    n_safety_needed = d_cfg["n_safety_sample"]
    safety_direction = d_cfg.get("safety_direction", "safer")
    sampled_safety = []
    n_excl_length_safety = 0
    safety_funnel = None
    if n_safety_needed > 0:
        safety_pool, safety_funnel = build_safety_pairs(seed=seed, system_prompt=system_prompt, direction=safety_direction)
        print(f"[safety pool, direction={safety_direction}] {safety_funnel}")
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
        print(f"[sampled safety] n={len(sampled_safety)} (seed={seed}, direction={safety_direction}, "
              f"from pool of {len(safety_pool)})")

    # ---- Matched-volume assertion (Methodological Safeguards v2, rule 2) ---------------------
    total_pairs = len(sampled_helpful) + len(sampled_safety)
    print(f"[matched volume] helpful={len(sampled_helpful)} safety={len(sampled_safety)} total={total_pairs}")

    sampled_helpful_ids = [pair_id(p) for p in sampled_helpful]
    sampled_safety_ids = [pair_id(p) for p in sampled_safety]

    # ---- T_ctrl-specific check: its 15,000 helpfulness pairs must be the IDENTICAL sample
    #      to T's (same seed, same pool -> deterministic, but verify empirically against T's
    #      own manifest rather than just trusting the determinism argument) -----------------
    if arm_name == "T_ctrl":
        t_cfg_for_lookup = load_config(os.path.join(os.path.dirname(args.config), "dpo_t.yaml"))
        t_output_dir = args.t_output_dir or t_cfg_for_lookup["training"]["output_dir_template"].format(seed=seed)
        t_manifest_path = os.path.join(t_output_dir, "dpo_data_manifest.json")
        if not os.path.isfile(t_manifest_path):
            raise RuntimeError(
                f"T_ctrl requires T's manifest to verify identical helpful-pair sampling, but "
                f"{t_manifest_path} was not found. Run T seed {seed} first (per the coordinator's "
                "sequencing: B2 -> T -> T_ctrl), or pass --t_output_dir explicitly if T was written "
                "to a non-default directory. Refusing to proceed without this verification."
            )
        with open(t_manifest_path, "r", encoding="utf-8") as f:
            t_manifest = json.load(f)
        t_helpful_ids = t_manifest.get("sampled_helpful_pair_ids")
        if t_helpful_ids != sampled_helpful_ids:
            n_diff = len(set(t_helpful_ids or []) ^ set(sampled_helpful_ids))
            raise RuntimeError(
                f"T_CTRL/T HELPFUL-SAMPLE MATCH ASSERT FAILED: T_ctrl's sampled helpful pair ids do "
                f"not exactly match T's ({t_manifest_path}). Symmetric-difference size: {n_diff}. "
                "T_ctrl is defined as using the IDENTICAL 15,000 helpfulness pairs as T -- this must "
                "not silently diverge. Refusing to proceed."
            )
        print(f"[T_ctrl/T helpful-sample match assert] PASSED: identical {len(sampled_helpful_ids)} "
              f"helpful pair ids as {t_manifest_path}.")

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
        "safety_direction": safety_direction if n_safety_needed > 0 else None,
        "safety_funnel": safety_funnel,
        "n_excluded_helpful_by_max_length": n_excl_length_helpful,
        "n_excluded_safety_by_max_length": n_excl_length_safety,
        "max_length": d_cfg["max_length"],
        "sampled_helpful_pair_ids": sampled_helpful_ids,
        "sampled_safety_pair_ids": sampled_safety_ids,
    }
    manifest_path = os.path.join(output_dir, "dpo_data_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[written] {manifest_path}")

    train_pairs = sampled_helpful + sampled_safety
    assert_zero_truncation_dpo(train_pairs, tokenizer, d_cfg["max_length"])
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
        max_steps=args.max_steps if args.max_steps is not None else -1,
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

    # ---- DPO analogue of B1's assistant-only-loss masking verification (critical gate) ------
    verify_reference_and_gradients(trainer, model, tokenizer, output_dir)

    # ---- Train --------------------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0
    print(f"[train result] {result}")
    print(f"[wall_clock_sec] {wall:.1f}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[peak_vram_gb] {peak_gb:.2f}")

    model.save_pretrained(output_dir, selected_adapters=["default"])  # save ONLY the trained
    #     adapter, not the frozen "ref" clone TRL added alongside it.
    tokenizer.save_pretrained(output_dir)
    print(f"[done] model + tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
