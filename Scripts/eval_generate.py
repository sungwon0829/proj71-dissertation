"""
Project 71 -- PASS 1 of 2: batch generation over the frozen red-team suite.

Feeds: every number in Table 1 (ASR, over-refusal, helpfulness) and Table 2 (per-category
ASR). This script produces NO metric itself. It produces the raw model outputs that
scripts/eval_score.py (pass 2) turns into numbers. Generation and scoring are separate
passes so judges can be re-run, swapped or re-calibrated WITHOUT regenerating.

RAW GENERATIONS ARE NEVER OVERWRITTEN. If the output file already exists this script
refuses to run unless --allow_new_version is given, in which case it writes
generations.v2.jsonl / .v3.jsonl / ... and leaves every earlier file untouched.

IDENTICAL TREATMENT ACROSS ARMS: the same suite, the same pinned system prompt, the same
chat template and the same pinned decoding settings are used for B0, B1, B2, B3 and T.
Everything that could differ is read from configs/eval_generation.yaml and copied verbatim
into the output header, so the settings live with the data.

Decoding is GREEDY (do_sample=false). This is deliberate, not a default: the headline claim
is a PAIRED comparison over a fixed prompt set, and reported uncertainty comes from
bootstrap over prompts plus 3 training seeds. Decoding noise would inflate variance without
adding information about the treatment, and greedy makes every number exactly reproducible.

SUITE SCHEMA (one JSON object per line):
    {"id": "<category>_###",
     "category": "prefilling|persona|many_shot|crisis_adjacent|benign_sensitive",
     "messages": [{"role": "user"|"assistant", "content": "..."}],
     "attack_note": "...", "expected_safe_behaviour": "..."}
  - messages contains NO system message; the pinned one is prepended here.
  - prefilling items END with a partial assistant turn. Those are rendered with
    continue_final_message=True so the model CONTINUES that turn instead of starting a new
    one (verified against transformers 5.14.1: the renderer appends a sentinel to the final
    message, then truncates the rendered string at the sentinel, leaving the prefill in
    place with no <|im_end|>).

Usage:
    python Scripts\\eval_generate.py --arm b0 --seed 42 \
        --suite data\\redteam\\redteam_suite.jsonl
    python Scripts\\eval_generate.py --arm b1 --seed 42 \
        --adapter results\\B1_sft_seed42\\checkpoint-290 \
        --suite data\\redteam\\redteam_suite.jsonl
    python Scripts\\eval_generate.py --arm b0 --seed 42 --suite ... --verify_batching
"""

import argparse
import datetime
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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_CATEGORIES = {"prefilling", "persona", "many_shot", "crisis_adjacent", "benign_sensitive"}
ATTACK_CATEGORIES = {"prefilling", "persona", "many_shot", "crisis_adjacent"}


# --------------------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------------------
def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path_like) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------
# suite loading -- fail loudly, never silently skip
# --------------------------------------------------------------------------------------
def load_suite(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(
            f"Red-team suite not found: {path}\n"
            "The real suite is data/redteam/redteam_suite.jsonl. For plumbing work before "
            "it exists, pass --suite data/redteam/_fixture_dev.jsonl (DEV FIXTURE -- its "
            "numbers are NOT paper numbers)."
        )
    items, seen_ids = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {e}") from e

            for field in ("id", "category", "messages"):
                if field not in rec:
                    raise ValueError(f"{path}:{lineno}: missing required field '{field}'")
            if rec["id"] in seen_ids:
                raise ValueError(f"{path}:{lineno}: duplicate id {rec['id']!r}")
            seen_ids.add(rec["id"])

            if rec["category"] not in VALID_CATEGORIES:
                raise ValueError(
                    f"{path}:{lineno}: unknown category {rec['category']!r}; "
                    f"expected one of {sorted(VALID_CATEGORIES)}"
                )
            msgs = rec["messages"]
            if not isinstance(msgs, list) or not msgs:
                raise ValueError(f"{path}:{lineno}: 'messages' must be a non-empty list")
            for m in msgs:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    raise ValueError(f"{path}:{lineno}: every message needs 'role' and 'content'")
                if m["role"] == "system":
                    raise ValueError(
                        f"{path}:{lineno}: suite items must contain NO system message; the "
                        "harness prepends the pinned one from configs/system_prompt.txt"
                    )
                if m["role"] not in ("user", "assistant"):
                    raise ValueError(f"{path}:{lineno}: role must be 'user' or 'assistant', got {m['role']!r}")

            # prefilling <=> final message is a partial assistant turn.
            final_role = msgs[-1]["role"]
            if rec["category"] == "prefilling":
                if final_role != "assistant":
                    raise ValueError(
                        f"{path}:{lineno}: category 'prefilling' but the final message is "
                        f"role={final_role!r}; a prefilling item must END with the partial "
                        "assistant turn the model has to continue."
                    )
                if not str(msgs[-1]["content"]).strip():
                    raise ValueError(f"{path}:{lineno}: prefilling item has an empty assistant prefill")
            else:
                if final_role != "user":
                    raise ValueError(
                        f"{path}:{lineno}: category {rec['category']!r} must END with a user "
                        f"turn, got role={final_role!r}. Only 'prefilling' ends on assistant."
                    )
            items.append(rec)

    if not items:
        raise ValueError(f"{path}: suite is empty")
    return items


# --------------------------------------------------------------------------------------
# prompt rendering
# --------------------------------------------------------------------------------------
def render_prompt(tokenizer, system_text: str, rec: dict) -> tuple:
    """Return (rendered_prompt_string, is_prefill).

    Rendered as a STRING (tokenize=False) so the exact text can be saved verbatim and
    batch-tokenised uniformly afterwards. Mixing prefill and non-prefill items in one
    apply_chat_template call is impossible; rendering per item then batching the strings
    avoids the problem entirely and keeps every item's prompt auditable.
    """
    messages = [{"role": "system", "content": system_text}] + list(rec["messages"])
    is_prefill = rec["category"] == "prefilling"
    if is_prefill:
        # CONTINUE the partial assistant turn; do NOT start a new one.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        prefill = rec["messages"][-1]["content"]
        if not text.endswith(prefill.rstrip()) and not text.endswith(prefill):
            raise RuntimeError(
                f"[{rec['id']}] continue_final_message did not leave the prefill at the end "
                f"of the rendered prompt. Rendered tail: {text[-120:]!r}"
            )
        if text.rstrip().endswith("<|im_end|>"):
            raise RuntimeError(
                f"[{rec['id']}] rendered prefill prompt ends with <|im_end|> -- the assistant "
                "turn was closed, so the model would start a NEW turn instead of continuing."
            )
    else:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not text.endswith("<|im_start|>assistant\n"):
            raise RuntimeError(
                f"[{rec['id']}] expected the rendered prompt to end with the assistant "
                f"generation header. Tail: {text[-60:]!r}"
            )
    return text, is_prefill


# --------------------------------------------------------------------------------------
# batched generation with LEFT padding
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list, gen_cfg: GenerationConfig, device: str) -> list:
    """Greedy-generate continuations for a list of already-rendered prompt strings.

    Left padding is mandatory for decoder-only batched generation: with right padding the
    pad tokens sit BETWEEN the prompt and the first generated token, and outputs are
    silently corrupted. tokenizer.padding_side is set to 'left' by the caller; it is
    re-asserted here because getting this wrong produces plausible-looking garbage.
    """
    assert tokenizer.padding_side == "left", "decoder-only batched generation requires left padding"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,   # the chat template already emitted all control tokens
    ).to(device)

    out = model.generate(**enc, generation_config=gen_cfg)
    prompt_len = enc["input_ids"].shape[1]
    texts = []
    for i in range(out.shape[0]):
        new_ids = out[i, prompt_len:]
        texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return texts


@torch.no_grad()
def _last_logits(model, tokenizer, prompts: list, row: int, device: str):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    out = model(**enc)
    return out.logits[row, -1, :].float().cpu(), enc["input_ids"].shape[1]


@torch.no_grad()
def padding_probe(model, tokenizer, short_prompt: str, long_prompt: str, batch_size: int, device: str) -> dict:
    """Decisive left-padding correctness test.

    Exact string match between batched and batch-size-1 generation is NOT a valid test on
    this stack: in bf16 the matmul reduction order depends on batch size, so logits shift by
    ~0.3-0.4 regardless of padding, which flips greedy choices at near-ties and makes
    continuations diverge legitimately. That test therefore fails even when padding is
    perfect, and would send you hunting a bug that isn't there.

    This probe separates the two causes by comparing next-token logits for ONE prompt under:
        A  alone                       (batch 1, zero padding)
        B  batched with the longest prompt in the suite   (batch 2, HEAVY left padding)
        C  batched with batch_size copies of itself       (batch N, ZERO padding)
    If left padding is correct, A-vs-B (padding + batching) is the same order of magnitude
    as A-vs-C (batching alone), because padding contributes nothing.
    A right-padded control is run to show the scale of an actual padding bug.
    """
    assert tokenizer.padding_side == "left"
    la, n_a = _last_logits(model, tokenizer, [short_prompt], 0, device)
    lb, n_b = _last_logits(model, tokenizer, [short_prompt, long_prompt], 0, device)
    lc, _ = _last_logits(model, tokenizer, [short_prompt] * max(batch_size, 2), 0, device)
    tokenizer.padding_side = "right"
    try:
        lr, _ = _last_logits(model, tokenizer, [short_prompt, long_prompt], 0, device)
    finally:
        tokenizer.padding_side = "left"

    def dmax(x, y):
        return float((x - y).abs().max())

    pad_plus_batch = dmax(la, lb)      # left padding + batching
    batch_only = dmax(la, lc)          # batching alone, no padding
    bug_control = dmax(la, lr)         # right padding = known-wrong
    # Padding must add nothing beyond batch numerics, and must be nowhere near the bug scale.
    passed = (pad_plus_batch <= max(3.0 * batch_only, 0.5)) and (pad_plus_batch < 0.10 * bug_control)
    return {
        "test": "next-token logit comparison for one prompt under three padding/batching conditions",
        "left_pad_tokens_in_condition_B": n_b - n_a,
        "batch_size_condition_C": max(batch_size, 2),
        "max_abs_logit_delta": {
            "A_alone_vs_B_heavy_left_padding": round(pad_plus_batch, 5),
            "A_alone_vs_C_batched_zero_padding": round(batch_only, 5),
            "A_alone_vs_RIGHT_PADDING_known_bug_control": round(bug_control, 5),
        },
        "argmax_same": {
            "A_vs_B": int(la.argmax()) == int(lb.argmax()),
            "A_vs_C": int(la.argmax()) == int(lc.argmax()),
            "A_vs_RIGHT_PAD": int(la.argmax()) == int(lr.argmax()),
        },
        "criterion": "PASS if A-vs-B <= max(3 * A-vs-C, 0.5) AND A-vs-B < 0.10 * right-pad control",
        "passed": bool(passed),
        "interpretation": (
            "left padding correct; any batched-vs-unbatched text divergence is bf16 "
            "batch-size numerics, not padding" if passed else
            "LEFT PADDING LOOKS WRONG -- do not trust these generations"
        ),
    }


def next_version_path(base: Path) -> Path:
    """generations.jsonl -> generations.v2.jsonl -> generations.v3.jsonl ..."""
    stem, suffix = base.stem, base.suffix
    n = 2
    while True:
        cand = base.with_name(f"{stem}.v{n}{suffix}")
        if not cand.exists():
            return cand
        n += 1


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Project 71 pass 1: red-team suite generation")
    ap.add_argument("--arm", required=True, help="Arm label, e.g. b0, b1, b2, b3, t (used in the output path)")
    ap.add_argument("--adapter", default=None, help="Path to a LoRA adapter directory. Omit for B0 (base model).")
    ap.add_argument("--suite", required=True, help="Path to the red-team suite JSONL")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/eval_generation.yaml")
    ap.add_argument("--out", default=None, help="Override the output file path")
    ap.add_argument("--batch_size", type=int, default=None, help="Override the pinned batch size (DIAGNOSTIC ONLY)")
    ap.add_argument("--limit", type=int, default=None, help="Generate for only the first N items (DEBUG ONLY -- marks the output)")
    ap.add_argument("--allow_new_version", action="store_true",
                    help="If the output file exists, write generations.vN.jsonl instead of refusing. Never overwrites.")
    ap.add_argument("--verify_batching", action="store_true",
                    help="Also regenerate the first --verify_n items at batch size 1 and report exact-match against the batched result.")
    ap.add_argument("--verify_n", type=int, default=6)
    args = ap.parse_args()

    t_start = time.time()
    cfg = load_config(resolve(args.config))
    m_cfg, p_cfg, d_cfg, b_cfg, o_cfg = (cfg["model"], cfg["prompting"], cfg["decoding"],
                                         cfg["batching"], cfg["output"])

    set_seed_everywhere(args.seed)

    # ---- suite -------------------------------------------------------------------------
    suite_path = resolve(args.suite)
    suite = load_suite(suite_path)
    suite_sha = sha256_file(suite_path)
    is_fixture = "_fixture" in suite_path.name
    if args.limit is not None:
        suite = suite[: args.limit]
    cat_counts = {}
    for r in suite:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    print(f"[suite] {suite_path}")
    print(f"[suite] n={len(suite)}  sha256={suite_sha}")
    print(f"[suite] categories: {cat_counts}")
    if is_fixture:
        print("[suite] *** DEV FIXTURE -- outputs are for plumbing only, NOT paper numbers ***")
    if args.limit is not None:
        print(f"[suite] *** --limit {args.limit} active: PARTIAL RUN, NOT paper numbers ***")

    # ---- output path (never overwrite) --------------------------------------------------
    if args.out:
        out_path = resolve(args.out)
    else:
        out_dir = resolve(o_cfg["dir_template"].format(arm=args.arm, seed=args.seed))
        out_path = out_dir / o_cfg["filename"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if not args.allow_new_version:
            raise FileExistsError(
                f"RAW GENERATIONS ALREADY EXIST: {out_path}\n"
                "This script never overwrites raw model outputs. Re-run with "
                "--allow_new_version to write generations.vN.jsonl alongside it, or point "
                "--out somewhere else."
            )
        out_path = next_version_path(out_path)
        print(f"[output] existing file preserved; writing new version -> {out_path}")

    # ---- system prompt (read from file; never retyped) ----------------------------------
    sys_path = resolve(p_cfg["system_prompt_file"])
    system_text = sys_path.read_text(encoding="utf-8").strip()
    system_sha = sha256_file(sys_path)

    # ---- tokenizer + pinned chat template ------------------------------------------------
    tok = AutoTokenizer.from_pretrained(
        m_cfg["name_or_path"], cache_dir=m_cfg["cache_dir"],
        revision=m_cfg.get("revision"), trust_remote_code=m_cfg["trust_remote_code"],
    )
    tpl_path = resolve(p_cfg["chat_template_path"])
    tok.chat_template = tpl_path.read_text(encoding="utf-8")
    tpl_sha = sha256_file(tpl_path)
    tok.padding_side = "left"                       # MANDATORY for decoder-only generation
    tok.pad_token_id = int(d_cfg["pad_token_id"])
    tok.pad_token = tok.convert_ids_to_tokens(tok.pad_token_id)

    # ---- model ---------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, m_cfg["dtype"])
    print(f"[model] base {m_cfg['name_or_path']} rev={m_cfg.get('revision')} "
          f"dtype={m_cfg['dtype']} attn={m_cfg['attn_implementation']}")
    model = AutoModelForCausalLM.from_pretrained(
        m_cfg["name_or_path"], cache_dir=m_cfg["cache_dir"], revision=m_cfg.get("revision"),
        dtype=dtype, attn_implementation=m_cfg["attn_implementation"],
        trust_remote_code=m_cfg["trust_remote_code"],
    )

    adapter_info = None
    if args.adapter:
        from peft import PeftModel
        adapter_path = resolve(args.adapter)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
        acfg_path = adapter_path / "adapter_config.json"
        acfg = json.loads(acfg_path.read_text(encoding="utf-8"))
        if acfg.get("base_model_name_or_path") != m_cfg["name_or_path"]:
            raise RuntimeError(
                f"Adapter was trained on {acfg.get('base_model_name_or_path')!r} but this "
                f"config loads {m_cfg['name_or_path']!r}."
            )
        model = PeftModel.from_pretrained(model, str(adapter_path))
        adapter_info = {
            "path": str(adapter_path),
            "adapter_config_sha256": sha256_file(acfg_path),
            "weights_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
            "r": acfg.get("r"), "lora_alpha": acfg.get("lora_alpha"),
            "target_modules": acfg.get("target_modules"),
        }
        print(f"[adapter] {adapter_path} (r={acfg.get('r')}, alpha={acfg.get('lora_alpha')})")
    else:
        print("[adapter] none -- base model (B0)")

    model.to(device)
    model.eval()

    # ---- pinned generation config -------------------------------------------------------
    # Qwen2.5-7B-Instruct ships do_sample=true / temperature=0.7 / top_p=0.8 / top_k=20 /
    # repetition_penalty=1.05 in generation_config.json. Those apply SILENTLY unless
    # replaced. Build the config from scratch so nothing is inherited.
    gen_cfg = GenerationConfig(
        do_sample=bool(d_cfg["do_sample"]),
        num_beams=int(d_cfg["num_beams"]),
        max_new_tokens=int(d_cfg["max_new_tokens"]),
        min_new_tokens=int(d_cfg["min_new_tokens"]),
        repetition_penalty=float(d_cfg["repetition_penalty"]),
        eos_token_id=list(d_cfg["eos_token_ids"]),
        pad_token_id=int(d_cfg["pad_token_id"]),
    )
    for k in ("temperature", "top_p", "top_k"):
        setattr(gen_cfg, k, None)
    model.generation_config = gen_cfg
    effective_decoding = {
        "do_sample": gen_cfg.do_sample, "num_beams": gen_cfg.num_beams,
        "temperature": gen_cfg.temperature, "top_p": gen_cfg.top_p, "top_k": gen_cfg.top_k,
        "repetition_penalty": gen_cfg.repetition_penalty,
        "max_new_tokens": gen_cfg.max_new_tokens, "min_new_tokens": gen_cfg.min_new_tokens,
        "eos_token_id": gen_cfg.eos_token_id, "pad_token_id": gen_cfg.pad_token_id,
    }
    print(f"[decoding] {effective_decoding}")

    # ---- render every prompt -------------------------------------------------------------
    rendered = []
    for rec in suite:
        text, is_prefill = render_prompt(tok, system_text, rec)
        rendered.append({"rec": rec, "prompt": text, "is_prefill": is_prefill})
    n_prefill = sum(r["is_prefill"] for r in rendered)
    print(f"[render] {len(rendered)} prompts ({n_prefill} rendered with continue_final_message=True)")

    # ---- batch (length-sorted) -------------------------------------------------------------
    batch_size = args.batch_size or int(b_cfg["batch_size"])
    for r in rendered:
        r["n_tokens"] = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
    order = sorted(range(len(rendered)), key=lambda i: rendered[i]["n_tokens"]) \
        if b_cfg.get("length_sorted_batching", True) else list(range(len(rendered)))

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_gen0 = time.time()
    results = {}
    batch_composition = []
    for bstart in range(0, len(order), batch_size):
        idxs = order[bstart: bstart + batch_size]
        batch_composition.append([rendered[i]["rec"]["id"] for i in idxs])
        prompts = [rendered[i]["prompt"] for i in idxs]
        set_seed_everywhere(args.seed)   # no-op under greedy, but keeps the run reproducible if decoding is ever changed
        outs = generate_batch(model, tok, prompts, gen_cfg, device)
        for i, text in zip(idxs, outs):
            results[i] = text
        print(f"  [gen] {min(bstart + batch_size, len(order))}/{len(order)} "
              f"(max prompt tokens in batch: {max(rendered[i]['n_tokens'] for i in idxs)})")
    gen_seconds = time.time() - t_gen0
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device == "cuda" else None

    # ---- batched-vs-unbatched padding verification -------------------------------------------
    batching_check = None
    if args.verify_batching:
        print("\n[verify] --- left-padding correctness probe (logit-level, decisive) ---")
        short_i = min(range(len(rendered)), key=lambda i: rendered[i]["n_tokens"])
        long_i = max(range(len(rendered)), key=lambda i: rendered[i]["n_tokens"])
        probe = padding_probe(model, tok, rendered[short_i]["prompt"], rendered[long_i]["prompt"],
                              batch_size, device)
        probe["probe_short_id"] = rendered[short_i]["rec"]["id"]
        probe["probe_long_id"] = rendered[long_i]["rec"]["id"]
        for k, v in probe["max_abs_logit_delta"].items():
            print(f"  [verify] max|dlogit| {k}: {v}")
        print(f"  [verify] {'PASS' if probe['passed'] else '*** FAIL ***'} -- {probe['interpretation']}")

        # Informational: how often does batched text differ from batch-size-1 text? Under bf16
        # this is EXPECTED to be non-zero; it quantifies decoding noise, it is not a pass/fail.
        print(f"\n[verify] --- batched vs batch-size-1 text divergence on {args.verify_n} longest items ---")
        check_idxs = sorted(range(len(rendered)), key=lambda i: -rendered[i]["n_tokens"])[: args.verify_n]
        divergences = []
        for i in check_idxs:
            set_seed_everywhere(args.seed)
            single = generate_batch(model, tok, [rendered[i]["prompt"]], gen_cfg, device)[0]
            if single != results[i]:
                k = 0
                while k < min(len(single), len(results[i])) and single[k] == results[i][k]:
                    k += 1
                divergences.append({
                    "id": rendered[i]["rec"]["id"],
                    "common_prefix_chars": k,
                    "batched_len": len(results[i]), "unbatched_len": len(single),
                    "batched_at_divergence": results[i][k:k + 80],
                    "unbatched_at_divergence": single[k:k + 80],
                })
                print(f"  [verify] {rendered[i]['rec']['id']}: diverges after {k} identical chars")
            else:
                print(f"  [verify] {rendered[i]['rec']['id']}: identical")

        batching_check = {
            "padding_probe": probe,
            "text_divergence": {
                "note": ("Non-zero divergence is EXPECTED and is not a padding fault: in bf16 the "
                         "matmul reduction order depends on batch size, shifting logits by ~0.3-0.4 "
                         "and flipping greedy choices at near-ties. Outputs remain bit-reproducible "
                         "for a FIXED (suite, batch_size, length-sort) triple, and batch composition "
                         "is identical across arms because it depends only on prompt lengths, which "
                         "are identical across arms. See padding_probe for the actual correctness test."),
                "n_checked": len(check_idxs),
                "n_diverged": len(divergences),
                "batch_size_used": batch_size,
                "details": divergences,
            },
        }
        n_div = len(divergences)
        print(f"  [verify] {n_div}/{len(check_idxs)} diverged (expected under bf16; see padding_probe for correctness)")
        if not probe["passed"]:
            raise RuntimeError(
                "PADDING PROBE FAILED -- batched generation is not equivalent to unbatched. "
                "These generations must not be used. Check tokenizer.padding_side and the attention mask."
            )

    # ---- write ------------------------------------------------------------------------------
    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    header = {
        "record_type": "run_header",
        "schema_version": 1,
        "arm": args.arm,
        "seed": args.seed,
        "timestamp": timestamp,
        "script": "Scripts/eval_generate.py",
        "config_file": str(args.config),
        "config_sha256": sha256_file(resolve(args.config)),
        "suite_path": str(suite_path),
        "suite_sha256": suite_sha,
        "suite_n_items": len(suite),
        "suite_category_counts": cat_counts,
        "suite_is_dev_fixture": is_fixture,
        "partial_run_limit": args.limit,
        "is_paper_number": (not is_fixture) and args.limit is None,
        "system_prompt_file": str(sys_path),
        "system_prompt_sha256": system_sha,
        "system_prompt_text": system_text,
        "chat_template_file": str(tpl_path),
        "chat_template_sha256": tpl_sha,
        "model": {k: m_cfg[k] for k in m_cfg},
        "adapter": adapter_info,
        "decoding_effective": effective_decoding,
        "batching": {"batch_size": batch_size,
                     "padding_side": tok.padding_side,
                     "length_sorted": bool(b_cfg.get("length_sorted_batching", True)),
                     # Recorded so a reviewer (or repro-auditor) can confirm that batch
                     # composition was IDENTICAL across arms. It depends only on prompt token
                     # lengths, which are identical across arms, so it must match exactly.
                     "batch_composition_ids": batch_composition},
        "batching_verification": batching_check,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": (__import__("peft").__version__ if args.adapter else None),
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        },
        "performance": {
            "generation_seconds": round(gen_seconds, 2),
            "items_per_second": round(len(suite) / gen_seconds, 3) if gen_seconds > 0 else None,
            "peak_vram_gb": round(peak_vram_gb, 2) if peak_vram_gb is not None else None,
            "total_wallclock_seconds": None,   # filled below
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        header["performance"]["total_wallclock_seconds"] = round(time.time() - t_start, 2)
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i, r in enumerate(rendered):
            rec = r["rec"]
            continuation = results[i]
            prefill = rec["messages"][-1]["content"] if r["is_prefill"] else ""
            f.write(json.dumps({
                "record_type": "generation",
                "id": rec["id"],
                "category": rec["category"],
                "is_attack_category": rec["category"] in ATTACK_CATEGORIES,
                "arm": args.arm,
                "seed": args.seed,
                "messages": rec["messages"],
                "attack_note": rec.get("attack_note"),
                "expected_safe_behaviour": rec.get("expected_safe_behaviour"),
                "rendered_prompt": r["prompt"],
                "prompt_n_tokens": r["n_tokens"],
                "is_prefill": r["is_prefill"],
                "assistant_prefill": prefill,
                # what the model actually produced:
                "response_continuation": continuation,
                # what the user would actually receive (prefill + continuation). This is what
                # the judges see; see configs/judges.yaml judge_target.
                "response_full_turn": (prefill + continuation) if r["is_prefill"] else continuation,
                "response_empty": len(continuation.strip()) == 0,
                "timestamp": timestamp,
            }, ensure_ascii=False) + "\n")

    n_empty = sum(1 for i in range(len(rendered)) if not results[i].strip())
    print(f"\n[written] {out_path}  ({len(suite)} generations + 1 header record)")
    print(f"[perf] generation {gen_seconds:.1f}s  "
          f"({len(suite) / gen_seconds:.2f} items/s)  peak VRAM "
          f"{peak_vram_gb:.2f} GB" if peak_vram_gb else f"[perf] generation {gen_seconds:.1f}s")
    if n_empty:
        print(f"[warn] {n_empty} EMPTY continuation(s) -- inspect before scoring.")
    if is_fixture:
        print("[reminder] DEV FIXTURE run: these are NOT paper numbers.")


if __name__ == "__main__":
    main()
