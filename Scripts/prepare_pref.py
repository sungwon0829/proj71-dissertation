"""
prepare_pref.py -- Build DPO preference-pair JSONL files for Project 71.

Produces:
  1. data/processed/pref_safety.jsonl   from data/raw/pku_saferlhf (PKU-Alignment/PKU-SafeRLHF)
  2. data/processed/pref_helpful.jsonl  from data/raw/psychocounsel_pref (Psychotherapy-LLM/PsychoCounsel-Preference)

Both files use the TRL 1.9 DPOTrainer conversational schema:
    {"prompt": [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
     "chosen": [{"role": "assistant", "content": ...}],
     "rejected": [{"role": "assistant", "content": ...}]}

Fails loudly (raises) on any schema mismatch or missing raw data. Never modifies data/raw.

Usage:
    python scripts/prepare_pref.py --seed 0
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tokenizer load must be offline per spec

from datasets import load_from_disk  # noqa: E402

REPO_ROOT = Path("C:/proj71")
RAW_PKU_DIR = REPO_ROOT / "data" / "raw" / "pku_saferlhf"
RAW_PREF_DIR = REPO_ROOT / "data" / "raw" / "psychocounsel_pref"
REDTEAM_DIR = REPO_ROOT / "data" / "redteam"
SYSTEM_PROMPT_FILE = REPO_ROOT / "configs" / "system_prompt.txt"
OUT_SAFETY = REPO_ROOT / "data" / "processed" / "pref_safety.jsonl"
OUT_HELPFUL = REPO_ROOT / "data" / "processed" / "pref_helpful.jsonl"
OUT_SAFETY_CTRL = REPO_ROOT / "data" / "processed" / "pref_safety_ctrl.jsonl"  # T_ctrl only
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 2048

# ---------------------------------------------------------------------------
# Documented relevance filter for PKU-SafeRLHF -> therapy/mental-health domain
# ---------------------------------------------------------------------------
# Harm-category fields plausibly reachable in a counselling conversation.
# Excluded categories (Cybercrime, Economic Crime, Privacy Violation,
# White-Collar Crime, Insulting Behavior, Endangering National Security,
# Disrupting Public Order, Discriminatory Behavior, Copyright Issues,
# Endangering Public Health, Environmental Damage, Animal Abuse) are judged
# implausible as topics a client would raise with a counsellor and are dropped.
RELEVANT_HARM_CATEGORIES = {
    "Physical Harm",
    "Psychological Harm",
    "Mental Manipulation",
    "Violence",
    "Drugs",
    "Human Trafficking",
    "Sexual Content",
}

# Keyword buckets, one per domain named in the task spec. Matched against the
# PROMPT text only (the client-side trigger), case-insensitive, word-boundary
# regex to avoid spurious substring hits (e.g. "class" containing "ass").
KEYWORD_BUCKETS = {
    "self_harm_suicide": [
        r"suicid\w*", r"kill myself", r"end my life", r"ending my life",
        r"self[- ]?harm\w*", r"cutting myself", r"cut myself", r"want(ed)? to die",
        r"don'?t want to live", r"hurt(ing)? myself", r"take my own life",
        r"self[- ]?injur\w*", r"no reason to live",
    ],
    "substance_use": [
        r"alcoholic\w*", r"drinking problem", r"addict(ed|ion)?", r"relapse\w*",
        r"heroin", r"opioid\w*", r"meth(amphetamine)?", r"cocaine",
        r"overdos\w*", r"substance abuse", r"drug abuse", r"withdrawal symptoms",
    ],
    "disordered_eating": [
        r"anorexi\w*", r"bulimi\w*", r"binge eating", r"purging", r"purge after eating",
        r"starving myself", r"eating disorder\w*", r"not eating",
    ],
    "violence_toward_self_or_others": [
        r"hurt someone", r"hurt others", r"violent thoughts", r"domestic violence",
        r"abusive partner", r"hit(s)? me", r"beats? me", r"attack(ing)? someone",
    ],
    "medication_overdose": [
        r"overdos\w*", r"too many pills", r"took (\d+ )?pills", r"medication overdose",
        r"prescription pills", r"painkillers",
    ],
    "abuse_disclosure": [
        r"abused", r"sexually abused", r"molested", r"assaulted", r"\brape\b", r"raped",
        r"sexual assault", r"domestic abuse", r"being abused", r"my abuser",
    ],
    "severe_emotional_distress": [
        r"hopeless", r"worthless", r"can'?t go on", r"breakdown", r"panic attack\w*",
        r"\bcrisis\b", r"can'?t cope", r"mental breakdown",
    ],
}
KEYWORD_PATTERN = re.compile(
    "|".join(p for bucket in KEYWORD_BUCKETS.values() for p in bucket),
    flags=re.IGNORECASE,
)

REQUIRED_PKU_FIELDS = {
    "prompt", "response_0", "response_1", "is_response_0_safe", "is_response_1_safe",
    "response_0_harm_category", "response_1_harm_category", "better_response_id",
    "safer_response_id",
}
REQUIRED_PREF_FIELDS = {
    "question", "chosen", "rejected", "chosen_safety_rating", "rejected_safety_rating",
}


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"Pinned system prompt file missing: {SYSTEM_PROMPT_FILE}")
    text = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Pinned system prompt file is empty: {SYSTEM_PROMPT_FILE}")
    return text


def row_is_relevant(row) -> tuple[bool, bool, bool]:
    """Returns (category_hit, keyword_hit, combined)."""
    cat0 = row["response_0_harm_category"] or {}
    cat1 = row["response_1_harm_category"] or {}
    category_hit = any(cat0.get(c) for c in RELEVANT_HARM_CATEGORIES) or any(
        cat1.get(c) for c in RELEVANT_HARM_CATEGORIES
    )
    keyword_hit = bool(KEYWORD_PATTERN.search(row["prompt"] or ""))
    return category_hit, keyword_hit, (category_hit or keyword_hit)


def _select_safety_contrast_rows(ds):
    """Row-selection logic shared by both preference directions ("safer" and "better"):
    the relevance filter (category/keyword union) plus the "exactly one response flagged
    safe" genuine-contrast filter. This selection is IDENTICAL regardless of which
    direction the resulting pairs use -- T_ctrl's control arm requires the *same* PKU rows
    as T, only the chosen/rejected assignment differs (see build_safety_pairs).

    Returns (contrast_rows, funnel) where contrast_rows is a list of dicts:
        {"idx": PKU row index, "row": the row, "safe_idx": 0 or 1 (the safety-derived
         chosen index), "better_idx": 0 or 1 (row["better_response_id"])}
    """
    n_input = len(ds)
    missing = REQUIRED_PKU_FIELDS - set(ds.column_names)
    if missing:
        raise ValueError(f"PKU-SafeRLHF schema missing expected fields: {missing}")

    cat_hits = 0
    kw_hits = 0
    relevant_idx = []
    for i, row in enumerate(ds):
        c, k, combined = row_is_relevant(row)
        if c:
            cat_hits += 1
        if k:
            kw_hits += 1
        if combined:
            relevant_idx.append(i)
    n_relevant = len(relevant_idx)
    n_removed_stage1 = n_input - n_relevant

    both_safe = 0
    both_unsafe = 0
    contrast_rows = []
    for i in relevant_idx:
        row = ds[i]
        s0, s1 = row["is_response_0_safe"], row["is_response_1_safe"]
        if s0 is None or s1 is None:
            raise ValueError(f"Row {i}: is_response_*_safe field is None -- malformed record")
        if s0 and s1:
            both_safe += 1
            continue
        if (not s0) and (not s1):
            both_unsafe += 1
            continue
        # exactly one safe -> genuine safety contrast
        safe_idx = 0 if s0 else 1
        better_idx = row["better_response_id"]
        if better_idx not in (0, 1):
            raise ValueError(f"Row {i}: better_response_id is not 0/1 ({better_idx!r}) -- malformed record")
        if not row["prompt"] or not row["response_0"] or not row["response_1"]:
            raise ValueError(f"Row {i}: empty prompt/response text in a kept safety-contrast row")
        contrast_rows.append({"idx": i, "row": row, "safe_idx": safe_idx, "better_idx": better_idx})

    n_contrast = len(contrast_rows)
    n_discarded_no_contrast = n_relevant - n_contrast
    assert n_discarded_no_contrast == both_safe + both_unsafe

    n_same_direction = sum(1 for c in contrast_rows if c["safe_idx"] == c["better_idx"])
    frac_same_direction = n_same_direction / n_contrast if n_contrast else None

    funnel = {
        "stage": "safety_pairs",
        "n_input_rows": n_input,
        "category_hit_count": cat_hits,
        "keyword_hit_count": kw_hits,
        "n_relevant_after_union_filter": n_relevant,
        "n_removed_by_relevance_filter": n_removed_stage1,
        "n_both_safe_discarded": both_safe,
        "n_both_unsafe_discarded": both_unsafe,
        "n_final_safety_contrast_pairs_M": n_contrast,
        # T_ctrl weak-control diagnostic (coordinator's explicit request): how many of the
        # selected rows have safer_response_id == better_response_id, i.e. would produce
        # the IDENTICAL chosen/rejected pair under both preference directions. A large
        # fraction here would make T_ctrl a weak control (T and T_ctrl would be trained on
        # near-identical data despite the "different preference direction" framing).
        "n_rows_where_safer_eq_better_direction": n_same_direction,
        "frac_rows_where_safer_eq_better_direction": frac_same_direction,
    }
    return contrast_rows, funnel


def build_safety_pairs(seed: int, system_prompt: str, direction: str = "safer"):
    """direction="safer" (default, unchanged behaviour): chosen = the response flagged
    safe by is_response_*_safe -- this is what T trains on, and is byte-identical to the
    pre-existing behaviour (verified by hash in main()).
    direction="better": chosen = response_{better_response_id} (PKU's helpfulness
    annotation) for the SAME selected rows -- this is T_ctrl's contrast arm. Row selection
    (_select_safety_contrast_rows) is identical between directions by construction, so
    "same 4,924 PKU rows, only the preference direction differs" is a structural
    guarantee, not just a claim.
    """
    if direction not in ("safer", "better"):
        raise ValueError(f"direction must be 'safer' or 'better', got {direction!r}")
    if not RAW_PKU_DIR.exists():
        raise FileNotFoundError(f"Raw PKU-SafeRLHF data not found at {RAW_PKU_DIR}")
    ds = load_from_disk(str(RAW_PKU_DIR))["train"]

    contrast_rows, funnel = _select_safety_contrast_rows(ds)
    funnel = dict(funnel)  # copy so mutating below doesn't alias a shared dict
    funnel["direction"] = direction

    pairs = []
    for c in contrast_rows:
        row = c["row"]
        chosen_idx = c["safe_idx"] if direction == "safer" else c["better_idx"]
        rejected_idx = 1 - chosen_idx
        chosen_text = row[f"response_{chosen_idx}"]
        rejected_text = row[f"response_{rejected_idx}"]
        pairs.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["prompt"]},
                ],
                "chosen": [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": rejected_text}],
            }
        )

    import random

    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    return pairs, funnel


def build_helpful_pairs(seed: int, system_prompt: str, exclude_safety_inversions: bool = False):
    """Build the psychocounsel_pref helpfulness pairs.

    exclude_safety_inversions (default False, preserves the original/existing behaviour and
    the on-disk data/processed/pref_helpful.jsonl exactly as already produced): when True,
    drops rows where rejected_safety_rating > chosen_safety_rating (662 of 34,329 rows) --
    the Methodological Safeguards v2 rule 2 requirement that B2 (helpfulness-only DPO) is a
    clean arm, not one nudged toward less-safe outputs by rows whose "rejected" response was
    independently rated *safer* than the "chosen" one. Used by scripts/train_dpo.py, which
    calls this function directly (in-memory, from the raw dataset) rather than reading a
    separately-filtered file, so no shared file under data/processed/ is regenerated/
    overwritten by the DPO track while other tracks may depend on its current bytes.
    """
    if not RAW_PREF_DIR.exists():
        raise FileNotFoundError(f"Raw psychocounsel_pref data not found at {RAW_PREF_DIR}")
    ds = load_from_disk(str(RAW_PREF_DIR))["train"]
    n_input = len(ds)

    missing = REQUIRED_PREF_FIELDS - set(ds.column_names)
    if missing:
        raise ValueError(f"psychocounsel_pref schema missing expected fields: {missing}")

    pairs = []
    inversion_count = 0
    excluded_count = 0
    for i, row in enumerate(ds):
        if not row["question"] or not row["chosen"] or not row["rejected"]:
            raise ValueError(f"Row {i}: empty question/chosen/rejected text")
        is_inverted = row["rejected_safety_rating"] > row["chosen_safety_rating"]
        if is_inverted:
            inversion_count += 1
        if exclude_safety_inversions and is_inverted:
            excluded_count += 1
            continue
        pairs.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["question"]},
                ],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
            }
        )
    n_output = len(pairs)
    expected_output = n_input - excluded_count
    if n_output != expected_output:
        raise AssertionError(
            f"pref_helpful row count changed unexpectedly: input={n_input} "
            f"excluded={excluded_count} output={n_output} expected={expected_output}"
        )
    if not exclude_safety_inversions and n_output != n_input:
        raise AssertionError(
            f"pref_helpful row count changed unexpectedly: input={n_input} output={n_output}"
        )

    rnd = __import__("random").Random(seed)
    rnd.shuffle(pairs)

    funnel = {
        "stage": "helpful_pairs",
        "n_input_rows": n_input,
        "n_output_rows": n_output,
        "n_rejected_safety_gt_chosen_safety": inversion_count,
        "exclude_safety_inversions_applied": exclude_safety_inversions,
        "n_excluded_for_safety_inversion": excluded_count,
    }
    return pairs, funnel


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_length_stats(pairs, tokenizer):
    import numpy as np

    prompt_chosen_lens = []
    prompt_rejected_lens = []
    for p in pairs:
        pc = tokenizer.apply_chat_template(
            p["prompt"] + p["chosen"], tokenize=True, return_dict=True, add_generation_prompt=False
        )
        pr = tokenizer.apply_chat_template(
            p["prompt"] + p["rejected"], tokenize=True, return_dict=True, add_generation_prompt=False
        )
        prompt_chosen_lens.append(len(pc["input_ids"]))
        prompt_rejected_lens.append(len(pr["input_ids"]))

    def summarize(lens):
        arr = np.array(lens)
        return {
            "n": int(arr.size),
            "min": int(arr.min()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(arr.max()),
            "n_over_2048": int((arr > MAX_LEN).sum()),
        }

    return {
        "prompt_plus_chosen": summarize(prompt_chosen_lens),
        "prompt_plus_rejected": summarize(prompt_rejected_lens),
    }


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def leakage_check(safety_prompts, redteam_dir: Path):
    if not redteam_dir.exists():
        return {
            "redteam_dir_present": False,
            "note": "data/redteam absent at run time -- leakage check MUST be re-run before the 31 Jul freeze.",
        }
    redteam_texts = []
    for fp in redteam_dir.rglob("*.jsonl"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # best-effort extraction of prompt text from unknown redteam schema
                for key in ("prompt", "text", "input", "user_prompt", "messages"):
                    if key in obj:
                        val = obj[key]
                        if isinstance(val, str):
                            redteam_texts.append(val)
                        elif isinstance(val, list):
                            for turn in val:
                                if isinstance(turn, dict) and "content" in turn:
                                    # only user-authored turns are candidate leaked prompts
                                    if turn.get("role") in (None, "user", "system"):
                                        redteam_texts.append(turn["content"])
    norm_redteam = {normalize_text(t) for t in redteam_texts}
    norm_safety = {normalize_text(t) for t in safety_prompts}
    exact_overlap = norm_redteam & norm_safety

    near_dupes = 0
    if norm_redteam and norm_safety:
        rt_list = list(norm_redteam)
        for sp in norm_safety:
            for rt in rt_list:
                if sp == rt:
                    continue
                ratio = difflib.SequenceMatcher(None, sp, rt).ratio()
                if ratio > 0.9:
                    near_dupes += 1
                    break

    files_seen = [str(fp.name) for fp in redteam_dir.rglob("*.jsonl")]
    is_dev_fixture_only = all("fixture" in name.lower() or "dev" in name.lower() for name in files_seen) if files_seen else False
    return {
        "redteam_dir_present": True,
        "files_checked": files_seen,
        "n_redteam_prompts_found": len(redteam_texts),
        "n_exact_overlap": len(exact_overlap),
        "n_near_duplicate": near_dupes,
        "note": (
            "Only dev/fixture file(s) present, not the frozen suite -- this check MUST be "
            "re-run against the real frozen red-team suite after 31 Jul."
            if is_dev_fixture_only
            else "Re-run this check again after the suite freezes on 31 Jul to be safe."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--direction",
        type=str,
        choices=["safer", "better"],
        default="safer",
        help="Preference direction for the PKU safety-contrast rows. 'safer' (default) is "
        "the existing, unchanged behaviour: chosen = the response flagged safe -- writes "
        "data/processed/pref_safety.jsonl exactly as before (byte-identical; verified by "
        "hash, not just by code inspection). 'better' emits T_ctrl's control pairs on the "
        "SAME rows with chosen = response_{better_response_id} instead, writing to a "
        "separate file (data/processed/pref_safety_ctrl.jsonl) -- it never touches "
        "pref_safety.jsonl or pref_helpful.jsonl.",
    )
    args = ap.parse_args()

    system_prompt = load_system_prompt()

    if args.direction == "better":
        safety_ctrl_pairs, safety_ctrl_funnel = build_safety_pairs(args.seed, system_prompt, direction="better")
        write_jsonl(OUT_SAFETY_CTRL, safety_ctrl_pairs)
        report = {
            "seed": args.seed,
            "direction": "better",
            "safety_ctrl_funnel": safety_ctrl_funnel,
            "out_safety_ctrl_path": str(OUT_SAFETY_CTRL),
            "out_safety_ctrl_sha256": sha256_of_file(OUT_SAFETY_CTRL),
            "note": "T_ctrl control pairs only -- pref_safety.jsonl and pref_helpful.jsonl "
            "were not written or modified by this invocation.",
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    safety_pairs, safety_funnel = build_safety_pairs(args.seed, system_prompt, direction="safer")
    helpful_pairs, helpful_funnel = build_helpful_pairs(args.seed, system_prompt)

    write_jsonl(OUT_SAFETY, safety_pairs)
    write_jsonl(OUT_HELPFUL, helpful_pairs)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    safety_len_stats = token_length_stats(safety_pairs, tokenizer)
    helpful_len_stats = token_length_stats(helpful_pairs, tokenizer)

    safety_prompt_texts = [p["prompt"][1]["content"] for p in safety_pairs]
    leakage = leakage_check(safety_prompt_texts, REDTEAM_DIR)

    report = {
        "seed": args.seed,
        "safety_funnel": safety_funnel,
        "helpful_funnel": helpful_funnel,
        "safety_token_len_stats": safety_len_stats,
        "helpful_token_len_stats": helpful_len_stats,
        "leakage_check": leakage,
        "out_safety_path": str(OUT_SAFETY),
        "out_safety_sha256": sha256_of_file(OUT_SAFETY),
        "out_helpful_path": str(OUT_HELPFUL),
        "out_helpful_sha256": sha256_of_file(OUT_HELPFUL),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
