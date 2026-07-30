"""
prepare_sft.py — Build SFT JSONL files from ESConv and CounselChat raw datasets.

Feeds: Table 1/2 do not consume this directly, but it produces the B1 SFT training
data (LoRA SFT arm) that every downstream arm (B2/B3/T) is built on top of.

Outputs:
  data/processed/sft_esconv.jsonl
  data/processed/sft_counsel.jsonl
  data/processed/sft_train.jsonl   (merged, deterministically shuffled with --seed;
                                     the single SFT training file per the repo layout
                                     contract in CLAUDE.md)

Optional (--system-prompt-file PATH): prepends a {"role": "system", ...} message to
every conversation in all three output files, and copies the exact string used to
configs/system_prompt.txt so training and eval provably share it. The prompt text
itself is never invented here — it must be supplied by the caller.

data/raw/ is read-only and is never modified by this script.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import datasets
import ftfy

RAW_ESCONV = Path(r"C:\proj71\data\raw\esconv")
RAW_COUNSEL = Path(r"C:\proj71\data\raw\counsel_chat")
OUT_DIR = Path(r"C:\proj71\data\processed")
CONFIGS_DIR = Path(r"C:\proj71\configs")

OUT_ESCONV = OUT_DIR / "sft_esconv.jsonl"
OUT_COUNSEL = OUT_DIR / "sft_counsel.jsonl"
OUT_TRAIN = OUT_DIR / "sft_train.jsonl"
SYSTEM_PROMPT_OUT = CONFIGS_DIR / "system_prompt.txt"

QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TOKEN_THRESHOLDS = (1024, 2048, 4096, 8192)

FFFD = "\ufffd"
# Classic UTF-8-decoded-as-Latin-1/CP1252 mojibake: "Ã<latin1-supplement-char>",
# "â€<latin1-supplement-char>" (curly quotes/dashes), "Â<latin1-supplement-char>"
# (stray non-breaking-space style artifacts).
MOJIBAKE_LATIN1_RE = re.compile(r"Ã[\u0080-\u00ff]|\u00e2\u0080[\u0080-\u00ff]|Â[\u0080-\u00ff]")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def norm_ws(s: str) -> str:
    return " ".join(s.split()).strip()


def s_or_empty(x) -> str:
    return "" if x is None else str(x)


def mojibake_counts(s: str):
    """Return (fffd_count, latin1_mojibake_char_count) for a string."""
    if not s:
        return 0, 0
    fffd = s.count(FFFD)
    matches = MOJIBAKE_LATIN1_RE.findall(s)
    latin1_chars = sum(len(m) for m in matches)
    return fffd, latin1_chars


# ---------------------------------------------------------------------------
# ESConv
# ---------------------------------------------------------------------------

def process_esconv_split(split_dataset, split_name: str):
    """Shared per-row ESConv dialog processing (json.loads, usr/sys -> user/assistant
    mapping, merge consecutive same-speaker turns, drop leading assistant turns).
    Used for both the SFT training split ('train') and, read-only, the validation
    split ('validation') for eval purposes (see scripts/eval_val_loss.py) so the
    exact same pipeline is applied everywhere -- no separate/duplicated logic.

    Return (list_of_records, report_dict).
    """
    n_input = len(split_dataset)
    if n_input == 0:
        raise ValueError(f"ESConv {split_name} split is empty.")

    records = []
    dropped_empty_or_no_assistant = 0
    dropped_reasons = []

    for i, row in enumerate(split_dataset):
        raw_text = row.get("text")
        if raw_text is None or not str(raw_text).strip():
            raise ValueError(f"ESConv {split_name} row {i}: empty/missing 'text' field.")
        try:
            obj = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"ESConv {split_name} row {i}: failed to json.loads 'text': {e}")

        dialog = obj.get("dialog")
        if dialog is None:
            raise ValueError(f"ESConv {split_name} row {i}: missing 'dialog' field.")

        # Merge consecutive same-speaker turns; map speaker -> role; raise on unknown speaker
        merged = []  # list of (role, [texts])
        for turn in dialog:
            speaker = turn.get("speaker")
            text = turn.get("text", "")
            if speaker == "usr":
                role = "user"
            elif speaker == "sys":
                role = "assistant"
            else:
                raise ValueError(
                    f"ESConv {split_name} row {i}: unrecognized speaker value {speaker!r}"
                )
            if merged and merged[-1][0] == role:
                merged[-1][1].append(text)
            else:
                merged.append((role, [text]))

        messages = [
            {"role": role, "content": norm_ws("\n".join(texts))}
            for role, texts in merged
        ]

        # Drop leading assistant turns so conversation starts with user
        while messages and messages[0]["role"] == "assistant":
            messages.pop(0)

        has_assistant = any(m["role"] == "assistant" for m in messages)
        if not messages or not has_assistant:
            dropped_empty_or_no_assistant += 1
            dropped_reasons.append(i)
            continue

        records.append({"messages": messages})

    report = {
        "input_rows": n_input,
        "output_rows": len(records),
        "dropped_empty_or_no_assistant": dropped_empty_or_no_assistant,
        "split_used": split_name,
        "split_note": "validation and test splits reserved, not used for SFT"
        if split_name == "train"
        else "eval-only split, never used for training",
    }
    return records, report


def process_esconv():
    """Return (list_of_records, report_dict) for the ESConv TRAIN split (SFT data)."""
    if not RAW_ESCONV.exists():
        raise FileNotFoundError(f"ESConv raw path not found: {RAW_ESCONV}")
    dd = datasets.load_from_disk(str(RAW_ESCONV))
    if "train" not in dd:
        raise ValueError(f"ESConv DatasetDict has no 'train' split: {list(dd.keys())}")
    return process_esconv_split(dd["train"], "train")


def process_esconv_validation():
    """Return (list_of_records, report_dict) for the ESConv VALIDATION split.
    Read-only, eval-only -- never used for training. Same pipeline as process_esconv()
    via process_esconv_split()."""
    if not RAW_ESCONV.exists():
        raise FileNotFoundError(f"ESConv raw path not found: {RAW_ESCONV}")
    dd = datasets.load_from_disk(str(RAW_ESCONV))
    if "validation" not in dd:
        raise ValueError(f"ESConv DatasetDict has no 'validation' split: {list(dd.keys())}")
    return process_esconv_split(dd["validation"], "validation")


# ---------------------------------------------------------------------------
# CounselChat: load + mojibake repair, then dedup, then message-building
# ---------------------------------------------------------------------------

def load_and_clean_counsel():
    """Load CounselChat, run mojibake detection/repair and the empty-answer drop.
    Returns (candidates, report) where candidates is a list of dicts:
      {idx, qid, title_s, qtext_s, answer_s, upvotes, views}
    ready for dedup + message construction.
    """
    if not RAW_COUNSEL.exists():
        raise FileNotFoundError(f"CounselChat raw path not found: {RAW_COUNSEL}")

    dd = datasets.load_from_disk(str(RAW_COUNSEL))
    if "train" not in dd:
        raise ValueError(f"CounselChat DatasetDict has no 'train' split: {list(dd.keys())}")
    train = dd["train"]
    n_input = len(train)
    if n_input == 0:
        raise ValueError("CounselChat train split is empty.")

    field_anomalies = []
    dropped_empty_answer_ids = []

    before_fffd_rows = 0
    before_fffd_chars = 0
    before_latin1_rows = 0
    before_latin1_chars = 0

    candidates = []

    for i, row in enumerate(train):
        qid = row.get("questionID")
        title = row.get("questionTitle")
        qtext = row.get("questionText")
        answer = row.get("answerText")
        upvotes = row.get("upvotes")
        views = row.get("views")

        title_raw = s_or_empty(title)
        qtext_raw = s_or_empty(qtext)
        answer_raw = s_or_empty(answer)

        # --- mojibake: before-fix counts across all three fields ---
        row_fffd = 0
        row_latin1 = 0
        for field_raw in (title_raw, qtext_raw, answer_raw):
            f, l = mojibake_counts(field_raw)
            row_fffd += f
            row_latin1 += l
        if row_fffd > 0:
            before_fffd_rows += 1
            before_fffd_chars += row_fffd
        if row_latin1 > 0:
            before_latin1_rows += 1
            before_latin1_chars += row_latin1

        # --- mojibake: recoverable-pattern repair via ftfy ---
        title_fixed = ftfy.fix_text(title_raw)
        qtext_fixed = ftfy.fix_text(qtext_raw)
        answer_fixed = ftfy.fix_text(answer_raw)

        title_s = title_fixed.strip()
        qtext_s = qtext_fixed.strip()

        if not title_s and not qtext_s:
            raise ValueError(
                f"CounselChat row {i} (questionID={qid}): both questionTitle and "
                "questionText are empty/None — hard error."
            )

        if title is None or (isinstance(title, str) and title.strip() == ""):
            field_anomalies.append((qid, "questionTitle missing/empty, used empty string"))
        if qtext is None or (isinstance(qtext, str) and qtext.strip() == ""):
            field_anomalies.append((qid, "questionText missing/empty, used empty string"))

        if not answer_fixed.strip():
            dropped_empty_answer_ids.append(qid)
            continue

        candidates.append(
            {
                "idx": i,
                "qid": qid,
                "title_s": title_s,
                "qtext_s": qtext_s,
                "answer_s": answer_fixed.strip(),
                "upvotes": upvotes,
                "views": views,
            }
        )

    # --- mojibake: residual (unrecoverable) U+FFFD after ftfy, among surviving rows ---
    residual_ids = []
    for c in candidates:
        f_t, _ = mojibake_counts(c["title_s"])
        f_q, _ = mojibake_counts(c["qtext_s"])
        f_a, _ = mojibake_counts(c["answer_s"])
        if (f_t + f_q + f_a) > 0:
            residual_ids.append(c["idx"])

    residual_count = len(residual_ids)
    residual_pct = 100.0 * residual_count / n_input if n_input else 0.0

    mojibake_decision = "none"
    mojibake_dropped_qids = []
    if residual_count > 0:
        if residual_pct < 2.0:
            mojibake_decision = "drop"
            drop_idx_set = set(residual_ids)
            kept = []
            for c in candidates:
                if c["idx"] in drop_idx_set:
                    mojibake_dropped_qids.append(c["qid"])
                else:
                    kept.append(c)
            candidates = kept
        else:
            mojibake_decision = "strip"
            residual_idx_set = set(residual_ids)
            for c in candidates:
                if c["idx"] in residual_idx_set:
                    c["title_s"] = c["title_s"].replace(FFFD, "")
                    c["qtext_s"] = c["qtext_s"].replace(FFFD, "")
                    c["answer_s"] = c["answer_s"].replace(FFFD, "")

    report = {
        "input_rows": n_input,
        "dropped_empty_answer_count": len(dropped_empty_answer_ids),
        "dropped_empty_answer_questionIDs": dropped_empty_answer_ids,
        "field_anomalies": field_anomalies,
        "mojibake_before": {
            "rows_with_fffd": before_fffd_rows,
            "chars_fffd": before_fffd_chars,
            "rows_with_latin1_pattern": before_latin1_rows,
            "chars_latin1_pattern": before_latin1_chars,
        },
        "mojibake_after_ftfy_residual": {
            "rows_with_residual_fffd_or_pattern": residual_count,
            "pct_of_input_rows": round(residual_pct, 3),
            "decision": mojibake_decision,
            "dropped_questionIDs": mojibake_dropped_qids if mojibake_decision == "drop" else [],
        },
        "split_used": "train (only split available)",
    }
    return candidates, report


def dedup_mode_type(s: str):
    if s in ("all", "first"):
        return s
    m = re.fullmatch(r"cap:(\d+)", s)
    if m:
        n = int(m.group(1))
        if n < 1:
            raise argparse.ArgumentTypeError("cap:N must have N >= 1")
        return ("cap", n)
    raise argparse.ArgumentTypeError(
        f"invalid --counsel-dedup value: {s!r} (choices: all, first, cap:N)"
    )


def dedup_mode_label(mode) -> str:
    return mode if isinstance(mode, str) else f"cap:{mode[1]}"


def dedup_counsel(candidates, mode):
    """Group candidates by questionID, report the answers-per-question distribution,
    then select the subset per `mode`. Returns (selected_candidates, distribution_dict,
    dropped_count)."""
    groups = defaultdict(list)
    for c in candidates:
        groups[c["qid"]].append(c)

    distribution = Counter(len(v) for v in groups.values())

    if mode == "all":
        selected = list(candidates)
    elif mode == "first":
        seen = set()
        selected = []
        for c in candidates:
            if c["qid"] not in seen:
                seen.add(c["qid"])
                selected.append(c)
    elif isinstance(mode, tuple) and mode[0] == "cap":
        n = mode[1]
        keep_idx = set()
        for qid, group in groups.items():
            ranked = sorted(
                group,
                key=lambda c: (-(c["upvotes"] if c["upvotes"] is not None else -1),
                                -(c["views"] if c["views"] is not None else -1)),
            )
            for c in ranked[:n]:
                keep_idx.add(c["idx"])
        # preserve original dataset order for the audit file
        selected = [c for c in candidates if c["idx"] in keep_idx]
    else:
        raise ValueError(f"Unknown --counsel-dedup mode: {mode!r}")

    dropped_count = len(candidates) - len(selected)
    return selected, dict(sorted(distribution.items())), dropped_count


def build_counsel_records(selected_candidates):
    records = []
    for c in selected_candidates:
        user_content = norm_ws(c["title_s"] + "\n\n" + c["qtext_s"])
        assistant_content = norm_ws(c["answer_s"])
        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        )
    return records


# ---------------------------------------------------------------------------
# Token statistics (Qwen2.5-7B-Instruct tokenizer, local cache only)
# ---------------------------------------------------------------------------

def load_qwen_tokenizer():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, local_files_only=True)
    except Exception as e:
        return None, str(e)
    return tok, None


def token_stat_summary(lengths):
    arr = np.array(lengths, dtype=np.int64)
    summary = {
        "n": int(arr.size),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }
    for t in TOKEN_THRESHOLDS:
        summary[f"n_gt_{t}"] = int((arr > t).sum())
    return summary


def compute_token_stats(esconv_records, counsel_records, tokenizer):
    """Per-conversation total token counts (apply_chat_template, no system prompt,
    add_generation_prompt=False) plus each source's share of total assistant-turn
    tokens (assistant message content tokenized alone, summed)."""
    def conv_lengths_and_assistant_tokens(records):
        conv_lengths = []
        assistant_tokens = 0
        for r in records:
            enc = tokenizer.apply_chat_template(
                r["messages"], tokenize=True, return_dict=True, add_generation_prompt=False
            )
            conv_lengths.append(len(enc["input_ids"]))
            for m in r["messages"]:
                if m["role"] == "assistant":
                    ids = tokenizer(m["content"], add_special_tokens=False)["input_ids"]
                    assistant_tokens += len(ids)
        return conv_lengths, assistant_tokens

    esconv_lengths, esconv_assistant_tokens = conv_lengths_and_assistant_tokens(esconv_records)
    counsel_lengths, counsel_assistant_tokens = conv_lengths_and_assistant_tokens(counsel_records)

    total_assistant_tokens = esconv_assistant_tokens + counsel_assistant_tokens
    share = {
        "esconv_assistant_tokens": esconv_assistant_tokens,
        "counsel_assistant_tokens": counsel_assistant_tokens,
        "total_assistant_tokens": total_assistant_tokens,
        "esconv_pct": round(100.0 * esconv_assistant_tokens / total_assistant_tokens, 3)
        if total_assistant_tokens else None,
        "counsel_pct": round(100.0 * counsel_assistant_tokens / total_assistant_tokens, 3)
        if total_assistant_tokens else None,
    }

    return {
        "esconv": token_stat_summary(esconv_lengths),
        "counsel": token_stat_summary(counsel_lengths),
        "assistant_token_share": share,
    }


# ---------------------------------------------------------------------------
# System prompt handling
# ---------------------------------------------------------------------------

def read_system_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"--system-prompt-file not found: {path}")
    raw = path.read_bytes().decode("utf-8")
    # Strip exactly one trailing newline sequence (a text file's terminator is not
    # part of the intended prompt content); everything else is preserved verbatim.
    if raw.endswith("\r\n"):
        raw = raw[:-2]
    elif raw.endswith("\n") or raw.endswith("\r"):
        raw = raw[:-1]
    if not raw.strip():
        raise ValueError(f"--system-prompt-file {path} is empty after trimming — refusing to use a blank system prompt.")
    return raw


def add_system_prompt(records, system_text: str):
    out = []
    for r in records:
        new_messages = [{"role": "system", "content": system_text}] + list(r["messages"])
        out.append({"messages": new_messages})
    return out


# ---------------------------------------------------------------------------
# Merge, validate, write
# ---------------------------------------------------------------------------

def write_jsonl(records, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_starts_with_user(messages, idx, context):
    if not messages or not isinstance(messages, list):
        raise ValueError(f"{context}: record {idx} missing/invalid 'messages'")
    start = 1 if messages[0]["role"] == "system" else 0
    if start >= len(messages) or messages[start]["role"] != "user":
        raise ValueError(
            f"{context}: record {idx} does not start with a user role "
            f"(after optional leading system message)"
        )


def merge_and_shuffle(esconv_records, counsel_records, seed: int):
    """Concatenate the two per-source record lists and deterministically shuffle
    them with random.Random(seed). Validate every record has {"messages": [...]}
    starting with a user role (optionally preceded by one system message) before
    writing — fail loudly on any violation."""
    combined = list(esconv_records) + list(counsel_records)
    expected_total = len(esconv_records) + len(counsel_records)
    if len(combined) != expected_total:
        raise RuntimeError(
            f"Merge sanity check failed: combined len {len(combined)} != "
            f"expected {expected_total}"
        )

    for i, r in enumerate(combined):
        validate_starts_with_user(r.get("messages"), i, "sft_train merge")

    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic shuffle of sft_train.jsonl")
    parser.add_argument(
        "--counsel-dedup",
        type=dedup_mode_type,
        default="cap:2",
        help="CounselChat answers-per-question dedup: all | first | cap:N (default cap:2)",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Path to a text file whose exact contents are prepended as a system "
        "message to every conversation in all three output files. If omitted, no "
        "system message is added (current/default behavior).",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ESConv processing")
    print("=" * 70)
    esconv_records, esconv_report = process_esconv()
    if not esconv_records:
        raise RuntimeError("ESConv produced zero output records — aborting, not writing empty file.")
    print(json.dumps(esconv_report, indent=2))

    print()
    print("=" * 70)
    print("CounselChat: load, mojibake repair, empty-answer drop")
    print("=" * 70)
    candidates, counsel_load_report = load_and_clean_counsel()
    if not candidates:
        raise RuntimeError("CounselChat produced zero surviving rows before dedup — aborting.")
    print(json.dumps(counsel_load_report, indent=2))

    print()
    print("=" * 70)
    print(f"CounselChat dedup (mode={dedup_mode_label(args.counsel_dedup)})")
    print("=" * 70)
    selected, answers_per_question_dist, dedup_dropped = dedup_counsel(candidates, args.counsel_dedup)
    counsel_records = build_counsel_records(selected)
    if not counsel_records:
        raise RuntimeError("CounselChat produced zero output records after dedup — aborting.")
    counsel_report = dict(counsel_load_report)
    counsel_report["counsel_dedup_mode"] = dedup_mode_label(args.counsel_dedup)
    counsel_report["answers_per_question_distribution"] = answers_per_question_dist
    counsel_report["rows_before_dedup"] = len(candidates)
    counsel_report["rows_dropped_by_dedup"] = dedup_dropped
    counsel_report["output_rows"] = len(counsel_records)
    print(json.dumps(
        {
            "answers_per_question_distribution": answers_per_question_dist,
            "rows_before_dedup": len(candidates),
            "rows_dropped_by_dedup": dedup_dropped,
            "output_rows": len(counsel_records),
        },
        indent=2,
    ))

    print()
    print("=" * 70)
    print("Token statistics (Qwen2.5-7B-Instruct tokenizer, local cache only)")
    print("=" * 70)
    tokenizer, tok_err = load_qwen_tokenizer()
    if tokenizer is None:
        print(f"SKIPPED: tokenizer not available locally ({tok_err}). No token stats computed.")
        token_stats = None
    else:
        token_stats = compute_token_stats(esconv_records, counsel_records, tokenizer)
        print(json.dumps(token_stats, indent=2))

    # --- optional system prompt ---
    system_text = None
    if args.system_prompt_file is not None:
        print()
        print("=" * 70)
        print("System prompt injection")
        print("=" * 70)
        system_text = read_system_prompt(args.system_prompt_file)
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        SYSTEM_PROMPT_OUT.write_text(system_text, encoding="utf-8")
        print(f"Read system prompt from {args.system_prompt_file}, copied to {SYSTEM_PROMPT_OUT}")
        esconv_records = add_system_prompt(esconv_records, system_text)
        counsel_records = add_system_prompt(counsel_records, system_text)
    else:
        print()
        print("No --system-prompt-file supplied: writing without a system message (default).")

    write_jsonl(esconv_records, OUT_ESCONV)
    print(f"Wrote {len(esconv_records)} records -> {OUT_ESCONV}")
    write_jsonl(counsel_records, OUT_COUNSEL)
    print(f"Wrote {len(counsel_records)} records -> {OUT_COUNSEL}")

    print()
    print("=" * 70)
    print("Merge -> sft_train.jsonl")
    print("=" * 70)
    merged = merge_and_shuffle(esconv_records, counsel_records, args.seed)
    expected = len(esconv_records) + len(counsel_records)
    if len(merged) != expected:
        raise RuntimeError(
            f"sft_train.jsonl row count mismatch: got {len(merged)}, expected {expected} "
            f"({len(esconv_records)} esconv + {len(counsel_records)} counsel)"
        )
    write_jsonl(merged, OUT_TRAIN)
    print(
        json.dumps(
            {
                "esconv_rows": len(esconv_records),
                "counsel_rows": len(counsel_records),
                "merged_rows": len(merged),
                "seed": args.seed,
                "system_prompt_applied": system_text is not None,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(merged)} records -> {OUT_TRAIN}")


if __name__ == "__main__":
    main()
