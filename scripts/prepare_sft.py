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
import unicodedata
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
# CounselChat therapist-identity scrubber
#
# CounselChat's answerText fields end with the (non-anonymised) therapist's own
# sign-off ("Robin J. Landwehr, DBH, LPCC, NCC", "- Landwehr, DBH", "www.<their
# practice>.com", personal phone numbers, ...). therapistInfo / therapistURL give
# us ground truth for *that row's own* therapist, and we use ONLY that per-row
# ground truth to find and remove identity from the row's own answer -- never a
# global name blocklist, and never third-party names a therapist happens to
# mention (e.g. "As Sherry mentioned, ..." in someone else's answer is left
# alone; it is not this row's own identity).
#
# Method, in order:
#  1. Derive (first_name, last_name) for the therapist from therapistInfo, cross-
#     checked against therapistURL (see derive_therapist_name / notebook write-up
#     for the full rationale). Falls back to therapistURL parsing, then to a tiny
#     manual-override table for the handful of cases where the source text is too
#     ambiguous for either automated path (documented inline).
#  2. Remove "First [Middle...] Last[, CREDENTIALS...]" wherever it appears in
#     that row's own answerText (case-insensitive, word-boundaried). Also a
#     narrower fallback: a bare last-name sign-off + credentials sitting at the
#     very end of the answer (first name dropped).
#  3. Remove personal phone numbers -- but NEVER a number that sits next to
#     crisis/hotline/lifeline language (verified against the raw corpus: national
#     and local crisis lines appear throughout and must never be stripped).
#  4. Remove practice/clinic URLs: (a) a URL the therapist self-declares inside
#     their own therapistInfo bio, (b) a URL whose host contains their own name,
#     (c) a URL immediately preceded by "my website/practice/site/..." in their
#     own answer. Generic reference/resource links (hotlines, articles, other
#     people's blogs) are deliberately left untouched -- that is therapeutic
#     content, not identity.
#  5. No email addresses were found anywhere in the raw corpus (verified below).
#
# This is a content-preserving scrub: only the identity span itself is removed,
# never replaced with a placeholder the model could learn to emit, and sign-off
# language ("Be well,", "Warmly,") is left in place.
# ---------------------------------------------------------------------------

TITLE_PREFIXES = {"dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "rev", "rev.", "prof", "prof."}
_CAMEL_PROTECT_MIN_RUN = 3  # do not treat a lower->upper transition as a name/tagline
                            # boundary unless the run since the last space/hyphen is
                            # this long (protects "Ph"|"D", "La"|"Rose", "Mc"|"Donald")
_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")

# Handful of therapistInfo strings where neither the camelCase/comma heuristic nor
# the therapistURL fallback resolves correctly (verified by hand against the raw
# corpus -- see notebook/pending_scrub.md for the full derivation trace):
#   - "JanaLee" is itself internally capitalised (no separating space), which is
#     indistinguishable, by rule, from the name/tagline fusion boundary we rely on
#     everywhere else.
#   - "Ilse de León" / "Eric Ström": the therapistURL slug drops the accented
#     character entirely, so the URL cross-check can't confirm the correct name.
#   - "Cory Ian Shafer": three-token name (first, middle, last) with no comma
#     before the credential block, so the generic segmentation can't tell "Ian"
#     apart from the surname.
#   - "Christina McGrath Fair": double-barrelled surname with no comma and no
#     camelCase boundary before the tagline (a curly-quote opens the tagline).
_MANUAL_NAME_OVERRIDES = {
    "JanaLee WagnerHope through life's complications. ": ("JanaLee", "Wagner"),
    "Ilse de LeónYou matter!": ("Ilse", "León"),
    "Eric Ström, JD, MA, LMHCAttorney & Licensed Mental Health Counselor": ("Eric", "Ström"),
    "Cory Ian Shafer LPCPsychotherapist, Jungian, Hypnotherapy": ("Cory", "Shafer"),
    'Christina McGrath Fair"Enlightenment is when a wave realizes it is the ocean." -Thich Nhat Hanh': ("Christina", "Fair"),
}

# therapistURL slug tokens that are credential/qualifier codes, not name tokens
# (used only for the therapistURL fallback path).
_CREDENTIAL_STOPLIST = {
    "ma", "ms", "msw", "mft", "lmft", "lpc", "lpcc", "lpca", "lcsw", "licsw", "lmsw",
    "ncc", "phd", "psyd", "edd", "dbh", "lmhc", "mhc", "rn", "bcba", "cadc", "atr",
    "cst", "ceds", "bc", "tmh", "cctp", "cchi", "cch", "s", "llc", "pllc", "pc", "jd",
    "mdiv", "dmin", "np", "otr", "l", "clt", "caps", "dcc", "emdr", "lcac", "ads",
    "mph", "mbe", "abpp", "cbist", "casac", "mac", "cip", "laadc", "ccmi", "lcdc",
    "lac", "rmft", "msc", "cht", "med",
}


def _norm_cmp(s: str) -> str:
    """Diacritic-, case-, and punctuation-insensitive comparison key."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _camel_first_segment(s: str) -> str:
    """therapistInfo fuses "Name" directly onto a free-text tagline with no
    separator ("Robin Landwehr, DBH, LPCC, NCCMental Health in a Primary Care
    Setting"). Return the text up to the first genuine name/tagline boundary
    (a lowercase letter immediately followed by an uppercase letter), ignoring
    boundaries whose preceding run (since the last space or hyphen) is shorter
    than _CAMEL_PROTECT_MIN_RUN -- these are credential-initial or compound-name
    fragments ("Ph"|"D", "La"|"Rose"), not real boundaries."""
    run_start = 0
    for i in range(1, len(s)):
        if s[i - 1].islower() and s[i].isupper() and (i - run_start) >= _CAMEL_PROTECT_MIN_RUN:
            return s[:i]
        if s[i] in " -":
            run_start = i + 1
    return s


def _primary_derive_name(info_raw: str):
    info = ftfy.fix_text(info_raw or "").strip()
    seg = _camel_first_segment(info)
    if "," in seg:
        seg = seg.split(",", 1)[0]
    tokens = seg.strip().split()
    while tokens and tokens[0].lower() in TITLE_PREFIXES:
        tokens = tokens[1:]
    if not tokens:
        return None, None
    first = tokens[0].strip(".")
    last = tokens[-1].strip(".") if len(tokens) > 1 else first
    return first, last


def _slug_tokens(url: str):
    slug = url.rstrip("/").split("/")[-1]
    toks = [t for t in slug.split("-") if t]
    while toks and toks[0] in TITLE_PREFIXES:
        toks = toks[1:]
    return toks


def _slug_confirms(first: str, last: str, slug_toks) -> bool:
    """Confirm `last` appears in slug_toks at some position after `first`, but
    never search past the first credential/qualifier token (or a numeric
    disambiguator) -- otherwise a credential that happens to equal the (wrongly
    derived) `last` candidate, e.g. "...-lmhc-..." confirming a bad last="LMHC",
    would falsely validate a broken primary derivation (verified against the
    raw corpus: this exact failure mode occurred for "Barika Grayson LMHC")."""
    if not slug_toks or _norm_cmp(slug_toks[0]) != _norm_cmp(first):
        return False
    boundary = len(slug_toks)
    for idx in range(1, len(slug_toks)):
        if slug_toks[idx] in _CREDENTIAL_STOPLIST or slug_toks[idx].isdigit():
            boundary = idx
            break
    last_n = _norm_cmp(last)
    for start in range(1, boundary):
        for k in range(1, 4):
            if start + k > boundary:
                continue
            if _norm_cmp("".join(slug_toks[start:start + k])) == last_n:
                return True
    return False


def derive_therapist_name(info_raw: str, url: str):
    """Return (first_name, last_name, source) for one therapistInfo/therapistURL
    pair. source is one of: manual_override, therapistInfo, url_fallback,
    url_fallback_single, url_fallback_no_last, non_personal (a CounselChat
    account for an organisation, not an individual -- name scrubbing is skipped
    and this is reported explicitly), unresolved (no usable slug at all)."""
    if info_raw in _MANUAL_NAME_OVERRIDES:
        first, last = _MANUAL_NAME_OVERRIDES[info_raw]
        return first, last, "manual_override"

    first, last, source = _derive_therapist_name_uncapped(info_raw, url)

    # A last name shorter than 3 characters (e.g. a bare initial like "C.") is
    # too generic to use as a corpus-wide removal anchor -- it produces false
    # positives against unrelated text (verified against the raw corpus: a
    # therapist signing only "Stephanie C." caused dozens of coincidental
    # matches elsewhere). Skip name-based scrubbing for these rather than ship
    # a regex known to over-match; reported explicitly via this source label.
    if last is not None and len(last) < 3:
        return None, None, "insufficient_confidence_short_last_name"
    return first, last, source


def _derive_therapist_name_uncapped(info_raw: str, url: str):
    first, last = _primary_derive_name(info_raw)
    slug_toks = _slug_tokens(url)

    if (first and _NAME_TOKEN_RE.match(first) and last and _NAME_TOKEN_RE.match(last)
            and _slug_confirms(first, last, slug_toks)):
        return first, last, "therapistInfo"

    if not slug_toks:
        return None, None, "unresolved"
    f = slug_toks[0]
    if not re.match(r"^[a-z]", f):
        return None, None, "non_personal"
    rest = slug_toks[1:]
    if not rest:
        return f.capitalize(), f.capitalize(), "url_fallback_single"
    if rest[0] in _CREDENTIAL_STOPLIST or rest[0].isdigit():
        return f.capitalize(), f.capitalize(), "url_fallback_no_last"
    return f.capitalize(), rest[0].capitalize(), "url_fallback"


# --- removal patterns --------------------------------------------------------

_MIDDLE = r"(?:\s+[A-Z][A-Za-z'\.\-]*){0,3}"  # optional middle name(s)/initial(s)
_CREDS = r"(?:[,.]?\s*[A-Z]{1,6}(?:[/\-][A-Za-z]{1,4})?){0,6}"  # trailing credential list

_HOTLINE_KEYWORDS_RE = re.compile(
    r"hotline|lifeline|crisis|talkline|trevor|prevention|suicide|911|988|helpline|"
    r"text\s*line|textline",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_SELF_REF_RE = re.compile(
    r"\bmy\s+(?:website|practice|site|blog|page|clinic|office)\b", re.IGNORECASE
)


def _name_pattern(first: str, last: str):
    return re.compile(
        rf"\b{re.escape(first)}{_MIDDLE}\s+{re.escape(last)}\b{_CREDS}", re.IGNORECASE
    )


def scrub_name(answer: str, first: str, last: str):
    """Remove "First [Middle...] Last[, CREDENTIALS...]" wherever it appears.
    Deliberately does NOT also strip a bare last-name-only sign-off: tested
    against the raw corpus, a solo-surname rule was found to false-positive on
    quote attributions that happen to end an answer (e.g. a different
    therapist's answer ending "...~Brene Brown", where "Brown" coincidentally
    matches another therapist's own surname) while adding zero verified true
    positives over the combined first+last pattern above."""
    return _name_pattern(first, last).subn("", answer)


def scrub_phone(answer: str):
    """Remove phone numbers except ones sitting next to crisis/hotline language
    (verified against the raw corpus: national/local crisis lines are referenced
    throughout CounselChat answers and must never be stripped -- this is a
    therapy-support safety dataset)."""
    out = []
    n_removed = 0
    last_end = 0
    for m in _PHONE_RE.finditer(answer):
        ctx = answer[max(0, m.start() - 80): m.end() + 130]
        if _HOTLINE_KEYWORDS_RE.search(ctx):
            continue
        out.append(answer[last_end:m.start()])
        last_end = m.end()
        n_removed += 1
    out.append(answer[last_end:])
    return "".join(out), n_removed


def scrub_email(answer: str):
    return _EMAIL_RE.subn("", answer)


def _url_host(u: str) -> str:
    u = re.sub(r"^https?://", "", u, flags=re.IGNORECASE)
    u = re.sub(r"^www\.", "", u, flags=re.IGNORECASE)
    return re.split(r"[/?#\s]", u, 1)[0]


def _declared_domain(info_raw: str):
    """A URL the therapist embedded in their own therapistInfo bio -- self-
    declared ground truth, safe to treat as personal."""
    m = _URL_RE.search(ftfy.fix_text(info_raw or ""))
    return _norm_cmp(_url_host(m.group())) if m else None


def scrub_url(answer: str, info_raw: str, first, last):
    """Remove practice/clinic URLs grounded in (a) the therapist's own
    self-declared URL, (b) their own name appearing in the host, or (c) an
    explicit in-answer self-reference ("my website/practice/..."). Generic
    reference links (hotlines, articles, other people's sites) are left alone
    -- see notebook/pending_scrub.md for the documented residual list of
    practice URLs this does *not* catch (URLs with neither the therapist's own
    name nor a self-referential phrase nearby)."""
    dom = _declared_domain(info_raw)
    first_n = _norm_cmp(first) if first else ""
    last_n = _norm_cmp(last) if last else ""
    n_removed = 0
    out = []
    last_end = 0
    for m in _URL_RE.finditer(answer):
        host_n = _norm_cmp(_url_host(m.group()))
        personal = (
            (dom and (host_n == dom or dom in host_n))
            or (len(last_n) >= 4 and last_n in host_n)
            or (len(first_n) >= 5 and first_n in host_n)
            or bool(_SELF_REF_RE.search(answer[max(0, m.start() - 60):m.start()]))
        )
        if personal:
            out.append(answer[last_end:m.start()])
            last_end = m.end()
            n_removed += 1
    out.append(answer[last_end:])
    return "".join(out), n_removed


def cleanup_trailing_punctuation(answer: str) -> str:
    """After removing a sign-off from the very end of an answer, strip any
    now-dangling connector punctuation ("Be well," -> "Be well") without ever
    touching a real sentence-ending period."""
    return re.sub(r"[\s,;:\-–~]+$", "", answer)


def scrub_therapist_identity(answer_fixed: str, info_raw: str, url: str, name_cache: dict):
    """Scrub one CounselChat answerText of its own therapist's identity.
    Returns (scrubbed_answer, per_row_stats). name_cache memoises
    derive_therapist_name() per distinct therapistInfo string."""
    if info_raw not in name_cache:
        name_cache[info_raw] = derive_therapist_name(info_raw, url)
    first, last, source = name_cache[info_raw]

    stats = {"name_removed": 0, "phone_removed": 0, "url_removed": 0}
    ans = answer_fixed
    if first is not None:
        ans, stats["name_removed"] = scrub_name(ans, first, last)
    ans, stats["phone_removed"] = scrub_phone(ans)
    ans, n_email = scrub_email(ans)
    stats["email_removed"] = n_email
    ans, stats["url_removed"] = scrub_url(ans, info_raw, first, last)
    ans = cleanup_trailing_punctuation(ans)
    return ans, stats


def verify_no_residual_names(records, name_cache, context: str):
    """Corpus-wide diagnostic sweep: does ANY known therapist's own (first,
    last) name pattern appear ANYWHERE in the (already-scrubbed) output?

    This is broader than the actual scrub (which only ever removes a row's OWN
    author name from its OWN answer) and broader than the same-row hard gate
    enforced inside load_and_clean_counsel() (which already guarantees, by
    construction, that no row's own identity survives in its own answer --
    that check raises immediately if it ever fails). Any hit reported here is
    therefore necessarily a *different* author's name coincidentally appearing
    in someone else's text -- e.g. a quote attribution ("...as Fred Rogers once
    said...") or a shared surname -- not a scrub failure. Reported for full
    auditability, not treated as fatal."""
    residual = []
    for r in records:
        for m in r["messages"]:
            if m["role"] != "assistant":
                continue
            text = m["content"]
            for info_raw, (first, last, source) in name_cache.items():
                if first is None:
                    continue
                mm = _name_pattern(first, last).search(text)
                if mm:
                    ctx = text[max(0, mm.start() - 60): mm.end() + 20]
                    residual.append((first, last, ctx))
    if residual:
        print(f"Cross-reference name-string hits in {context}: {len(residual)} "
              "(same-row self-identity is already guaranteed impossible -- see "
              "load_and_clean_counsel()'s per-row hard gate; these are a "
              "different author's name coincidentally present in someone "
              "else's text, e.g. a quote attribution):")
        for first, last, ctx in residual:
            print(f"  {first} {last}: ...{ctx!r}...")
    else:
        print(f"Residual name check ({context}): 0 hits across {len(records)} records.")
    return len(residual)


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
    dropped_empty_after_scrub_ids = []

    before_fffd_rows = 0
    before_fffd_chars = 0
    before_latin1_rows = 0
    before_latin1_chars = 0

    candidates = []

    # Therapist-identity scrub bookkeeping (see scrub_therapist_identity() /
    # derive_therapist_name() above). name_cache memoises the derived name per
    # distinct therapistInfo string so it is computed once, not per-row.
    name_cache = {}
    scrub_rows_with_name_hit = 0
    scrub_total_name_removals = 0
    scrub_total_phone_removals = 0
    scrub_total_email_removals = 0
    scrub_total_url_removals = 0
    scrub_before_lens = []
    scrub_after_lens = []
    scrub_big_loss = []  # (qid, therapistInfo, before_len, after_len, snippet)

    for i, row in enumerate(train):
        qid = row.get("questionID")
        title = row.get("questionTitle")
        qtext = row.get("questionText")
        answer = row.get("answerText")
        upvotes = row.get("upvotes")
        views = row.get("views")
        therapist_info = row.get("therapistInfo")
        therapist_url = row.get("therapistURL")

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

        answer_before_scrub = answer_fixed.strip()
        answer_scrubbed, scrub_stats = scrub_therapist_identity(
            answer_before_scrub, therapist_info, therapist_url, name_cache
        )

        # Same-row hard gate: this row's OWN derived name must not survive in
        # its OWN scrubbed answer. Fail loudly and immediately rather than
        # discover a leak later -- this is the actual privacy guarantee; the
        # corpus-wide sweep in verify_no_residual_names() is a broader,
        # informational diagnostic on top of this.
        own_first, own_last, _own_source = name_cache[therapist_info]
        if own_first is not None and _name_pattern(own_first, own_last).search(answer_scrubbed):
            raise RuntimeError(
                f"Therapist-identity scrub failed on its OWN row: questionID={qid}, "
                f"therapistInfo={therapist_info!r} — derived name "
                f"({own_first!r}, {own_last!r}) still present in the scrubbed answer. "
                "Aborting rather than shipping a known leak."
            )

        before_len = len(answer_before_scrub)
        after_len = len(answer_scrubbed.strip())
        scrub_before_lens.append(before_len)
        scrub_after_lens.append(after_len)
        if scrub_stats["name_removed"]:
            scrub_rows_with_name_hit += 1
        scrub_total_name_removals += scrub_stats["name_removed"]
        scrub_total_phone_removals += scrub_stats["phone_removed"]
        scrub_total_email_removals += scrub_stats["email_removed"]
        scrub_total_url_removals += scrub_stats["url_removed"]
        if before_len > 0 and (before_len - after_len) / before_len > 0.20:
            scrub_big_loss.append((qid, therapist_info, before_len, after_len, answer_scrubbed[:200]))

        if not answer_scrubbed.strip():
            # Scrubbing must never silently swallow an answer -- if it does,
            # drop it loudly and count it (requirement: fail loudly, never
            # silently skip/truncate).
            dropped_empty_after_scrub_ids.append(qid)
            continue

        candidates.append(
            {
                "idx": i,
                "qid": qid,
                "title_s": title_s,
                "qtext_s": qtext_s,
                "answer_s": answer_scrubbed.strip(),
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

    # --- therapist-identity scrub summary ---
    name_sources = Counter(source for (_, _, source) in name_cache.values())
    non_personal_therapists = [
        info for info, (f, l, source) in name_cache.items() if source == "non_personal"
    ]
    unresolved_therapists = [
        info for info, (f, l, source) in name_cache.items() if source == "unresolved"
    ]
    b = np.array(scrub_before_lens, dtype=np.int64) if scrub_before_lens else np.array([0])
    a = np.array(scrub_after_lens, dtype=np.int64) if scrub_after_lens else np.array([0])
    scrub_report = {
        "distinct_therapists": len(name_cache),
        "distinct_therapist_name_sources": dict(name_sources),
        "non_personal_therapist_accounts": non_personal_therapists,
        "unresolved_therapists": unresolved_therapists,
        "answers_with_name_removed": scrub_rows_with_name_hit,
        "total_name_span_removals": scrub_total_name_removals,
        "total_phone_removals": scrub_total_phone_removals,
        "total_email_removals": scrub_total_email_removals,
        "total_url_removals": scrub_total_url_removals,
        "dropped_empty_after_scrub_count": len(dropped_empty_after_scrub_ids),
        "dropped_empty_after_scrub_questionIDs": dropped_empty_after_scrub_ids,
        "length_before": {
            "min": int(b.min()), "median": float(np.median(b)),
            "p95": float(np.percentile(b, 95)), "max": int(b.max()), "mean": float(b.mean()),
        },
        "length_after": {
            "min": int(a.min()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": int(a.max()), "mean": float(a.mean()),
        },
        "rows_losing_over_20pct_length": len(scrub_big_loss),
        "rows_losing_over_20pct_length_examples": [
            {"questionID": qid, "therapistInfo": info, "before_len": bl, "after_len": al, "after_snippet": snip}
            for qid, info, bl, al, snip in scrub_big_loss[:10]
        ],
    }

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
        "therapist_identity_scrub": scrub_report,
        "split_used": "train (only split available)",
    }
    return candidates, report, name_cache


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
    candidates, counsel_load_report, therapist_name_cache = load_and_clean_counsel()
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
    print("CounselChat therapist-identity scrub: residual verification")
    print("=" * 70)
    verify_no_residual_names(
        counsel_records, therapist_name_cache, "sft_counsel.jsonl (post-scrub, post-dedup)"
    )

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

    print()
    print("=" * 70)
    print("Final residual verification: sft_train.jsonl (as written to disk)")
    print("=" * 70)
    verify_no_residual_names(
        merged, therapist_name_cache, "sft_train.jsonl (final, on-disk)"
    )


if __name__ == "__main__":
    main()
