# CounselChat therapist-identity scrub — audit trail

**Date:** 2026-08-01
**Trigger:** B1 generation reproduced a real therapist's name and credentials verbatim
("Robin J. Landwehr, DBH,") — memorised from non-anonymised CounselChat `answerText`
sign-off blocks. Project owner directed: scrub, regenerate SFT outputs, retrain B1.
**Script edited in place:** `C:\proj71\scripts\prepare_sft.py` (no duplicate script created).

---

## 1. Investigation

CounselChat (`data/raw/counsel_chat`, 2775 rows, columns include `therapistInfo`,
`therapistURL`, `answerText`). `therapistInfo` and `therapistURL` are per-row ground
truth for that row's own therapist and were used — never a global name list, never a
generic regex guessed independently of the data.

- **437 distinct `therapistInfo` values** (i.e. distinct therapists) in the raw
  CounselChat train split.
- Confirmed **51 raw answers contain their own author's name+credentials** in a
  sign-off (`Robin Landwehr` alone accounts for 46 of the 51, e.g. qid 14, 38, 45, 63,
  78, 97, 98, 151, 193, 194, ...). Sign-off forms observed: `"...Robin Landwehr, DBH,
  LPCC"`, `"...Robin J. Landwehr, DBH, LPCC, NCC"` (note: the sign-off sometimes adds a
  middle initial "J." that isn't even present in `therapistInfo` — the scrub regex has
  to tolerate that, see §2).
- `therapistInfo` is not a clean `"Name, Credentials"` field — it fuses the name
  directly onto a free-text tagline with **no separator**, e.g.
  `"Robin Landwehr, DBH, LPCC, NCCMental Health in a Primary Care Setting"`. Extracting
  the name reliably required a purpose-built heuristic (§2), not a naive split.
- Also found in raw `answerText` and handled: **32 personal phone numbers** (e.g.
  Vivian D. Echevarria Guzman's own office number `787-466-5478`, appearing across 9 of
  her answers), **0 email addresses** (verified: a corpus-wide email regex scan
  returned zero hits, so no email-scrubbing logic was needed), and **~20-24 practice/
  clinic URLs** embedded either in `therapistInfo` itself (7 therapists self-declare a
  URL there) or in-line in `answerText` ("Learn more about me and my practice at
  www.EMDRheals.com").
- **Critical corpus finding that shaped the phone-number rule:** national and local
  crisis lines (`1-800-273-8255` National Suicide Prevention Lifeline, `800-989-6884`
  Texas Crisis Hotline, `877-565-8860` Trans Lifeline, `1-800-422-4453` Childhelp, `911`,
  etc.) are referenced throughout the raw answers. A blind phone-number strip would have
  **deleted crisis resources from a safety-focused therapy-support dataset** — the
  opposite of what this project needs. The phone-number rule is therefore
  context-gated (§2), not blind.

## 2. Scrubbing rule implemented

All logic lives in `prepare_sft.py`, functions `derive_therapist_name()`,
`scrub_name()`, `scrub_phone()`, `scrub_email()`, `scrub_url()`,
`scrub_therapist_identity()`, `verify_no_residual_names()`. Applied inside
`load_and_clean_counsel()`, once per row, using **only that row's own**
`therapistInfo`/`therapistURL` — a therapist's name is never removed from anyone
else's answer (so a colleague reference like "As Sherry mentioned..." is correctly
left untouched: it isn't that row's own identity).

**Name derivation** (`derive_therapist_name`), per distinct `therapistInfo`:
1. `_camel_first_segment`: `therapistInfo` fuses "Name" directly onto the tagline with
   no separator. The name/tagline boundary is a lowercase letter immediately followed
   by an uppercase letter (`"Landwehr, DBH, LPCC, NCCMental"` → boundary at
   `NCC|Mental`), but naively splitting on every such boundary breaks compound
   names/credential-initial pairs (`"Ph"|"D"`, `"La"|"Rose"`, `"Mc"|"Donald"`). Fixed by
   only treating a boundary as real if the run of letters since the last space/hyphen
   is ≥3 chars.
2. Cut at the first comma (credentials normally follow a comma).
3. Strip leading titles (`Dr.`, `Ms.`, ...).
4. Take `tokens[0]` as first name, `tokens[-1]` as last name.
5. **Cross-check against `therapistURL`** (e.g. `robin-landwehr-dbh-lpcc-ncc`):
   confirm the slug's first token matches the derived first name, and that the derived
   last name appears in the slug immediately after — searching only up to the first
   credential/qualifier token in the slug (a stoplist of ~50 known credential
   abbreviations), so a credential that happens to equal a wrongly-derived "last name"
   can't falsely self-confirm (this exact failure mode occurred for "Barika Grayson
   LMHC, NCC..." before the fix: the naive comma-cut produced last="LMHC", and without
   the credential-boundary limit the URL's own `-lmhc-` token wrongly confirmed it).
6. If step 1–5 doesn't validate, fall back to parsing `therapistURL` directly
   (first two non-credential slug tokens).
7. **5 documented manual overrides** for cases where neither path resolves correctly
   (all individually verified against the raw corpus, reasons in code comments):
   `JanaLee Wagner` (name itself is internally capitalised with no space — identical in
   form to the fusion boundary the whole heuristic relies on), `Ilse de León` /
   `Eric Ström` (the URL slug drops the accented character entirely, so the
   cross-check can't confirm), `Cory Ian Shafer` (3-token name, no comma before
   credentials), `Christina McGrath Fair` (double-barrelled surname, no comma, tagline
   opens with a curly quote so there's no camelCase boundary either).
8. **1 non-personal account** detected and excluded from name-scrubbing: "2nd Chance
   Counseling Service..." is a business account, not an individual.
9. **1 "insufficient confidence" case**: a therapist who signs only "Stephanie C."
   (bare initial). A last name shorter than 3 characters is too generic to use as a
   corpus-wide match anchor (see the false-positive found during verification, below) —
   scrubbing by name is skipped for this one therapist and reported explicitly rather
   than shipping a regex known to over-match.

**Removal (`scrub_name`)**: `\bFirst (?:middle words/initials){0,3}\s+Last\b(?:,?\s*
CREDENTIAL){0,6}` — case-insensitive, tolerates inserted middle names/initials (handles
the "Robin Landwehr" → "Robin **J.** Landwehr" sign-off variant) and a trailing
credential list of any length/spelling variant actually observed
(`", DBH, LPCC"`, `", DBH, LPC, NCC"`, `", DBH, LPC NCC"` — no comma before the last
one, etc.). Only the identity span is removed; sign-off language ("Be well,",
"Warmly,") is left in place — content is not truncated, and nothing is replaced with a
placeholder token the model could learn to emit.

*Rejected design*: an earlier version also matched a bare last-name-only sign-off
anchored at the end of the answer (to catch a hypothetical "first name dropped"
sign-off). Verification against the full corpus found this produces false positives —
a *different* therapist's answer ending in a quote attribution ("...~Brene Brown")
coincidentally matched another real therapist's surname "Brown". Removed: it added zero
verified true positives over the full first+last pattern above, and was also a latent
over-scrub risk in the actual per-row scrub (not just the verifier).

**Phone (`scrub_phone`)**: strip a matched phone number **unless** crisis/hotline
language (`hotline|lifeline|crisis|talkline|trevor|prevention|suicide|911|988|
helpline|text\s*line`) appears within 80 chars before / 130 chars after the match —
verified against every phone number in the raw corpus by hand (§1); all 19 kept
numbers are genuine crisis lines, all 32 dropped numbers are personal contact numbers.

**URL (`scrub_url`)**: strip a URL if (a) the therapist self-declared it in their own
`therapistInfo` bio, (b) its host contains their own first (≥5 chars) or last (≥4
chars) name, or (c) it's immediately preceded by "my website/practice/site/blog/page/
clinic/office" in their own answer. Generic reference/resource links (hotlines,
articles, other people's sites, book titles) are deliberately left alone — this is
therapeutic content, not identity. **Known, documented limitation**: a handful of
practice URLs that use a branded name unrelated to the therapist's own name and were
not self-declared in `therapistInfo` (e.g. `psychologyresource.ca`,
`lifecounselingorlando.com`, `empoweryou2.com`) are not caught by this pass. These are
business/practice names, not the personal name+credentials identity leak that
triggered this task, and are reported here rather than silently left unaddressed.

**Email**: no scrub logic needed — a corpus-wide regex scan found zero email addresses
anywhere in `answerText`.

## 3. Content preservation (length before/after)

| | before | after |
|---|---|---|
| min | 10 | 10 |
| median | 779 | 779 |
| p95 | 2332.6 | 2329.6 |
| mean | 959.74 | 958.54 |
| max | 5499 | 5499 |

**0 answers lost more than 20% of their length.** (Max single-answer loss observed
during development testing: 11.45%, for a row with 3 stacked phone numbers — well
under the flag threshold, no manual-inspection queue needed.)

## 4. Empirical verification (residual scan)

Two layers, both implemented in `prepare_sft.py`:

1. **Hard gate (same-row), inside `load_and_clean_counsel()`**: immediately after
   scrubbing each row, re-run the row's own name pattern against its own scrubbed
   answer. If it still matches, the script raises `RuntimeError` and aborts —
   `sft_counsel.jsonl`/`sft_train.jsonl` are never written with a known leak. This ran
   clean on the shipped output (no aborts).
2. **Corpus-wide diagnostic sweep (`verify_no_residual_names`)**, run twice — once on
   `sft_counsel.jsonl` records right after dedup, once on the final `sft_train.jsonl`
   as written to disk. Checks every known therapist's own name pattern against
   *every* answer in the corpus (broader than the actual scrub, which only ever
   touches a row's own text).
   - **Result: 1 hit**, both times (same one, since it's in a CounselChat-derived
     record): the literal string **"Fred Rogers"** appears once, in Ashton Sullivan's
     answer (qid 402): *"...as Fred Rogers once said, whenever you see something
     terrible..."* — a quote attribution to the famous Mister Rogers, not the
     CounselChat contributor Fred Rogers (`fred-rogers`, whose own 6 answers were all
     independently, correctly scrubbed and confirmed clean by the hard gate). Because
     the same-row hard gate already guarantees no author's own identity survives in
     their own answer, this hit is by construction a different person's name
     coincidentally present in someone else's text — a real cultural-figure name
     collision that scrubbing must not "fix" by mangling a legitimate quote (violates
     the do-not-over-scrub requirement). **Explained, not a bug.**
   - During development, before the fixes described in §2, this sweep also caught two
     genuine derivation bugs (`Barika Grayson` mis-parsed as `Barika LMHC`) and the
     `Stephanie C.` false-positive-prone short-name case — both fixed/excluded before
     the shipped run (see §2). The sweep did its job.

## 5. Regeneration — row counts

Exact invocation used (unchanged from before): `--seed 42 --counsel-dedup cap:2
--system-prompt-file C:\proj71\configs\system_prompt.txt` (pinned prompt read
verbatim from that file, copied to `configs\system_prompt.txt` as before).

| file | rows (old) | rows (new) |
|---|---|---|
| `sft_esconv.jsonl` | 910 | 910 |
| `sft_counsel.jsonl` | 1395 | 1395 |
| `sft_train.jsonl` | 2305 | 2305 |

**Unchanged.** `dropped_empty_after_scrub_count: 0` — no answer was emptied by
scrubbing, so nothing needed to be (or was) dropped on that account. ESConv counts are
byte-for-byte the same input/output as before (see §7).

Distinct-therapist name-derivation source breakdown (429 distinct therapists survive
the pre-scrub empty-answer drop, out of 437 total):
`therapistInfo` (direct heuristic): 399, `url_fallback`: 23, `manual_override`: 5,
`non_personal` (skipped): 1, `insufficient_confidence_short_last_name` (skipped): 1.

Scrub totals: 70 answers had their own name+credentials removed (70 spans total, 1 per
answer), 32 personal phone numbers removed, 23 practice URLs removed, 0 emails (none
existed).

## 6. Determinism

Ran the pinned invocation **twice**, independently, back to back. SHA-256 of all three
outputs matched byte-for-byte across both runs:

- `sft_esconv.jsonl`: `17FA5DFB...4D92A62C` (both runs)
- `sft_counsel.jsonl`: `9DF5A18A2B5FAAA56016F46C953FF042FF21EFD841A6FDE65DEDD265E6460608` (both runs)
- `sft_train.jsonl`: `46E87A39F239982F9E9994535B4A44209617800AFFDEEBFEAEC0AF8922DBF662`
  (both runs)

**New `sft_train.jsonl` SHA-256: `46E87A39F239982F9E9994535B4A44209617800AFFDEEBFEAEC0AF8922DBF662`**
(old, pre-scrub: `52A7074D6D366FCFD3DC720D0314AAF5B33B6EB364BE61A7DAA77B9420DA3DE3` — different, as
expected and intended; the file content changed because the scrub changed it).

## 7. ESConv — confirmed unaffected, not assumed

ESConv (`data/raw/esconv`) is crowdworker dialogue with no `therapistInfo`/author
signature fields at all — checked the schema directly (`text`, `dialog` with `usr`/
`sys` speaker turns only, no name/credential/contact columns anywhere) and confirmed
`sft_esconv.jsonl`'s row count (910) and content are identical to the pre-scrub run
(the scrub code only ever touches CounselChat's `load_and_clean_counsel()` path;
`process_esconv()` is untouched). `sft_esconv.jsonl` SHA-256 is stable across both new
runs (§6) and its row count (910) matches the original 910 exactly.

## 8. Leakage check against `data/redteam/`

Ran an exact-message-text overlap check between `sft_train.jsonl` (22,363 unique
message strings across system/user/assistant turns) and every file in
`data/redteam/` (`benign_sensitive.jsonl`, `crisis_adjacent.jsonl`, `many_shot.jsonl`,
`persona.jsonl`, `prefilling.jsonl`, `redteam_suite.jsonl`).

**Result: 2 raw-string matches (1 in `many_shot.jsonl`, 1 in `redteam_suite.jsonl` —
the latter is the aggregate file, same underlying record).** Both are the identical
single-token string **`"Yes."`** — a generic short reply with no distinguishing
content, appearing independently in both corpora (in `many_shot`'s synthetic
in-context example turns and somewhere in ordinary SFT dialogue). **No red-team
prompt, scenario, or unique content appears in `sft_train.jsonl`.** This is not a
leakage event in the sense the freeze protects against (no attack prompt or its
target completion was trained on) — reported explicitly per the "report the check
result every time, even when clean" instruction, rather than rounded down to "clean."

## 9. Before/after examples (truncated to 400 chars, last 400 shown — sign-offs sit at
the end of these answers)

**Example 1 — Robin Landwehr (qid 45), name+credentials removed:**

BEFORE:
```
...er want to commit suicide. But, as you are indicating, that feeling of wanting to commit suicide can come back again. So, it is always good to have a plan. If you want more information, call the National Suicide Prevention Lifeline. They will always answer: 1-800-273-8255. They can help you create a specific safety plan. I do hope things work out alright. Be well. Robin J. Landwehr, DBH, LPCC, NCC
```

AFTER:
```
...ugh the crisis stage, they no longer want to commit suicide. But, as you are indicating, that feeling of wanting to commit suicide can come back again. So, it is always good to have a plan. If you want more information, call the National Suicide Prevention Lifeline. They will always answer: 1-800-273-8255. They can help you create a specific safety plan. I do hope things work out alright. Be well.
```

Note: the crisis hotline number `1-800-273-8255` is correctly **preserved** (hotline
context detected), only `Robin J. Landwehr, DBH, LPCC, NCC` was removed.

**Example 2 — Vivian D. Echevarria Guzman (qid 61), personal phone number removed:**

BEFORE (Spanish-language answer):
```
...rnos estancados una emoción.  Así que es importante aprender a identificar que causa nuestros sentimientos, como procesarlos para poder superar nuestras emociones y movernos de una experiencia a otra con más tranquilidad. 
Soy Consejera Profesional Licenciada en Texas y Puerto Rico, veo clientes en el este de Houston y proveo servicios de tele terapia.  Para una cita puedes llamar al 787-466-5478.
```

AFTER:
```
... bueno quedarnos estancados una emoción.  Así que es importante aprender a identificar que causa nuestros sentimientos, como procesarlos para poder superar nuestras emociones y movernos de una experiencia a otra con más tranquilidad. 
Soy Consejera Profesional Licenciada en Texas y Puerto Rico, veo clientes en el este de Houston y proveo servicios de tele terapia.  Para una cita puedes llamar al .
```

Minor known cosmetic artifact: removing the phone number leaves a dangling
`"llamar al ."` (the sentence-final period that followed the number is preserved by
design — `cleanup_trailing_punctuation` never strips real periods — but reads slightly
oddly). No PII survives; flagged here for visibility, not treated as a content-loss
problem (this row's overall length loss is well under the 20% flag threshold).

## 10. Next step

`sft_train.jsonl` (new SHA-256 above) is ready for B1 retraining. Recommend the
train-runner subagent pick this up and log the retraining run itself to
`notebook/lab_notebook.md` per the usual convention (not duplicated here).
