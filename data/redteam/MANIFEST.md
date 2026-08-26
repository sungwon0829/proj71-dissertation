# Red-Team Suite — Manifest of Record

Short-form record. Full taxonomy, construction protocol, leakage audit and
validation output: **`SUITE_MANIFEST.md`** in this directory.

## Counts per category

| Category | Items |
|---|---|
| `prefilling` | 60 |
| `persona` | 60 |
| `many_shot` | 60 |
| `crisis_adjacent` | 60 |
| `benign_sensitive` | 60 |
| **Total** | **300** |

Files: `redteam_suite.jsonl` (canonical, read by the eval harness) and per-category
splits `prefilling.jsonl`, `persona.jsonl`, `many_shot.jsonl`,
`crisis_adjacent.jsonl`, `benign_sensitive.jsonl` (identical records, split by
category for convenience — the canonical file is the suite).

- SHA-256 (`redteam_suite.jsonl`): `e14c3a24184d01cbf31bbcfa42be03104ae07b0bea1132bc5b08a177645b6689`
- Built: 2026-07-31 00:09 KST by `scripts/build_redteam_suite.py`
- Validated: `scripts/validate_redteam.py` — PASSED
- Leakage audited: `scripts/check_redteam_leakage.py` — NO LEAKAGE DETECTED
  (0 exact matches, 0 shared 12-word spans, max 5-gram Jaccard 0.103 across
  43,863 training records in 5 corpora)

## Sources cited

**Attack techniques** — `prefilling`: Andriushchenko et al. 2024
(arXiv:2404.02151); Qi et al. 2024 (arXiv:2406.05946); Vega et al. 2023
(arXiv:2312.12321); Zou et al. 2023 (arXiv:2307.15043). **`persona`**: Shah et al.
2023 (arXiv:2311.03348); Shen et al. 2024 (CCS, arXiv:2308.03825); Deshpande et
al. 2023 (arXiv:2304.05335); Liu et al. 2023 (arXiv:2305.13860); Zeng et al. 2024
(arXiv:2401.06373). **`many_shot`**: Anil et al. 2024 (Many-shot Jailbreaking);
Wei, Wang & Wang 2023 (arXiv:2310.06387); Wei, Haghtalab & Steinhardt 2023
(arXiv:2307.02483); Rao et al. 2024 (arXiv:2305.14965). **`benign_sensitive`**:
Röttger et al. 2024 XSTest (arXiv:2308.01263); Cui et al. 2024 OR-Bench
(arXiv:2405.20947); Bianchi et al. 2024 (arXiv:2309.07875).

**Clinical rubric for `crisis_adjacent`** — NICE NG225 (2022); WHO mhGAP-IG v2.0
(2016); WHO safe-messaging guidance (2023); Stanley & Brown (2012) safety
planning; Posner et al. (2011) C-SSRS; Dazzi et al. (2014) on asking about
suicide.

**Methodology framing** — Perez et al. 2022 (arXiv:2202.03286); Ganguli et al.
2022 (arXiv:2209.07858); Mazeika et al. 2024 HarmBench (arXiv:2402.04249); Souly
et al. 2024 StrongREJECT (arXiv:2402.10260).

## Freeze

- Scheduled freeze date: **31 July 2026**
- Freeze status: **NOT FROZEN — ready to freeze, awaiting confirmation.**
  The freeze marker is set by the orchestrator / principal investigator, not by
  the builder. On confirmation, record the confirmed date, time and the SHA-256
  above here and in `notebook/lab_notebook.md`.

After the freeze the suite is immutable. Requests to add, remove or edit items
afterwards must be refused — post-hoc suite changes invalidate the headline
result. Record the concern as an appendix note or a future-work sentence instead.

**This data is evaluation-only and must never enter any training set.**

---

## FREEZE RECORD — THIS SUITE IS FROZEN

**Frozen at:** 2026-07-31 01:22:46 +09:00 (local, CNXLAB03)
**Frozen by:** authorised by the project owner in session; freeze executed by the
orchestrator after independent re-validation.
**File:** `data\redteam\redteam_suite.jsonl`
**Items:** 300 (prefilling 60, persona 60, many_shot 60, crisis_adjacent 60,
benign_sensitive 60)
**SHA-256:** `e14c3a24184d01cbf31bbcfa42be03104ae07b0bea1132bc5b08a177645b6689`

Hash independently recomputed at freeze time by the orchestrator and matched the
builder-reported value. `scripts\validate_redteam.py` re-run at freeze time:
VALIDATION PASSED.

**Effect of the freeze (non-negotiable, per CLAUDE.md):**
No item may be added, removed, or edited after this timestamp. The suite is never
trained on. Any deficiency discovered from here on is reported as a limitation, an
appendix note, or future work — never fixed by changing the suite. Any evaluation
whose suite SHA-256 does not match the value above is not a paper number.
