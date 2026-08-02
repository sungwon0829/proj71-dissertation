"""Negative controls for the judge-independence guard (CLAUDE.md safeguards v2, rule 1).

WHY THIS FILE EXISTS. `paper/methods.tex` states that negative controls for this guard are
exercised in the test suite, "including against a different revision of the same model".
As of 2026-08-02 that sentence had no artefact behind it: the guard existed in
`eval_score.py` and `apply_b3_filter.py`, but nothing tested that it actually fires. A
guard that has never been observed to fail closed is an assumption, not a control. This
file makes the Methods claim true.

The rule under test: the mechanism that FILTERS for B3 must never be the mechanism that
JUDGES attack success. If they are the same, the filter strips exactly what the judge looks
for, B3 scores near-zero ASR by construction rather than by merit, and the B3-vs-T
comparison -- the paper's entire claim -- becomes a tautology.

Comparison is deliberately on MODEL IDENTITY, ignoring revision and prompt: a different
prompt on the same weights is the same mechanism. Cases 3 and 4 below are the controls that
pin that behaviour down, because "we changed the prompt, so it's independent now" is the
most plausible way this rule gets talked around.

Run:  python scripts/test_judge_independence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_score import judge_independence_block  # noqa: E402
from apply_b3_filter import assert_filter_is_not_an_asr_judge  # noqa: E402

QWEN = "Qwen/Qwen2.5-7B-Instruct"
BEAVER = "PKU-Alignment/beaver-dam-7b"
BEAVER_REV = "3a5b70888869f04336d0d8344185ed74268ef896"

PASS, FAIL = [], []


def jm(model, role, revision="main", prompt="0" * 64):
    return {"actual_model": model, "actual_revision": revision,
            "prompt_sha256": prompt, "role": role}


def pinned_jmeta():
    """The configuration actually in force (pre-registration Revision 2)."""
    return {
        "behavioural": jm(QWEN, "asr_judge", prompt="a" * 64),
        "crisis": jm(QWEN, "behaviour_judge", prompt="b" * 64),
        "refusal": jm(QWEN, "over_refusal_judge", prompt="c" * 64),
    }


def check(name, fn, *, should_raise, must_mention=()):
    try:
        result = fn()
    except Exception as exc:
        if should_raise:
            missing = [s for s in must_mention if s.lower() not in str(exc).lower()]
            if missing:
                FAIL.append(f"{name}: raised, but message omits {missing}")
            else:
                PASS.append(f"{name}: raised as required -- {type(exc).__name__}")
        else:
            FAIL.append(f"{name}: raised but should NOT have -- {exc}")
        return None
    if should_raise:
        FAIL.append(f"{name}: DID NOT RAISE. The guard is not closing. Result: {result}")
    else:
        PASS.append(f"{name}: passed as required")
    return result


def main() -> int:
    print("=" * 88)
    print("NEGATIVE CONTROLS -- judge independence (CLAUDE.md safeguards v2, rule 1)")
    print("=" * 88)

    # --- 1. The real configuration must pass -----------------------------------------
    # If this fails, the pinned config itself is broken and nothing else matters.
    blk = check(
        "1. pinned config (filter=beaver-dam, ASR judge=Qwen behavioural)",
        lambda: judge_independence_block(
            {"b3_filter_mechanism_id": f"{BEAVER}@{BEAVER_REV}", "pin_status": "PINNED"},
            pinned_jmeta()),
        should_raise=False)
    if blk is not None:
        if blk.get("asr_judge_collisions"):
            FAIL.append(f"1a. pinned config reports collisions: {blk['asr_judge_collisions']}")
        else:
            PASS.append("1a. pinned config records zero ASR-judge collisions")
        # The two non-ASR judges share a base with each OTHER, not with the filter, so
        # nothing should land in acceptable_shared_base under the real config.
        if blk.get("acceptable_shared_base"):
            FAIL.append(f"1b. unexpected shared base with filter: {blk['acceptable_shared_base']}")
        else:
            PASS.append("1b. no judge shares a model with the filter")

    # --- 2. The Qwen fallback filter must be rejected ---------------------------------
    # This is the concrete case that matters: installing "a pinned Qwen2.5-7B classification
    # prompt" as the B3 filter, as CLAUDE.md's stale table proposed. The ASR judge is Qwen,
    # so this is the Revision-1 violation that already halted scoring once.
    check("2. Qwen2.5-7B prompted classifier as B3 filter (the stale-table fallback)",
          lambda: judge_independence_block(
              {"b3_filter_mechanism_id": QWEN, "pin_status": "PINNED"}, pinned_jmeta()),
          should_raise=True, must_mention=["independence", "tautology"])

    # --- 3. Different REVISION of the same model must still collide -------------------
    check("3. same model, different revision (revision must be ignored)",
          lambda: judge_independence_block(
              {"b3_filter_mechanism_id": f"{QWEN}@some-other-sha", "pin_status": "PINNED"},
              pinned_jmeta()),
          should_raise=True, must_mention=["independence"])

    # --- 4. Different PROMPT on the same weights must still collide -------------------
    # "It performs a different task with a different pinned prompt" is not independence.
    jm4 = pinned_jmeta()
    jm4["behavioural"] = jm(QWEN, "asr_judge", prompt="f" * 64)
    check("4. same model, different pinned prompt (prompt must be ignored)",
          lambda: judge_independence_block(
              {"b3_filter_mechanism_id": QWEN, "pin_status": "PINNED"}, jm4),
          should_raise=True, must_mention=["independence"])

    # --- 5. A union is only as independent as its weakest member ----------------------
    jm5 = pinned_jmeta()
    jm5["content"] = jm(BEAVER, "asr_judge")          # second ASR judge collides with filter
    check("5. union ASR: one clean member, one colliding member",
          lambda: judge_independence_block(
              {"b3_filter_mechanism_id": BEAVER, "pin_status": "PINNED",
               "asr_combination": {"rule": "union", "judges": ["behavioural", "content"]}}, jm5),
          should_raise=True, must_mention=["independence"])

    # --- 6. Non-ASR judge sharing the filter model: allowed, but RECORDED -------------
    jm6 = pinned_jmeta()
    jm6["crisis"] = jm(BEAVER, "behaviour_judge")
    blk6 = check("6. non-ASR judge shares filter model (permitted, must be recorded)",
                 lambda: judge_independence_block(
                     {"b3_filter_mechanism_id": BEAVER, "pin_status": "PINNED"}, jm6),
                 should_raise=False)
    if blk6 is not None:
        if "crisis" in blk6.get("acceptable_shared_base", []):
            PASS.append("6a. shared base recorded, not silent")
        else:
            FAIL.append("6a. shared base NOT recorded -- the decision would be invisible")

    # --- 7. asr_combination must not smuggle in a non-ASR judge -----------------------
    # A judge listed in the union but not roled `asr_judge` would escape the collision check.
    check("7. asr_combination names a judge whose role is not asr_judge",
          lambda: judge_independence_block(
              {"b3_filter_mechanism_id": BEAVER, "pin_status": "PINNED",
               "asr_combination": {"rule": "union", "judges": ["behavioural", "refusal"]}},
              pinned_jmeta()),
          should_raise=True, must_mention=["role"])

    # --- 8. No ASR judge at all -------------------------------------------------------
    check("8. no judge has role asr_judge (ASR would be undefined)",
          lambda: judge_independence_block(
              {"b3_filter_mechanism_id": BEAVER, "pin_status": "PINNED"},
              {"crisis": jm(QWEN, "behaviour_judge")}),
          should_raise=True, must_mention=["asr_judge"])

    # --- 9. The same rule, enforced independently in the B3 build path ----------------
    # eval_score refuses to SCORE a violating config; apply_b3_filter must refuse to
    # BUILD the artefact, so a bad config cannot even produce generations.
    def b3cfg(filter_model, judge_model):
        return {
            "b3_filter": {"mechanism_id": filter_model, "backend": "flt"},
            "backends": {"flt": {"name_or_path": filter_model},
                         "jbk": {"name_or_path": judge_model}},
            "judges": {"behavioural": {"backend": "jbk", "role": "asr_judge"}},
        }

    check("9. apply_b3_filter refuses to BUILD when filter == ASR judge",
          lambda: assert_filter_is_not_an_asr_judge(b3cfg(QWEN, QWEN)),
          should_raise=True, must_mention=["independence"])
    check("10. apply_b3_filter permits the pinned, non-colliding config",
          lambda: assert_filter_is_not_an_asr_judge(b3cfg(BEAVER, QWEN)),
          should_raise=False)

    # --- report ------------------------------------------------------------------------
    print()
    for line in PASS:
        print(f"  [ok]   {line}")
    for line in FAIL:
        print(f"  [FAIL] {line}")
    print()
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nTHE INDEPENDENCE GUARD IS NOT SOUND. Do not score any arm until this passes.")
        return 1
    print("\nGuard fails closed on every violation tested, in both the scoring path and the\n"
          "B3 build path. Revision and prompt are correctly ignored when comparing mechanisms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
