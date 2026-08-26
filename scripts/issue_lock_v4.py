"""One-shot issuer for configs/judges_pinned.lock.v4.json (Amendment 23).

Carries every instrument block (judges, b3_guardrail_filter, helpfulness_scorer) from
the v3 lock unchanged (deep-equality asserted at issuance), adds Amendment 23 to the
amendments-in-force enumeration, and records the post-hoc robustness seeds (B2/B3/T
seeds 2-3) in run_metadata WITHOUT touching the primary seed enumeration (B2/B3/T = [1],
Amendment 23 protection 1). judges.yaml's pin pointer is updated in tandem. v1, v2 and
v3 locks are never touched; their hashes are asserted before and after. Run once;
re-running overwrites v4 deterministically with the same content.

Modeled on scripts/issue_lock_v3.py (the Amendment 14a/21 legitimate re-pin path).
"""
import copy, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "configs/judges_pinned.lock.json"
V2 = ROOT / "configs/judges_pinned.lock.v2.json"
V3 = ROOT / "configs/judges_pinned.lock.v3.json"
V4 = ROOT / "configs/judges_pinned.lock.v4.json"
YAML = ROOT / "configs/judges.yaml"

V1_SHA = "444aa1b6022f4fec0032a7f56e160a23f3b99c280ead4242c687abd0787e3530"
V2_SHA = "4b478e0322a0f9fcf831082a32bfc2b4a5f50038e99604259bc86fd84a23e470"
# Expected sha of the v3 lock at supersession. Recomputed from disk below and asserted
# against this value; the recomputed value is what goes into the supersedes block.
V3_SHA_EXPECTED = "1c6572b63435e459150ace2be288d87448593909124167c350f4d8f6591fe57d"

AMENDMENTS_IN_FORCE = ["7a", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                       "17", "18", "18a", "19", "20", "21", "22", "23"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha256_file(V1) == V1_SHA, "v1 lock bytes changed -- refusing"
    assert sha256_file(V2) == V2_SHA, "v2 lock bytes changed -- refusing"
    v3_sha = sha256_file(V3)
    assert v3_sha == V3_SHA_EXPECTED, (
        f"v3 lock hashes to {v3_sha}, expected {V3_SHA_EXPECTED} -- v3 bytes changed, refusing")
    v3 = json.loads(V3.read_text(encoding="utf-8"))

    v4 = {}
    v4["_what_this_is"] = (
        "Judge/filter pin lock, VERSION 4. Issued 2026-08-27 under pre-registration "
        "Amendment 23 (seeds 2-3 reinstated as a post-hoc robustness check; a SIGHTED "
        "decision, protections enumerated in the amendment). NO INSTRUMENT CHANGED: the "
        "`judges`, `b3_guardrail_filter` and `helpfulness_scorer` blocks are carried from "
        "v3 unchanged (deep-equality asserted at issuance). What v4 adds: (i) Amendment 23 "
        "enters `preregistration_amendments_in_force`, so verify_judge_pin()'s staleness "
        "scan verifies again -- its refusal against v3 after Amendment 23 appeared was "
        "confirmed to fire before this lock was issued, as the guard's positive control; "
        "(ii) run_metadata records `posthoc_robustness_seeds` (B2/B3/T seeds 2-3) while "
        "the PRIMARY analysis stays seed 1 exactly as pinned (Amendment 23 protection 1). "
        "This lock SUPERSEDES v3; v1, v2 and v3 are retained unedited (Amendment 14a: a "
        "pinned artefact is never edited; corrections and successions live outside it).")
    v4["pin_status"] = "PINNED"
    v4["pin_date"] = v3["pin_date"]                      # 2026-08-01 -- not a re-pin of instruments
    v4["filter_pin_date"] = v3["filter_pin_date"]        # 2026-08-05
    v4["record_correction_date"] = v3["record_correction_date"]  # 2026-08-26 (Amendment 21)
    v4["posthoc_repin_date"] = "2026-08-27"
    v4["pinned_by"] = v3["pinned_by"]
    v4["authorized_by"] = ("Amendment 23 (2026-08-27, post-hoc robustness seeds; sighted "
                           "decision with pre-committed protections) via the Amendment "
                           "14a/21 legitimate re-pin path")
    v4["preregistration_revisions_in_force"] = v3["preregistration_revisions_in_force"]
    v4["preregistration_amendments_in_force"] = AMENDMENTS_IN_FORCE
    v4["immutable_after_pin"] = v3["immutable_after_pin"]
    v4["judges"] = copy.deepcopy(v3["judges"])
    v4["b3_guardrail_filter"] = copy.deepcopy(v3["b3_guardrail_filter"])
    v4["helpfulness_scorer"] = copy.deepcopy(v3["helpfulness_scorer"])
    v4["independence_statement"] = v3["independence_statement"]
    v4["labelling_caveat"] = v3["labelling_caveat"]
    v4["what_is_NOT_pinned_by_this_file"] = v3["what_is_NOT_pinned_by_this_file"]
    v4["_repin_note"] = v3["_repin_note"] + [
        "",
        "VERSION 4 ISSUED 2026-08-27 under Amendment 23. THE INSTRUMENTS ARE UNCHANGED:",
        "the `judges`, `b3_guardrail_filter` and `helpfulness_scorer` blocks are deep-equal",
        "to v3 (asserted at issuance by scripts/issue_lock_v4.py). Changes: Amendment 23",
        "added to preregistration_amendments_in_force (the staleness scan's refusal against",
        "v3 was verified to fire first, as the positive control); run_metadata gains",
        "posthoc_robustness_seeds (B2/B3/T = [2, 3], post-hoc robustness check, never",
        "pooled into the primary). The primary analysis remains seed 1, unchanged.",
    ]

    rm_v3 = v3["run_metadata"]
    rm = {}
    rm["_what_this_is"] = (
        rm_v3["_what_this_is"]
        + " Amendment 23 (2026-08-27) reinstates seeds 2-3 as a POST-HOC robustness check "
          "recorded in posthoc_robustness_seeds below; the primary enumeration in "
          "multi_seed_arms is unchanged.")
    rm["multi_seed_arms"] = {"B2": [1], "B3": [1], "T": [1]}
    rm["posthoc_robustness_seeds"] = {
        "B2": [2, 3],
        "B3": [2, 3],
        "T": [2, 3],
        "note": ("Amendment 23 (2026-08-27): a SIGHTED decision (seed-1 cross-arm results "
                 "existed and had been seen), reinstating seeds 2-3 as a clearly-labelled "
                 "POST-HOC robustness check. These seeds are NEVER pooled into the primary "
                 "analysis, which remains seed 1 exactly as pinned (protection 1); they "
                 "enter a separate, explicitly post-hoc Results panel, with whatever they "
                 "show reported (protection 3) and incomplete runs reported as "
                 "attempted-and-incomplete (protection 4). Over-refusal for these seeds is "
                 "the judge cross-check ONLY (kappa 0.074, stated wherever shown): the "
                 "Revision 4 human labels cover seed-1 B3/T responses only. B3 seed N is "
                 "derived from B2 seed N by apply_b3_filter.py, as always."),
    }
    for k in ("single_run_arms", "seed_semantics", "b3_derivation",
              "t_ctrl_is_single_seed", "evaluation_seed"):
        rm[k] = copy.deepcopy(rm_v3[k])
    rm["descope_note_amendment22"] = (
        rm_v3["descope_note_amendment22"]
        + " UPDATE (Amendment 23, 2026-08-27): 22.1 partially reversed -- seeds 2-3 "
          "reinstated as a post-hoc robustness check (see posthoc_robustness_seeds); the "
          "primary analysis remains seed 1 and the 'not measured' limitation is superseded "
          "by the measured spread where the runs complete.")
    rm["known_inconsistency_resolved"] = rm_v3["known_inconsistency_resolved"]
    assert set(rm) - {"posthoc_robustness_seeds"} == set(rm_v3), \
        "run_metadata keys drifted beyond the intended addition"
    v4["run_metadata"] = rm

    v4["supersedes"] = {
        "file": "configs/judges_pinned.lock.v3.json",
        "sha256_at_supersession": v3_sha,
        "reason": ("Amendment 23: re-pin to bring the amendment enumeration current and "
                   "record the post-hoc robustness seeds (B2/B3/T seeds 2-3). No judge, "
                   "filter or helpfulness instrument changed; the primary seed enumeration "
                   "is unchanged."),
        "chain": ["configs/judges_pinned.lock.json (v1, superseded 2026-08-05, sha "
                  + V1_SHA + ")",
                  "configs/judges_pinned.lock.v2.json (v2, superseded 2026-08-26, sha "
                  + V2_SHA + ")",
                  "configs/judges_pinned.lock.v3.json (v3, superseded 2026-08-27, sha "
                  + v3_sha + ")"],
    }

    # write with LF endings, no trailing platform conversion (.gitattributes -text)
    payload = json.dumps(v4, indent=2, ensure_ascii=False) + "\n"
    V4.write_bytes(payload.encode("utf-8"))

    # verify: instruments deep-equal to v3, old locks untouched
    v4_back = json.loads(V4.read_text(encoding="utf-8"))
    assert v4_back["judges"] == v3["judges"], "judges block drifted"
    assert v4_back["b3_guardrail_filter"] == v3["b3_guardrail_filter"], "filter block drifted"
    assert v4_back["helpfulness_scorer"] == v3["helpfulness_scorer"], \
        "helpfulness_scorer block drifted"
    assert v4_back["run_metadata"]["multi_seed_arms"] == {"B2": [1], "B3": [1], "T": [1]}, \
        "primary seed enumeration changed -- Amendment 23 protection 1 violated"
    assert sha256_file(V1) == V1_SHA and sha256_file(V2) == V2_SHA \
        and sha256_file(V3) == v3_sha
    v4_sha = sha256_file(V4)

    # update judges.yaml pin pointer in tandem (the sanctioned two-file edit)
    ytext = YAML.read_bytes().decode("utf-8")
    old_file = "pin_lock_file: configs/judges_pinned.lock.v3.json"
    new_file = "pin_lock_file: configs/judges_pinned.lock.v4.json"
    old_sha_line = 'pin_lock_sha256: "' + v3_sha + '"'
    new_sha_line = 'pin_lock_sha256: "' + v4_sha + '"'
    assert old_file in ytext and old_sha_line in ytext, "judges.yaml pin pointer not found"
    ytext = ytext.replace(old_file, new_file).replace(old_sha_line, new_sha_line)
    YAML.write_bytes(ytext.encode("utf-8"))

    print("v4 lock issued:", V4)
    print("v4 sha256:", v4_sha)
    print("v3 sha256 at supersession:", v3_sha)
    print("v1/v2/v3 verified untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
