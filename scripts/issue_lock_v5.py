"""One-shot issuer for configs/judges_pinned.lock.v5.json (Amendment 24).

Carries every instrument block (judges, b3_guardrail_filter, helpfulness_scorer) from
the v4 lock unchanged (deep-equality asserted at issuance), adds Amendment 24 to the
amendments-in-force enumeration, and records the exploratory ratio-ablation arms
(T_r050 / T_r200, Amendment 24.2) in run_metadata WITHOUT touching anything else in it:
run_metadata is a deep copy of v4's plus the single `exploratory_arms` key (asserted).
judges.yaml's pin pointer is updated in tandem. v1, v2, v3 and v4 locks are never
touched; their hashes are asserted before and after. Run once; re-running overwrites v5
deterministically with the same content.

Modeled on scripts/issue_lock_v4.py (the Amendment 14a/21 legitimate re-pin path).
"""
import copy, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "configs/judges_pinned.lock.json"
V2 = ROOT / "configs/judges_pinned.lock.v2.json"
V3 = ROOT / "configs/judges_pinned.lock.v3.json"
V4 = ROOT / "configs/judges_pinned.lock.v4.json"
V5 = ROOT / "configs/judges_pinned.lock.v5.json"
YAML = ROOT / "configs/judges.yaml"

V1_SHA = "444aa1b6022f4fec0032a7f56e160a23f3b99c280ead4242c687abd0787e3530"
V2_SHA = "4b478e0322a0f9fcf831082a32bfc2b4a5f50038e99604259bc86fd84a23e470"
V3_SHA = "1c6572b63435e459150ace2be288d87448593909124167c350f4d8f6591fe57d"
# Expected sha of the v4 lock at supersession. Recomputed from disk below and asserted
# against this value; the recomputed value is what goes into the supersedes block.
V4_SHA_EXPECTED = "8cabb12854df721b76dd4dd676c9cc3efa5774cef218fef8ae3151f79a3d9ea7"

AMENDMENTS_IN_FORCE = ["7a", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                       "17", "18", "18a", "19", "20", "21", "22", "23", "24"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha256_file(V1) == V1_SHA, "v1 lock bytes changed -- refusing"
    assert sha256_file(V2) == V2_SHA, "v2 lock bytes changed -- refusing"
    assert sha256_file(V3) == V3_SHA, "v3 lock bytes changed -- refusing"
    v4_sha = sha256_file(V4)
    assert v4_sha == V4_SHA_EXPECTED, (
        f"v4 lock hashes to {v4_sha}, expected {V4_SHA_EXPECTED} -- v4 bytes changed, refusing")
    v4 = json.loads(V4.read_text(encoding="utf-8"))

    v5 = {}
    v5["_what_this_is"] = (
        "Judge/filter pin lock, VERSION 5. Issued 2026-08-28 under pre-registration "
        "Amendment 24 (post-hoc exploratory analyses and the safety-pair ratio ablation; a "
        "SIGHTED decision, protections pre-specified in the amendment). NO INSTRUMENT "
        "CHANGED: the `judges`, `b3_guardrail_filter` and `helpfulness_scorer` blocks are "
        "carried from v4 unchanged (deep-equality asserted at issuance). What v5 adds: "
        "(i) Amendment 24 enters `preregistration_amendments_in_force`, so "
        "verify_judge_pin()'s staleness scan verifies again -- its refusal against v4 after "
        "Amendment 24 appeared was confirmed to fire before this lock was issued, as the "
        "guard's positive control; (ii) run_metadata gains `exploratory_arms` (T_r050 / "
        "T_r200, Amendment 24.2, post-hoc exploratory, seed 1 only, never primary, never "
        "Table 1 or Table 2) and is otherwise a deep copy of v4's (asserted). This lock "
        "SUPERSEDES v4; v1, v2, v3 and v4 are retained unedited (Amendment 14a: a pinned "
        "artefact is never edited; corrections and successions live outside it).")
    v5["pin_status"] = "PINNED"
    v5["pin_date"] = v4["pin_date"]                      # 2026-08-01 -- not a re-pin of instruments
    v5["filter_pin_date"] = v4["filter_pin_date"]        # 2026-08-05
    v5["record_correction_date"] = v4["record_correction_date"]  # 2026-08-26 (Amendment 21)
    v5["posthoc_repin_date"] = v4["posthoc_repin_date"]  # 2026-08-27 (Amendment 23)
    v5["exploratory_repin_date"] = "2026-08-28"
    v5["pinned_by"] = v4["pinned_by"]
    v5["authorized_by"] = ("Amendment 24 (2026-08-28, post-hoc exploratory analyses and the "
                           "safety-pair ratio ablation; sighted decision with pre-specified "
                           "outputs) via the Amendment 14a/21 legitimate re-pin path")
    v5["preregistration_revisions_in_force"] = v4["preregistration_revisions_in_force"]
    v5["preregistration_amendments_in_force"] = AMENDMENTS_IN_FORCE
    v5["immutable_after_pin"] = v4["immutable_after_pin"]
    v5["judges"] = copy.deepcopy(v4["judges"])
    v5["b3_guardrail_filter"] = copy.deepcopy(v4["b3_guardrail_filter"])
    v5["helpfulness_scorer"] = copy.deepcopy(v4["helpfulness_scorer"])
    v5["independence_statement"] = v4["independence_statement"]
    v5["labelling_caveat"] = v4["labelling_caveat"]
    v5["what_is_NOT_pinned_by_this_file"] = v4["what_is_NOT_pinned_by_this_file"]
    v5["_repin_note"] = v4["_repin_note"] + [
        "",
        "VERSION 5 ISSUED 2026-08-28 under Amendment 24. THE INSTRUMENTS ARE UNCHANGED:",
        "the `judges`, `b3_guardrail_filter` and `helpfulness_scorer` blocks are deep-equal",
        "to v4 (asserted at issuance by scripts/issue_lock_v5.py). Changes: Amendment 24",
        "added to preregistration_amendments_in_force (the staleness scan's refusal against",
        "v4 was verified to fire first, as the positive control); run_metadata gains",
        "exploratory_arms (T_r050 / T_r200, the Amendment 24.2 safety-pair ratio ablation,",
        "post-hoc exploratory, seed 1 only, never primary, never Table 1 or Table 2) and is",
        "otherwise unchanged. The primary analysis remains seed 1, unchanged.",
    ]

    # run_metadata: deep copy of v4's, plus the single exploratory_arms key. Nothing else.
    rm = copy.deepcopy(v4["run_metadata"])
    rm["exploratory_arms"] = {
        "arms": ["T_r050", "T_r200"],
        "seeds": {"T_r050": [1], "T_r200": [1]},
        "note": ("Amendment 24.2 (2026-08-28): safety-pair ratio ablation, POST-HOC "
                 "EXPLORATORY, a SIGHTED reinstatement of one cut ablation by the owner. "
                 "T_r050 trains on 50% of T's safety pairs and T_r200 on 200% (pool "
                 "contingency fixed at launch: if the filtered pool cannot supply 9,848 "
                 "rows the high arm uses the full pool and the achieved ratio is recorded "
                 "before any result exists), seed 1 only, matched volume preserved (total "
                 "pair count fixed at T's 19,924; the helpfulness count absorbs the "
                 "difference). Configs byte-identical to T except the sampling counts, arm "
                 "label and output template; scored by the identical frozen pipeline under "
                 "this lock. These arms are NEVER primary, NEVER in Table 1 or Table 2, "
                 "never in the headline sentence, and never used to evaluate the "
                 "pre-registered threshold; the pre-specified output (ASR primary and "
                 "crisis co-primary per ratio, with bootstrap CIs and a monotonicity "
                 "statement) is reported whatever the pattern is."),
    }
    assert set(rm) - {"exploratory_arms"} == set(v4["run_metadata"]), \
        "run_metadata keys drifted beyond the intended addition"
    v5["run_metadata"] = rm

    v5["supersedes"] = {
        "file": "configs/judges_pinned.lock.v4.json",
        "sha256_at_supersession": v4_sha,
        "reason": ("Amendment 24: re-pin to bring the amendment enumeration current and "
                   "record the exploratory ratio-ablation arms (T_r050 / T_r200, post-hoc "
                   "exploratory, seed 1). No judge, filter or helpfulness instrument "
                   "changed; the primary seed enumeration is unchanged."),
        "chain": ["configs/judges_pinned.lock.json (v1, superseded 2026-08-05, sha "
                  + V1_SHA + ")",
                  "configs/judges_pinned.lock.v2.json (v2, superseded 2026-08-26, sha "
                  + V2_SHA + ")",
                  "configs/judges_pinned.lock.v3.json (v3, superseded 2026-08-27, sha "
                  + V3_SHA + ")",
                  "configs/judges_pinned.lock.v4.json (v4, superseded 2026-08-28, sha "
                  + v4_sha + ")"],
    }

    # write with LF endings, no trailing platform conversion (.gitattributes -text)
    payload = json.dumps(v5, indent=2, ensure_ascii=False) + "\n"
    V5.write_bytes(payload.encode("utf-8"))

    # verify: instruments deep-equal to v4, run_metadata unchanged bar the addition,
    # old locks untouched
    v5_back = json.loads(V5.read_text(encoding="utf-8"))
    assert v5_back["judges"] == v4["judges"], "judges block drifted"
    assert v5_back["b3_guardrail_filter"] == v4["b3_guardrail_filter"], "filter block drifted"
    assert v5_back["helpfulness_scorer"] == v4["helpfulness_scorer"], \
        "helpfulness_scorer block drifted"
    rm_back = dict(v5_back["run_metadata"])
    rm_back.pop("exploratory_arms")
    assert rm_back == v4["run_metadata"], \
        "run_metadata changed beyond the exploratory_arms addition"
    assert v5_back["run_metadata"]["multi_seed_arms"] == {"B2": [1], "B3": [1], "T": [1]}, \
        "primary seed enumeration changed -- Amendment 23 protection 1 violated"
    assert sha256_file(V1) == V1_SHA and sha256_file(V2) == V2_SHA \
        and sha256_file(V3) == V3_SHA and sha256_file(V4) == v4_sha
    v5_sha = sha256_file(V5)

    # update judges.yaml pin pointer in tandem (the sanctioned two-file edit)
    ytext = YAML.read_bytes().decode("utf-8")
    old_file = "pin_lock_file: configs/judges_pinned.lock.v4.json"
    new_file = "pin_lock_file: configs/judges_pinned.lock.v5.json"
    old_sha_line = 'pin_lock_sha256: "' + v4_sha + '"'
    new_sha_line = 'pin_lock_sha256: "' + v5_sha + '"'
    assert old_file in ytext and old_sha_line in ytext, "judges.yaml pin pointer not found"
    ytext = ytext.replace(old_file, new_file).replace(old_sha_line, new_sha_line)
    YAML.write_bytes(ytext.encode("utf-8"))

    print("v5 lock issued:", V5)
    print("v5 sha256:", v5_sha)
    print("v4 sha256 at supersession:", v4_sha)
    print("v1/v2/v3/v4 verified untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
