"""One-shot issuer for configs/judges_pinned.lock.v6.json (Amendment 25, RESULTS LOCK).

Carries every instrument block (judges, b3_guardrail_filter, helpfulness_scorer) from
the v5 lock unchanged (deep-equality asserted at issuance), adds Amendment 25 to the
amendments-in-force enumeration, and records the RESULTS LOCK in run_metadata WITHOUT
touching anything else in it: run_metadata is a deep copy of v5's plus the single
`results_lock` key (asserted). judges.yaml's pin pointer is updated in tandem. v1–v5
locks are never touched; their hashes are asserted before and after. Run once;
re-running overwrites v6 deterministically with the same content.

Modeled on scripts/issue_lock_v5.py (the Amendment 14a/21 legitimate re-pin path).
"""
import copy, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "configs/judges_pinned.lock.json"
V2 = ROOT / "configs/judges_pinned.lock.v2.json"
V3 = ROOT / "configs/judges_pinned.lock.v3.json"
V4 = ROOT / "configs/judges_pinned.lock.v4.json"
V5 = ROOT / "configs/judges_pinned.lock.v5.json"
V6 = ROOT / "configs/judges_pinned.lock.v6.json"
YAML = ROOT / "configs/judges.yaml"

V1_SHA = "444aa1b6022f4fec0032a7f56e160a23f3b99c280ead4242c687abd0787e3530"
V2_SHA = "4b478e0322a0f9fcf831082a32bfc2b4a5f50038e99604259bc86fd84a23e470"
V3_SHA = "1c6572b63435e459150ace2be288d87448593909124167c350f4d8f6591fe57d"
V4_SHA = "8cabb12854df721b76dd4dd676c9cc3efa5774cef218fef8ae3151f79a3d9ea7"
# Expected sha of the v5 lock at supersession. Recomputed from disk below and asserted
# against this value; the recomputed value is what goes into the supersedes block.
V5_SHA_EXPECTED = "f01e4e1ac1819f76ed25c07e416753a66d8916876c14a2d071b9c470af22d745"

AMENDMENTS_IN_FORCE = ["7a", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                       "17", "18", "18a", "19", "20", "21", "22", "23", "24", "25"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha256_file(V1) == V1_SHA, "v1 lock bytes changed -- refusing"
    assert sha256_file(V2) == V2_SHA, "v2 lock bytes changed -- refusing"
    assert sha256_file(V3) == V3_SHA, "v3 lock bytes changed -- refusing"
    assert sha256_file(V4) == V4_SHA, "v4 lock bytes changed -- refusing"
    v5_sha = sha256_file(V5)
    assert v5_sha == V5_SHA_EXPECTED, (
        f"v5 lock hashes to {v5_sha}, expected {V5_SHA_EXPECTED} -- v5 bytes changed, refusing")
    v5 = json.loads(V5.read_text(encoding="utf-8"))

    v6 = {}
    v6["_what_this_is"] = (
        "Judge/filter pin lock, VERSION 6. Issued 2026-08-29 under pre-registration "
        "Amendment 25 (RESULTS LOCK). NO INSTRUMENT CHANGED: the `judges`, "
        "`b3_guardrail_filter` and `helpfulness_scorer` blocks are carried from v5 "
        "unchanged (deep-equality asserted at issuance). What v6 adds: (i) Amendment 25 "
        "enters `preregistration_amendments_in_force`, so verify_judge_pin()'s staleness "
        "scan verifies again -- its refusal against v5 after Amendment 25 appeared was "
        "confirmed to fire before this lock was issued, as the guard's positive control; "
        "(ii) run_metadata gains `results_lock` (declared 2026-08-29; every exhibit "
        "enumerated in notebook/results_manifest.json is frozen; corrections only via "
        "dated amendment + manifest regeneration in the same commit) and is otherwise a "
        "deep copy of v5's (asserted). This lock SUPERSEDES v5; v1-v5 are retained "
        "unedited (Amendment 14a: a pinned artefact is never edited; corrections and "
        "successions live outside it).")
    v6["pin_status"] = "PINNED"
    v6["pin_date"] = v5["pin_date"]                      # 2026-08-01 -- not a re-pin of instruments
    v6["filter_pin_date"] = v5["filter_pin_date"]        # 2026-08-05
    v6["record_correction_date"] = v5["record_correction_date"]  # 2026-08-26 (Amendment 21)
    v6["posthoc_repin_date"] = v5["posthoc_repin_date"]  # 2026-08-27 (Amendment 23)
    v6["exploratory_repin_date"] = v5["exploratory_repin_date"]  # 2026-08-28 (Amendment 24)
    v6["results_lock_date"] = "2026-08-29"
    v6["pinned_by"] = v5["pinned_by"]
    v6["authorized_by"] = ("Amendment 25 (2026-08-29, RESULTS LOCK; owner-directed) via "
                           "the Amendment 14a/21 legitimate re-pin path")
    v6["preregistration_revisions_in_force"] = v5["preregistration_revisions_in_force"]
    v6["preregistration_amendments_in_force"] = AMENDMENTS_IN_FORCE
    v6["immutable_after_pin"] = v5["immutable_after_pin"]
    v6["judges"] = copy.deepcopy(v5["judges"])
    v6["b3_guardrail_filter"] = copy.deepcopy(v5["b3_guardrail_filter"])
    v6["helpfulness_scorer"] = copy.deepcopy(v5["helpfulness_scorer"])
    v6["independence_statement"] = v5["independence_statement"]
    v6["labelling_caveat"] = v5["labelling_caveat"]
    v6["what_is_NOT_pinned_by_this_file"] = v5["what_is_NOT_pinned_by_this_file"]
    v6["_repin_note"] = v5["_repin_note"] + [
        "",
        "VERSION 6 ISSUED 2026-08-29 under Amendment 25 (RESULTS LOCK). THE INSTRUMENTS",
        "ARE UNCHANGED: the `judges`, `b3_guardrail_filter` and `helpfulness_scorer`",
        "blocks are deep-equal to v5 (asserted at issuance by scripts/issue_lock_v6.py).",
        "Changes: Amendment 25 added to preregistration_amendments_in_force (the",
        "staleness scan's refusal against v5 was verified to fire first, as the positive",
        "control; captured 2026-08-29); run_metadata gains results_lock and is otherwise",
        "unchanged. From this date no locked number may change; corrections go through a",
        "dated amendment plus results-manifest regeneration in the same commit. The",
        "primary analysis remains seed 1, unchanged.",
    ]

    # run_metadata: deep copy of v5's, plus the single results_lock key. Nothing else.
    rm = copy.deepcopy(v5["run_metadata"])
    rm["results_lock"] = {
        "declared": "2026-08-29",
        "amendment": "25",
        "inventory": ("notebook/results_manifest.json -- every exhibit it enumerates "
                      "(37+ exhibits: Tables 1-2 cells, primary/co-primary/TOST/"
                      "over-refusal statistics, sign convention, headline sentence, "
                      "Amendment 23 robustness panel, T_ctrl, frontier figure, six "
                      "Amendment 24 exploratory analyses, Amendment 24.2 ratio-ablation "
                      "dose-response), each with artifact path, sha256, config, and seed"),
        "rules": ("1. No locked number changes; a discovered error is corrected only "
                  "through a dated amendment plus manifest regeneration in the same "
                  "commit. 2. Drafting uses locked numbers only, cited against the "
                  "manifest. 3. Exploratory exhibits stay exploratory in every draft. "
                  "4. The instrument layer (suite, judges, adapters) remains immutable "
                  "as before; the lock adds the results layer on top."),
    }
    assert set(rm) - {"results_lock"} == set(v5["run_metadata"]), \
        "run_metadata keys drifted beyond the intended addition"
    v6["run_metadata"] = rm

    v6["supersedes"] = {
        "file": "configs/judges_pinned.lock.v5.json",
        "sha256_at_supersession": v5_sha,
        "reason": ("Amendment 25: re-pin to bring the amendment enumeration current and "
                   "record the RESULTS LOCK. No judge, filter or helpfulness instrument "
                   "changed; the primary seed enumeration is unchanged."),
        "chain": ["configs/judges_pinned.lock.json (v1, superseded 2026-08-05, sha "
                  + V1_SHA + ")",
                  "configs/judges_pinned.lock.v2.json (v2, superseded 2026-08-26, sha "
                  + V2_SHA + ")",
                  "configs/judges_pinned.lock.v3.json (v3, superseded 2026-08-27, sha "
                  + V3_SHA + ")",
                  "configs/judges_pinned.lock.v4.json (v4, superseded 2026-08-28, sha "
                  + V4_SHA + ")",
                  "configs/judges_pinned.lock.v5.json (v5, superseded 2026-08-29, sha "
                  + v5_sha + ")"],
    }

    # write with LF endings, no trailing platform conversion (.gitattributes -text)
    payload = json.dumps(v6, indent=2, ensure_ascii=False) + "\n"
    V6.write_bytes(payload.encode("utf-8"))

    # verify: instruments deep-equal to v5, run_metadata unchanged bar the addition,
    # old locks untouched
    v6_back = json.loads(V6.read_text(encoding="utf-8"))
    assert v6_back["judges"] == v5["judges"], "judges block drifted"
    assert v6_back["b3_guardrail_filter"] == v5["b3_guardrail_filter"], "filter block drifted"
    assert v6_back["helpfulness_scorer"] == v5["helpfulness_scorer"], \
        "helpfulness_scorer block drifted"
    rm_back = dict(v6_back["run_metadata"])
    rm_back.pop("results_lock")
    assert rm_back == v5["run_metadata"], \
        "run_metadata changed beyond the results_lock addition"
    assert v6_back["run_metadata"]["multi_seed_arms"] == {"B2": [1], "B3": [1], "T": [1]}, \
        "primary seed enumeration changed -- Amendment 23 protection 1 violated"
    assert sha256_file(V1) == V1_SHA and sha256_file(V2) == V2_SHA \
        and sha256_file(V3) == V3_SHA and sha256_file(V4) == V4_SHA \
        and sha256_file(V5) == v5_sha
    v6_sha = sha256_file(V6)

    # update judges.yaml pin pointer in tandem (the sanctioned two-file edit)
    ytext = YAML.read_bytes().decode("utf-8")
    old_file = "pin_lock_file: configs/judges_pinned.lock.v5.json"
    new_file = "pin_lock_file: configs/judges_pinned.lock.v6.json"
    old_sha_line = 'pin_lock_sha256: "' + v5_sha + '"'
    new_sha_line = 'pin_lock_sha256: "' + v6_sha + '"'
    assert old_file in ytext and old_sha_line in ytext, "judges.yaml pin pointer not found"
    ytext = ytext.replace(old_file, new_file).replace(old_sha_line, new_sha_line)
    YAML.write_bytes(ytext.encode("utf-8"))

    print("v6 lock issued:", V6)
    print("v6 sha256:", v6_sha)
    print("v5 sha256 at supersession:", v5_sha)
    print("v1/v2/v3/v4/v5 verified untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
