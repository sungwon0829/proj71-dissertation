"""One-shot issuer for configs/judges_pinned.lock.v3.json (Amendment 21).

Carries every instrument block (judges, b3_guardrail_filter) from the v2 lock
unchanged, corrects the record fields, and updates judges.yaml's pin pointer in
tandem. v1 and v2 locks are never touched; their hashes are asserted before and
after. Run once; re-running overwrites v3 deterministically with the same content.
"""
import copy, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "configs/judges_pinned.lock.json"
V2 = ROOT / "configs/judges_pinned.lock.v2.json"
V3 = ROOT / "configs/judges_pinned.lock.v3.json"
YAML = ROOT / "configs/judges.yaml"

V1_SHA = "444aa1b6022f4fec0032a7f56e160a23f3b99c280ead4242c687abd0787e3530"
V2_SHA = "4b478e0322a0f9fcf831082a32bfc2b4a5f50038e99604259bc86fd84a23e470"

AMENDMENTS_IN_FORCE = ["7a", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                       "17", "18", "18a", "19", "20", "21", "22"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha256_file(V1) == V1_SHA, "v1 lock bytes changed -- refusing"
    assert sha256_file(V2) == V2_SHA, "v2 lock bytes changed -- refusing"
    v2 = json.loads(V2.read_text(encoding="utf-8"))

    v3 = {}
    v3["_what_this_is"] = (
        "Judge/filter pin lock, VERSION 3. Issued 2026-08-26 under pre-registration "
        "Amendment 21 (record-correction re-pin) with the Amendment 22 descope recorded "
        "in run_metadata. NO INSTRUMENT CHANGED: the `judges` and `b3_guardrail_filter` "
        "blocks are carried from v2 unchanged (deep-equality asserted at issuance). What "
        "v3 corrects/adds: (i) the independence_statement, which v2 carried verbatim "
        "from v1 and which still named beaver-dam as the B3 filter after Amendment 20 "
        "installed Llama-Guard-3-8B; (ii) the helpfulness scorer enters the "
        "verification chain (closing the Amendment 17 audit gap); (iii) the full "
        "protocol in force is enumerated (revisions AND amendments) and machine-checked "
        "by verify_judge_pin(). This lock SUPERSEDES v2; v1 and v2 are retained "
        "unedited (Amendment 14a: a pinned artefact is never edited; corrections and "
        "successions live outside it).")
    v3["pin_status"] = "PINNED"
    v3["pin_date"] = v2["pin_date"]                      # 2026-08-01 -- not a re-pin of instruments
    v3["filter_pin_date"] = v2["filter_pin_date"]        # 2026-08-05
    v3["record_correction_date"] = "2026-08-26"
    v3["pinned_by"] = v2["pinned_by"]
    v3["authorized_by"] = ("Amendment 21 (2026-08-26, record-correction re-pin via the "
                           "Amendment 14a path) + Amendment 22 (descope, recorded in "
                           "run_metadata)")
    v3["preregistration_revisions_in_force"] = v2["preregistration_revisions_in_force"]
    v3["preregistration_amendments_in_force"] = AMENDMENTS_IN_FORCE
    v3["immutable_after_pin"] = v2["immutable_after_pin"]
    v3["judges"] = copy.deepcopy(v2["judges"])
    v3["b3_guardrail_filter"] = copy.deepcopy(v2["b3_guardrail_filter"])
    v3["helpfulness_scorer"] = {
        "role": "helpfulness_scorer",
        "not_a_judge": True,
        "model": "Psychotherapy-LLM/PsychoCounsel-Llama3-8B-Reward",
        "revision": "edab9eae1739cdcec1168db09593afe8943eb3ea",
        "license": "CC-BY-NC-4.0 (non-commercial; checkpoints scored with it inherit the restriction)",
        "note": ("Brought into verify_judge_pin()'s verification chain by Amendment 21, "
                 "closing the gap flagged in the Amendment 17 audit (revision-pinned in "
                 "judges.yaml but not tamper-evident). It has no prompt to hash; the lock "
                 "pins model + revision and verify_judge_pin() hard-fails on drift."),
        "authorized_by": "Amendment 21 (2026-08-26); role unchanged since safeguards v2",
    }
    v3["independence_statement"] = (
        "The B3 filter (meta-llama/Llama-Guard-3-8B, a Llama-family generative guard, "
        "installed under Amendments 19/20) shares no model with any judge: all three "
        "judges run Qwen2.5-7B-Instruct with pinned prompts. The three judges share a "
        "base model with EACH OTHER, which rule 1 permits; what it forbids is a judge "
        "sharing a mechanism with the FILTER, and none does. eval_score.py enumerates "
        "every asr_judge and raises on collision, comparing model name only so that a "
        "different revision or prompt on the same weights still counts as the same "
        "mechanism. The predecessor filter (PKU-Alignment/beaver-dam-7b) is "
        "diagnostic-only and is likewise no judge. CORRECTION NOTE: v1 and v2 carried an "
        "independence_statement naming beaver-dam as the filter; after Amendment 20 that "
        "was stale. Corrected here by Amendment 21; the property itself held throughout "
        "(test_judge_independence.py 15/15 in every configuration).")
    v3["labelling_caveat"] = (
        v2["labelling_caveat"]
        + " UPDATE 2026-08-26: the owner's 60-item human labelling pass is complete "
          "(results/human_validation/human_asr_kappa_report_20260826_231917.json: "
          "kappa 0.668 [0.453, 0.851] on the 54 non-gold items, human-vs-judge, gold "
          "check 6/6, intra-rater 10/10). The kappas recorded in `judges` above remain "
          "the inter-model figures they always were; the human figure is reported "
          "alongside them, never instead (preregistration.md CORRECTION 2026-08-01).")
    v3["what_is_NOT_pinned_by_this_file"] = v2["what_is_NOT_pinned_by_this_file"]
    v3["_repin_note"] = v2["_repin_note"] + [
        "",
        "VERSION 3 ISSUED 2026-08-26 under Amendment 21. THE INSTRUMENTS ARE UNCHANGED:",
        "the `judges` and `b3_guardrail_filter` blocks are deep-equal to v2 (asserted at",
        "issuance by scripts/issue_lock_v3.py). Changes: independence_statement corrected",
        "(v2's still named beaver-dam as the filter after the Amendment 20 install of",
        "Llama Guard); helpfulness scorer pinned (Amendment 17's deferred audit item);",
        "preregistration_amendments_in_force added and machine-checked; run_metadata",
        "records the Amendment 22 descope (1 seed per arm, T_ctrl conditional).",
    ]

    rm = copy.deepcopy(v2["run_metadata"])
    rm["_what_this_is"] = (
        "Which seeds constitute each arm. Original enumeration fixed 2026-08-01 BEFORE "
        "seeds 2-3 were launched; descoped to 1 seed per arm by Amendment 22 "
        "(2026-08-26), time-forced and decided blind (no cross-arm number existed; "
        "seeds 2-3 and T_ctrl were never trained).")
    rm["multi_seed_arms"] = {"B2": [1], "B3": [1], "T": [1]}
    rm["descope_note_amendment22"] = (
        "Amendment 22 (2026-08-26): seeds reduced to 1 per arm. Original 2026-08-01 "
        "enumeration was B2/B3/T = [1, 2, 3]; seeds 2-3 were never trained. Seed "
        "variance is reported as NOT MEASURED, with the reason stated in Methods and "
        "Limitations. The primary test operates over paired prompts (n=180 primary, "
        "crisis co-primary n=60, Amendment 13), not over seeds.")
    rm["t_ctrl_is_single_seed"] = (
        v2["run_metadata"]["t_ctrl_is_single_seed"]
        + " CONDITIONAL per Amendment 22.2: trained only if Tables 1-2 and the primary "
          "test are complete by 2026-08-30; otherwise reported as not run, with the "
          "unresolved preference-direction confound stated in Discussion.")
    rm.pop("KNOWN INCONSISTENCY, flagged not silently fixed", None)
    rm["known_inconsistency_resolved"] = (
        "v1/v2 carried a KNOWN INCONSISTENCY note claiming configs/dpo_b2.yaml and "
        "dpo_t.yaml contain `seed: 42`. That note was factually stale from 2026-08-01 "
        "(the field was removed from all four configs; assert_no_config_seed() guards "
        "its return) and was superseded by Amendment 14a. Recorded here as resolved "
        "rather than repeated.")
    v3["run_metadata"] = rm

    v3["supersedes"] = {
        "file": "configs/judges_pinned.lock.v2.json",
        "sha256_at_supersession": V2_SHA,
        "reason": ("Amendment 21: record-correction re-pin (stale independence_statement; "
                   "helpfulness scorer into the hash chain; amendments-in-force "
                   "enumeration) + Amendment 22 descope recorded in run_metadata. No "
                   "judge or filter instrument changed."),
        "chain": ["configs/judges_pinned.lock.json (v1, superseded 2026-08-05, sha "
                  + V1_SHA + ")",
                  "configs/judges_pinned.lock.v2.json (v2, superseded 2026-08-26, sha "
                  + V2_SHA + ")"],
    }

    # write with LF endings, no trailing platform conversion (.gitattributes -text)
    payload = json.dumps(v3, indent=2, ensure_ascii=False) + "\n"
    V3.write_bytes(payload.encode("utf-8"))

    # verify: instruments deep-equal to v2, old locks untouched
    v3_back = json.loads(V3.read_text(encoding="utf-8"))
    assert v3_back["judges"] == v2["judges"], "judges block drifted"
    assert v3_back["b3_guardrail_filter"] == v2["b3_guardrail_filter"], "filter block drifted"
    assert sha256_file(V1) == V1_SHA and sha256_file(V2) == V2_SHA
    v3_sha = sha256_file(V3)

    # update judges.yaml pin pointer in tandem (the sanctioned two-file edit)
    ytext = YAML.read_bytes().decode("utf-8")
    old_file = "pin_lock_file: configs/judges_pinned.lock.v2.json"
    new_file = "pin_lock_file: configs/judges_pinned.lock.v3.json"
    old_sha_line = 'pin_lock_sha256: "' + V2_SHA + '"'
    new_sha_line = 'pin_lock_sha256: "' + v3_sha + '"'
    assert old_file in ytext and old_sha_line in ytext, "judges.yaml pin pointer not found"
    ytext = ytext.replace(old_file, new_file).replace(old_sha_line, new_sha_line)
    YAML.write_bytes(ytext.encode("utf-8"))

    print("v3 lock issued:", V3)
    print("v3 sha256:", v3_sha)
    print("v1/v2 verified untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
