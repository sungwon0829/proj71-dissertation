#!/usr/bin/env bash
# Project 71 -- T_ctrl chain (train -> generate -> score), seed 1.
#
# Produces: the T_ctrl row referenced in CLAUDE.md's EXPERIMENTAL ARMS table and
# notebook/preregistration.md Revision 5 -- a single-run weak control (italics, no
# significance test, Amendment 9). Not one of the two headline arms (B3 vs T remains
# THE comparison); T_ctrl exists to bound whether a T-vs-B3 gain is attributable to
# safety-specific preference direction or merely to the presence of ~25% out-of-domain
# preference data (776 of 4,924 PKU rows actually flip direction; see
# results/T_ctrl_dpo_seed1/dpo_data_manifest.json's weak_control_note once written).
#
# Launched because Amendment 22.2's calendar condition FIRED: Tables 1-2 + the primary
# test completed 2026-08-28, before the 30 Aug gate (see notebook/lab_notebook.md
# 2026-08-28 "Pre-lock repro audit", finding M1, main-thread disposition).
#
# LAUNCH: this script must itself be started with `nohup ... & disown` from a single
# foreground Bash call -- NEVER the harness's run_in_background (hard ~60-minute kill
# trap, notebook/lab_notebook.md 2026-08-02), matching the project convention in
# scripts/run_seeds23_overnight.sh.
#
# No GPU gate: unlike run_seeds23_overnight.sh (which waited on an in-flight probe),
# nvidia-smi was confirmed idle (2 MiB used, no processes) immediately before this
# script was launched, so no gate stage is included here.
#
# Usage: bash scripts/run_t_ctrl_chain.sh

set -u
# Windows-style forward-slash path, deliberately NOT the /c/proj71 MSYS form -- this ROOT
# is also passed as literal argv/string content to python.exe (a non-MSYS binary), which
# does not receive Git Bash's automatic POSIX-path rewriting when embedded inside a quoted
# argument (same finding as run_seeds23_overnight.sh).
ROOT="C:/proj71"
cd "$ROOT" || { echo "FATAL: cannot cd to $ROOT"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/env/Scripts/activate"

PY="$ROOT/env/Scripts/python"
SUITE="$ROOT/data/redteam/redteam_suite.jsonl"
GEN_SEED=42     # fixed project-wide generation/decoding seed (greedy decoding -- generation
                # seed does not change output; kept for convention/determinism logging).
TRAIN_SEED=1    # T_ctrl is a single-seed arm (Amendment 9), paired with T seed 1.

OUTDIR="$ROOT/results/t_ctrl_chain"
mkdir -p "$OUTDIR"
STATUS="$OUTDIR/chain_status.json"
CHAIN_LOG="$OUTDIR/chain.log"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log_chain() { echo "[$(now_iso)] $*" | tee -a "$CHAIN_LOG"; }

# ---------------------------------------------------------------------------------------
# Status file writer (python -- no jq on this box). Rewrites chain_status.json after
# every stage transition (start AND end), never truncating other stages' records.
# ---------------------------------------------------------------------------------------
write_status () {
    # args: stage exit_code_or_empty_or_SKIPPED start_iso end_iso note
    "$PY" - "$STATUS" "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import json, sys, os, datetime

status_path, stage, exit_code, start_iso, end_iso, note = sys.argv[1:7]
data = {}
if os.path.isfile(status_path):
    try:
        data = json.load(open(status_path, encoding="utf-8"))
    except Exception:
        data = {}
data.setdefault("chain", "t_ctrl_seed1")
data.setdefault("amendment", "22.2")
data.setdefault("label", "T_ctrl (Revision 5 single-run weak control, Amendment 9) -- "
                          "condition fired 2026-08-28, lab_notebook.md M1 disposition")
data.setdefault("stages", {})

if exit_code in ("", "None"):
    ec = None
elif exit_code == "SKIPPED":
    ec = "SKIPPED"
else:
    ec = int(exit_code)

data["stages"][stage] = {
    "exit_code": ec,
    "start": start_iso or None,
    "end": end_iso or None,
    "note": note or None,
}
data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
tmp = status_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
os.replace(tmp, status_path)
PYEOF
}

# ---------------------------------------------------------------------------------------
# Pre-flight assertions -- verify, don't assume. Abort loudly rather than proceed on a
# stale/edited protocol.
# ---------------------------------------------------------------------------------------
preflight () {
    local start end
    start=$(now_iso)
    write_status "preflight" "" "$start" "" "running"

    if ! grep -q "22.2 T_ctrl: conditional" "$ROOT/notebook/preregistration_amendments.md"; then
        log_chain "PREFLIGHT FAILED: Amendment 22.2 not found in notebook/preregistration_amendments.md"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: Amendment 22.2 heading missing"
        exit 1
    fi

    local pin_line
    pin_line=$(grep -m1 "^pin_lock_sha256:" "$ROOT/configs/judges.yaml")
    case "$pin_line" in
        *8cabb128*) : ;;
        *)
            log_chain "PREFLIGHT FAILED: configs/judges.yaml pin_lock_sha256 does not start with 8cabb128 (got: $pin_line)"
            write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: judges.yaml not pinned to lock v4 (8cabb128...)"
            exit 1
            ;;
    esac

    # Config identity: dpo_t_ctrl.yaml must be byte-identical to the pre-launch-verified
    # value (sha computed and reported to the coordinator before this chain was launched).
    local tctrl_sha
    tctrl_sha=$("$PY" -c "import hashlib;print(hashlib.sha256(open('$ROOT/configs/dpo_t_ctrl.yaml','rb').read()).hexdigest())")
    if [ "$tctrl_sha" != "07ffb28767f6eac5bad6593ce2a1133564022f1258d97e5b8848ec73c7be552c" ]; then
        log_chain "PREFLIGHT FAILED: configs/dpo_t_ctrl.yaml sha256 drifted from the pre-launch-verified value (got $tctrl_sha)"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: dpo_t_ctrl.yaml sha mismatch"
        exit 1
    fi

    # T seed 1's manifest must already exist -- T_ctrl's helpful-sample cross-check inside
    # train_dpo.py requires it (fail fast here with a clearer message than the trainer's).
    if [ ! -f "$ROOT/results/T_dpo_seed1/dpo_data_manifest.json" ]; then
        log_chain "PREFLIGHT FAILED: results/T_dpo_seed1/dpo_data_manifest.json missing -- T_ctrl's helpful-sample cross-check requires T seed 1 to have trained first"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: T seed1 manifest missing"
        exit 1
    fi

    # B1 v2 base checkpoint must exist.
    if [ ! -d "$ROOT/results/B1_sft_seed42_v2/checkpoint-290" ]; then
        log_chain "PREFLIGHT FAILED: results/B1_sft_seed42_v2/checkpoint-290 (base adapter) missing"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: B1 v2 checkpoint-290 missing"
        exit 1
    fi

    # Collision guard: refuse to even start if any T_ctrl output path already exists.
    # (train_dpo.py / eval_generate.py already refuse to overwrite a non-empty/existing
    # target; this is a belt-and-braces early check with a clear message.)
    local p
    for p in \
        "$ROOT/results/T_ctrl_dpo_seed1" \
        "$ROOT/results/t_ctrl_seed42"
    do
        if [ -e "$p" ]; then
            log_chain "PREFLIGHT FAILED: target path already exists: $p"
            write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: pre-existing path $p"
            exit 1
        fi
    done

    end=$(now_iso)
    log_chain "PREFLIGHT PASSED: Amendment 22.2 present; judges.yaml pinned to lock v4 (8cabb128...); dpo_t_ctrl.yaml sha256 matches pre-launch value; T seed1 manifest present; B1 v2 checkpoint-290 present; no output-path collisions."
    write_status "preflight" "0" "$start" "$end" "ok"
}

# ---------------------------------------------------------------------------------------
# Generic stage runner. Each stage depends only on the previous one in this linear chain.
# ---------------------------------------------------------------------------------------
declare -A EXIT_CODE

run_stage () {
    local stage="$1"; shift
    local dep="$1"; shift    # dependency stage name, or "" for none
    local logfile="$OUTDIR/${stage}.log"

    if [ -n "$dep" ]; then
        local depcode="${EXIT_CODE[$dep]:-MISSING}"
        if [ "$depcode" != "0" ]; then
            log_chain "SKIP $stage: dependency '$dep' exit_code=$depcode (not 0)"
            EXIT_CODE[$stage]="SKIPPED"
            write_status "$stage" "SKIPPED" "" "$(now_iso)" "skipped: dependency $dep exit_code=$depcode"
            return 0
        fi
    fi

    local start_iso
    start_iso=$(now_iso)
    log_chain "START $stage :: $*"
    write_status "$stage" "" "$start_iso" "" "running"

    "$@" > "$logfile" 2>&1
    local code=$?
    local end_iso
    end_iso=$(now_iso)
    EXIT_CODE[$stage]="$code"
    if [ "$code" -eq 0 ]; then
        log_chain "END   $stage exit=0 (ok) log=$logfile"
        write_status "$stage" "$code" "$start_iso" "$end_iso" "ok"
    else
        log_chain "END   $stage exit=$code (FAILED -- attempted-and-incomplete, not auto-relaunched, Amendment 12 defect-only relaunch rule) log=$logfile"
        write_status "$stage" "$code" "$start_iso" "$end_iso" "FAILED: attempted-and-incomplete"
    fi
    return 0
}

# =========================================================================================
# MAIN
# =========================================================================================
log_chain "=== t_ctrl_seed1 chain starting (PID $$) ==="
preflight

run_stage train_t_ctrl "" \
    "$PY" scripts/train_dpo.py --config configs/dpo_t_ctrl.yaml --seed "$TRAIN_SEED"

run_stage gen_t_ctrl train_t_ctrl \
    "$PY" scripts/eval_generate.py --arm t_ctrl --adapter "results/T_ctrl_dpo_seed${TRAIN_SEED}" \
        --suite "$SUITE" --seed "$GEN_SEED" --config configs/eval_generation.yaml

# Over-refusal for t_ctrl has NO hand labels (Revision 4's hand-label requirement covers
# B3/T seed 1 only) -- the summary will report over-refusal via the judge cross-check only
# and mark is_paper_number False for that column. Expected, not an error; do not pass
# --over_refusal_labels here.
run_stage score_t_ctrl gen_t_ctrl \
    "$PY" scripts/eval_score.py --generations "results/t_ctrl_seed${GEN_SEED}/generations.jsonl" \
        --suite "$SUITE" --out "results/t_ctrl_seed${GEN_SEED}/scored_realsuite.jsonl"

log_chain "=== t_ctrl_seed1 chain FINISHED (see $STATUS for per-stage outcome) ==="
