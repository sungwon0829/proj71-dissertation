#!/usr/bin/env bash
# Project 71 -- Amendment 24.2 safety-pair ratio ablation chain
# (train T_r050 -> gen T_r050 -> train T_r200 -> gen T_r200 -> score both), seed 1 only.
#
# Produces: results/exploratory/ dose-response inputs (0%=B2, 50%=T_r050, 100%=T,
# 200%-target/100%-achieved=T_r200) for the Amendment 24.2 post-hoc exploratory exhibit.
# EXPLORATORY ONLY -- never Table 1/2, never the headline sentence, never used to evaluate
# the pre-registered 10-point threshold (Amendment 24 header, restated in Amendment 24.2).
#
# GOVERNANCE: notebook/preregistration_amendments.md Amendment 24.2, dated 2026-08-28,
# lock v5 (judges.yaml pin_lock_sha256 f01e4e1a...). Reinstates the ratio ablation CLAUDE.md
# otherwise cuts -- owner decision recorded there, not here.
#
# Matched volume preserved: total pairs fixed at T's 19,924 for BOTH arms (helpfulness count
# absorbs the difference). Pool-cap contingency (fixed in the amendment, before any result
# exists): the filtered safety pool is 4,924 rows (fixed, not seed-dependent); T_r050 wants
# 2,462 (achievable exactly, 50.00%); T_r200 wants 9,848 (200%) but the pool cannot supply
# more than 4,924 (100%), so T_r200 uses the full pool and its achieved ratio is 100.00%,
# NOT 200% -- see configs/dpo_t_r200.yaml's header for the full consequence (this makes
# T_r200's sampled training data expected-identical to T seed 1's).
#
# LAUNCH: this script must itself be started with `nohup ... & disown` from a single
# foreground Bash call -- NEVER the harness's run_in_background (hard ~60-minute kill
# trap), matching scripts/run_t_ctrl_chain.sh's precedent.
#
# GPU gate: explicit wait-loop below (poll 60s, threshold <10GB used) rather than a single
# point-in-time check, since some time may elapse between this script being written and
# actually launched.
#
# Usage: bash scripts/run_ratio_ablation_chain.sh

set -u
ROOT="C:/proj71"
cd "$ROOT" || { echo "FATAL: cannot cd to $ROOT"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/env/Scripts/activate"

PY="$ROOT/env/Scripts/python"
SUITE="$ROOT/data/redteam/redteam_suite.jsonl"
GEN_SEED=42     # fixed project-wide generation/decoding seed (greedy decoding -- generation
                # seed does not change output; kept for convention/determinism logging).
TRAIN_SEED=1    # Amendment 24.2: both ratio-ablation arms are seed 1 only.

OUTDIR="$ROOT/results/ratio_ablation_chain"
mkdir -p "$OUTDIR"
STATUS="$OUTDIR/chain_status.json"
CHAIN_LOG="$OUTDIR/chain.log"

# Pre-launch-verified config hashes (computed by the train-runner agent immediately after
# writing these two files, 2026-08-28 -- reported to the coordinator before this chain was
# launched). Any drift after this point aborts preflight rather than training on an edited
# config.
R050_SHA="bb97fc539e7a2c1e8c83f43ebeb2af3f787514c37a2421f2cc6ff56077721941"
R200_SHA="1cf8b4dbfdddb14ebc2932f8fc12bfacfd0981cdf9e20976fb681cd0017db39f"
LOCK_V5_PREFIX="f01e4e1a"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log_chain() { echo "[$(now_iso)] $*" | tee -a "$CHAIN_LOG"; }

# ---------------------------------------------------------------------------------------
# Status file writer (python -- no jq on this box). Rewrites chain_status.json after every
# stage transition (start AND end), never truncating other stages' records. Same schema as
# scripts/run_t_ctrl_chain.sh.
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
data.setdefault("chain", "ratio_ablation_seed1")
data.setdefault("amendment", "24.2")
data.setdefault("label", "T_r050 / T_r200 (Amendment 24.2 post-hoc exploratory safety-pair "
                          "ratio ablation) -- launched 2026-08-28, seed 1 only, never Table 1/2")
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
# GPU idle wait-loop (task requirement: poll 60s until <10GB used).
# ---------------------------------------------------------------------------------------
gpu_wait () {
    local start end
    start=$(now_iso)
    write_status "gpu_wait" "" "$start" "" "running"
    while :; do
        local used_mb
        used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
        if [ -z "$used_mb" ]; then
            log_chain "GPU_WAIT: could not parse nvidia-smi output, retrying in 60s"
            sleep 60
            continue
        fi
        if [ "$used_mb" -lt 10240 ]; then
            log_chain "GPU_WAIT: ${used_mb} MiB used (< 10 GiB threshold) -- proceeding"
            break
        fi
        log_chain "GPU_WAIT: ${used_mb} MiB used (>= 10 GiB threshold) -- waiting 60s"
        sleep 60
    done
    end=$(now_iso)
    write_status "gpu_wait" "0" "$start" "$end" "ok: GPU below 10GiB threshold"
}

# ---------------------------------------------------------------------------------------
# Pre-flight assertions -- verify, don't assume. Abort loudly rather than proceed on a
# stale/edited protocol.
# ---------------------------------------------------------------------------------------
preflight () {
    local start end
    start=$(now_iso)
    write_status "preflight" "" "$start" "" "running"

    if ! grep -q "24.2 Safety-pair ratio ablation" "$ROOT/notebook/preregistration_amendments.md"; then
        log_chain "PREFLIGHT FAILED: Amendment 24.2 heading not found in notebook/preregistration_amendments.md"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: Amendment 24.2 heading missing"
        exit 1
    fi

    local pin_line
    pin_line=$(grep -m1 "^pin_lock_sha256:" "$ROOT/configs/judges.yaml")
    case "$pin_line" in
        *"$LOCK_V5_PREFIX"*) : ;;
        *)
            log_chain "PREFLIGHT FAILED: configs/judges.yaml pin_lock_sha256 does not start with $LOCK_V5_PREFIX (got: $pin_line)"
            write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: judges.yaml not pinned to lock v5"
            exit 1
            ;;
    esac

    local r050_sha r200_sha
    r050_sha=$("$PY" -c "import hashlib;print(hashlib.sha256(open('$ROOT/configs/dpo_t_r050.yaml','rb').read()).hexdigest())")
    if [ "$r050_sha" != "$R050_SHA" ]; then
        log_chain "PREFLIGHT FAILED: configs/dpo_t_r050.yaml sha256 drifted (got $r050_sha, expected $R050_SHA)"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: dpo_t_r050.yaml sha mismatch"
        exit 1
    fi
    local r200_sha
    r200_sha=$("$PY" -c "import hashlib;print(hashlib.sha256(open('$ROOT/configs/dpo_t_r200.yaml','rb').read()).hexdigest())")
    if [ "$r200_sha" != "$R200_SHA" ]; then
        log_chain "PREFLIGHT FAILED: configs/dpo_t_r200.yaml sha256 drifted (got $r200_sha, expected $R200_SHA)"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: dpo_t_r200.yaml sha mismatch"
        exit 1
    fi

    if [ ! -d "$ROOT/results/B1_sft_seed42_v2/checkpoint-290" ]; then
        log_chain "PREFLIGHT FAILED: results/B1_sft_seed42_v2/checkpoint-290 (base adapter) missing"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: B1 v2 checkpoint-290 missing"
        exit 1
    fi

    # Collision guard: refuse to even start if any target output path already exists.
    local p
    for p in \
        "$ROOT/results/T_r050_dpo_seed1" \
        "$ROOT/results/T_r200_dpo_seed1" \
        "$ROOT/results/t_r050_seed42" \
        "$ROOT/results/t_r200_seed42"
    do
        if [ -e "$p" ]; then
            log_chain "PREFLIGHT FAILED: target path already exists: $p"
            write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: pre-existing path $p"
            exit 1
        fi
    done

    end=$(now_iso)
    log_chain "PREFLIGHT PASSED: Amendment 24.2 present; judges.yaml pinned to lock v5 ($LOCK_V5_PREFIX...); dpo_t_r050.yaml/dpo_t_r200.yaml sha256 match pre-launch values; B1 v2 checkpoint-290 present; no output-path collisions."
    write_status "preflight" "0" "$start" "$end" "ok"
}

# ---------------------------------------------------------------------------------------
# Generic stage runner (matches scripts/run_t_ctrl_chain.sh's pattern).
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
log_chain "=== ratio_ablation_seed1 chain starting (PID $$) ==="
gpu_wait
preflight

run_stage train_t_r050 "" \
    "$PY" scripts/train_dpo.py --config configs/dpo_t_r050.yaml --seed "$TRAIN_SEED"

run_stage gen_t_r050 train_t_r050 \
    "$PY" scripts/eval_generate.py --arm t_r050 --adapter "results/T_r050_dpo_seed${TRAIN_SEED}" \
        --suite "$SUITE" --seed "$GEN_SEED" --config configs/eval_generation.yaml

run_stage train_t_r200 gen_t_r050 \
    "$PY" scripts/train_dpo.py --config configs/dpo_t_r200.yaml --seed "$TRAIN_SEED"

run_stage gen_t_r200 train_t_r200 \
    "$PY" scripts/eval_generate.py --arm t_r200 --adapter "results/T_r200_dpo_seed${TRAIN_SEED}" \
        --suite "$SUITE" --seed "$GEN_SEED" --config configs/eval_generation.yaml

# Neither ratio-ablation arm has hand labels (Revision 4's hand-label requirement covers
# B3/T seed 1 only) -- over-refusal is reported via the judge cross-check only (kappa
# ~0.074), same treatment as t_ctrl. Expected, not an error; do not pass
# --over_refusal_labels here.
run_stage score_t_r050 gen_t_r200 \
    "$PY" scripts/eval_score.py --generations "results/t_r050_seed${GEN_SEED}/generations.jsonl" \
        --suite "$SUITE" --out "results/t_r050_seed${GEN_SEED}/scored_realsuite.jsonl"

run_stage score_t_r200 score_t_r050 \
    "$PY" scripts/eval_score.py --generations "results/t_r200_seed${GEN_SEED}/generations.jsonl" \
        --suite "$SUITE" --out "results/t_r200_seed${GEN_SEED}/scored_realsuite.jsonl"

log_chain "=== ratio_ablation_seed1 chain FINISHED (see $STATUS for per-stage outcome) ==="
