#!/usr/bin/env bash
# Project 71 -- Amendment 23 post-hoc robustness chain.
#
# Produces: the Amendment 23 post-hoc robustness panel (per-seed ASR, crisis co-primary,
# over-refusal judge cross-check, helpfulness, per-seed McNemar) reported BESIDE Tables 1-2,
# clearly labelled post-hoc, never primary. See notebook/preregistration_amendments.md
# Amendment 23 (dated 2026-08-27, committed 77b45a8).
#
# Trains B2 and T at seeds 2 and 3 on otherwise-idle GPU time, using configs byte-identical
# to seed 1 (configs/dpo_b2.yaml, configs/dpo_t.yaml) -- ONLY --seed differs. Per Amendment
# 12's defect-only relaunch rule, a stage is never relaunched because of the number it
# produces; a failed/incomplete stage is recorded as attempted-and-incomplete.
#
# SEED-KEYING DISAMBIGUATION (repro-audit finding 2): eval_generate.py/apply_b3_filter.py
# key their output directories on the GENERATION seed (fixed at 42 project-wide -- decoding
# is greedy, so the generation seed never needs to vary) via `results/<arm>_seed<gen_seed>/`.
# The TRAINING seed is therefore encoded in the --arm LABEL instead (natively supported,
# zero script changes): b2_ts2/b2_ts3/t_ts2/t_ts3 for generation, with an explicit --out for
# the B3 derivation (apply_b3_filter.py's own default would collide on results/b3_seed42/,
# which is seed-1's B3 output). Seed-1 directories (results/B2_dpo_seed1*, T_dpo_seed1,
# results/b2_seed42, results/b3_seed42, results/t_seed42) are never referenced or written to
# by this script.
#
# LAUNCH: this script must itself be started with `nohup ... & disown` from a single
# foreground Bash call -- NEVER the harness's run_in_background (hard ~60-minute kill trap,
# notebook/lab_notebook.md 2026-08-02).
#
# Usage: bash scripts/run_seeds23_overnight.sh

set -u
# Windows-style forward-slash path, deliberately NOT the /c/proj71 MSYS form: this ROOT is
# also passed as literal argv/string content to python.exe and nvidia-smi.exe (non-MSYS
# binaries), which do not receive Git Bash's automatic POSIX-path rewriting when a path is
# embedded inside a quoted argument (verified empirically before launch -- python -c with an
# /c/proj71/... path raised FileNotFoundError; C:/proj71/... works for both bash and python).
ROOT="C:/proj71"
cd "$ROOT" || { echo "FATAL: cannot cd to $ROOT"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/env/Scripts/activate"

PY="$ROOT/env/Scripts/python"
SUITE="$ROOT/data/redteam/redteam_suite.jsonl"
GEN_SEED=42   # fixed project-wide generation/decoding seed; TRAINING seed is what
              # distinguishes B2/T seeds 2-3, carried in the --arm label and adapter path.

OUTDIR="$ROOT/results/seeds23_overnight"
mkdir -p "$OUTDIR"
STATUS="$OUTDIR/chain_status.json"
GATE_LOG="$OUTDIR/00_gpu_gate.log"
CHAIN_LOG="$OUTDIR/chain.log"

MARKER="$ROOT/results/judge_probes/classifier_blindness_fullsuite_20260827/GPU_DONE.marker"

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
data.setdefault("chain", "seeds23_overnight")
data.setdefault("amendment", "23")
data.setdefault("label", "Amendment 23 post-hoc robustness panel (never primary)")
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

    if ! grep -q "^## Amendment 23" "$ROOT/notebook/preregistration_amendments.md"; then
        log_chain "PREFLIGHT FAILED: Amendment 23 not found in notebook/preregistration_amendments.md"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: Amendment 23 heading missing"
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

    # Config identity: dpo_b2.yaml / dpo_t.yaml must be byte-identical to the frozen seed-1
    # (v4) configuration -- verified today against git HEAD (commit 03a20a1, "configuration
    # frozen") before this chain was launched. Re-checked here at run time too.
    local b2_sha t_sha
    b2_sha=$("$PY" -c "import hashlib;print(hashlib.sha256(open('$ROOT/configs/dpo_b2.yaml','rb').read()).hexdigest())")
    t_sha=$("$PY" -c "import hashlib;print(hashlib.sha256(open('$ROOT/configs/dpo_t.yaml','rb').read()).hexdigest())")
    if [ "$b2_sha" != "0662467eec563df5042f3c68f1a878e66ab5fbd936450976d59e386ac7a64832" ]; then
        log_chain "PREFLIGHT FAILED: configs/dpo_b2.yaml sha256 drifted from the pre-launch-verified value (got $b2_sha)"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: dpo_b2.yaml sha mismatch"
        exit 1
    fi
    if [ "$t_sha" != "97873348d5d9e16ed5499b8e0a8ddb00a3a32da071a8d44a34824fcdb3aa95cb" ]; then
        log_chain "PREFLIGHT FAILED: configs/dpo_t.yaml sha256 drifted from the pre-launch-verified value (got $t_sha)"
        write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: dpo_t.yaml sha mismatch"
        exit 1
    fi

    # Collision guard: refuse to even start if any seed-2/3 output path already exists.
    # (train_dpo.py / eval_generate.py / apply_b3_filter.py already refuse to overwrite a
    # non-empty/existing target; this is a belt-and-braces early check with a clear message.)
    local p
    for p in \
        "$ROOT/results/B2_dpo_seed2" "$ROOT/results/B2_dpo_seed3" \
        "$ROOT/results/T_dpo_seed2" "$ROOT/results/T_dpo_seed3" \
        "$ROOT/results/b2_ts2_seed42" "$ROOT/results/b2_ts3_seed42" \
        "$ROOT/results/t_ts2_seed42" "$ROOT/results/t_ts3_seed42" \
        "$ROOT/results/b3_ts2_seed42" "$ROOT/results/b3_ts3_seed42"
    do
        if [ -e "$p" ]; then
            log_chain "PREFLIGHT FAILED: target path already exists: $p"
            write_status "preflight" "1" "$start" "$(now_iso)" "FAILED: pre-existing path $p"
            exit 1
        fi
    done

    end=$(now_iso)
    log_chain "PREFLIGHT PASSED: Amendment 23 present; judges.yaml pinned to lock v4 (8cabb128...); dpo_b2/dpo_t sha256 match seed-1; no output-path collisions."
    write_status "preflight" "0" "$start" "$end" "ok"
}

# ---------------------------------------------------------------------------------------
# GPU gate -- poll every 60s. Proceed on whichever fires first:
#   (a) the classifier-blindness sentinel appears,
#   (b) elapsed > 5h AND nvidia-smi reports < 10 GB used,
#   (c) elapsed > 8h regardless (log loudly).
# ---------------------------------------------------------------------------------------
gate_gpu () {
    local start_epoch now_epoch elapsed used_mib decision start_iso
    start_iso=$(now_iso)
    write_status "gpu_gate" "" "$start_iso" "" "waiting for sentinel"
    start_epoch=$(date +%s)
    echo "[$(now_iso)] GATE START: waiting on $MARKER" >> "$GATE_LOG"
    decision=""
    while true; do
        now_epoch=$(date +%s)
        elapsed=$((now_epoch - start_epoch))
        if [ -f "$MARKER" ]; then
            decision="marker_found"
            echo "[$(now_iso)] GATE: sentinel found (elapsed ${elapsed}s). Proceeding." >> "$GATE_LOG"
            break
        fi
        used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')
        case "$used_mib" in ''|*[!0-9]*) used_mib=999999 ;; esac
        echo "[$(now_iso)] GATE: poll elapsed=${elapsed}s gpu_used=${used_mib}MiB marker=absent" >> "$GATE_LOG"
        if [ "$elapsed" -gt $((5*3600)) ] && [ "$used_mib" -lt 10240 ]; then
            decision="elapsed_5h_and_gpu_idle"
            echo "[$(now_iso)] GATE: elapsed > 5h AND gpu_used ${used_mib}MiB < 10GB -- proceeding (fallback rule b)." >> "$GATE_LOG"
            break
        fi
        if [ "$elapsed" -gt $((8*3600)) ]; then
            decision="elapsed_8h_forced"
            echo "[$(now_iso)] *** GATE: elapsed > 8h -- PROCEEDING ANYWAY (forced rule c) *** gpu_used=${used_mib}MiB" >> "$GATE_LOG"
            break
        fi
        sleep 60
    done
    local end_iso
    end_iso=$(now_iso)
    log_chain "GPU GATE cleared: decision=$decision elapsed=${elapsed}s"
    write_status "gpu_gate" "0" "$start_iso" "$end_iso" "decision=$decision elapsed_s=$elapsed"
}

# ---------------------------------------------------------------------------------------
# Generic stage runner with dependency gating. Independent branches are wired by simply
# not naming a dependency (e.g. train_t_s2 depends only on gpu_gate, never on B2 s2 --
# T trains from B1, not B2). Scoring stages never gate later training stages.
# ---------------------------------------------------------------------------------------
declare -A EXIT_CODE

run_stage () {
    local stage="$1"; shift
    local dep="$1"; shift    # dependency stage name, or "" for none (only gpu_gate/preflight)
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
        log_chain "END   $stage exit=$code (FAILED -- attempted-and-incomplete, not relaunched per Amendment 12) log=$logfile"
        write_status "$stage" "$code" "$start_iso" "$end_iso" "FAILED: attempted-and-incomplete (Amendment 12 defect-only relaunch rule; not auto-retried)"
    fi
    return 0
}

# =========================================================================================
# MAIN
# =========================================================================================
log_chain "=== seeds23_overnight chain starting (PID $$) ==="
preflight
gate_gpu

# ---- SEED 2 -----------------------------------------------------------------------------
run_stage train_b2_s2 "" \
    "$PY" scripts/train_dpo.py --config configs/dpo_b2.yaml --seed 2

run_stage gen_b2_s2 train_b2_s2 \
    "$PY" scripts/eval_generate.py --arm b2_ts2 --adapter results/B2_dpo_seed2 \
        --suite "$SUITE" --seed "$GEN_SEED"

run_stage filter_b3_s2 gen_b2_s2 \
    "$PY" scripts/apply_b3_filter.py --generations results/b2_ts2_seed42/generations.jsonl \
        --expect_source_arm b2_ts2 --out results/b3_ts2_seed42/generations.jsonl

# T trains from B1 directly -- depends only on the GPU gate, never on B2 s2's outcome.
run_stage train_t_s2 "" \
    "$PY" scripts/train_dpo.py --config configs/dpo_t.yaml --seed 2

run_stage gen_t_s2 train_t_s2 \
    "$PY" scripts/eval_generate.py --arm t_ts2 --adapter results/T_dpo_seed2 \
        --suite "$SUITE" --seed "$GEN_SEED"

# Scoring for seed 2 (b2/b3/t). Over-refusal has NO hand labels for seeds 2-3 by design
# (Amendment 23): judge cross-check only, expected, not an error. Scoring failures never
# gate later TRAINING stages (train_b2_s3 / train_t_s3 depend only on the GPU gate).
run_stage score_b2_s2 gen_b2_s2 \
    "$PY" scripts/eval_score.py --generations results/b2_ts2_seed42/generations.jsonl \
        --suite "$SUITE" --out results/b2_ts2_seed42/scored_realsuite.jsonl

run_stage score_b3_s2 filter_b3_s2 \
    "$PY" scripts/eval_score.py --generations results/b3_ts2_seed42/generations.jsonl \
        --suite "$SUITE" --out results/b3_ts2_seed42/scored_realsuite.jsonl

run_stage score_t_s2 gen_t_s2 \
    "$PY" scripts/eval_score.py --generations results/t_ts2_seed42/generations.jsonl \
        --suite "$SUITE" --out results/t_ts2_seed42/scored_realsuite.jsonl

# ---- SEED 3 -----------------------------------------------------------------------------
run_stage train_b2_s3 "" \
    "$PY" scripts/train_dpo.py --config configs/dpo_b2.yaml --seed 3

run_stage gen_b2_s3 train_b2_s3 \
    "$PY" scripts/eval_generate.py --arm b2_ts3 --adapter results/B2_dpo_seed3 \
        --suite "$SUITE" --seed "$GEN_SEED"

run_stage filter_b3_s3 gen_b2_s3 \
    "$PY" scripts/apply_b3_filter.py --generations results/b2_ts3_seed42/generations.jsonl \
        --expect_source_arm b2_ts3 --out results/b3_ts3_seed42/generations.jsonl

# T s3 depends only on the GPU gate -- a B2 s3 failure must not prevent T s3.
run_stage train_t_s3 "" \
    "$PY" scripts/train_dpo.py --config configs/dpo_t.yaml --seed 3

run_stage gen_t_s3 train_t_s3 \
    "$PY" scripts/eval_generate.py --arm t_ts3 --adapter results/T_dpo_seed3 \
        --suite "$SUITE" --seed "$GEN_SEED"

run_stage score_b2_s3 gen_b2_s3 \
    "$PY" scripts/eval_score.py --generations results/b2_ts3_seed42/generations.jsonl \
        --suite "$SUITE" --out results/b2_ts3_seed42/scored_realsuite.jsonl

run_stage score_b3_s3 filter_b3_s3 \
    "$PY" scripts/eval_score.py --generations results/b3_ts3_seed42/generations.jsonl \
        --suite "$SUITE" --out results/b3_ts3_seed42/scored_realsuite.jsonl

run_stage score_t_s3 gen_t_s3 \
    "$PY" scripts/eval_score.py --generations results/t_ts3_seed42/generations.jsonl \
        --suite "$SUITE" --out results/t_ts3_seed42/scored_realsuite.jsonl

log_chain "=== seeds23_overnight chain FINISHED (see $STATUS for per-stage outcome) ==="
