#!/usr/bin/env bash
# Re-score seed-1 b3/t/b2 with the TRUSTWORTHY hand over-refusal labels, then run stats.
# Sanctioned by the 2026-08-27 Phase 2 plan ("After labelling: split -> per-arm files ->
# re-score b3/t with --over_refusal_labels"); b2 takes b3's file (benign responses
# byte-identical, filter fired 0/60 benign). Raw generations are never touched.
set -u
cd /c/proj71
source env/Scripts/activate
OUT=results/rescore_handlabels
mkdir -p "$OUT"
S=$OUT/status.txt
HV=results/human_validation
echo "start $(date -u +%FT%TZ)" > "$S"

run() {
  name=$1; shift
  echo "RUN $name $(date -u +%FT%TZ)" >> "$S"
  "$@" > "$OUT/$name.log" 2>&1
  rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)" >> "$S"
  return $rc
}

run score_b3 python scripts/eval_score.py --generations results/b3_seed42/generations.jsonl \
  --over_refusal_labels "$HV/over_refusal_labels_b3_from_merged.json" --allow_overwrite_scored \
  || echo "FAIL score_b3" >> "$S"
run score_t python scripts/eval_score.py --generations results/t_seed42/generations.jsonl \
  --over_refusal_labels "$HV/over_refusal_labels_t_from_merged.json" --allow_overwrite_scored \
  || echo "FAIL score_t" >> "$S"
run score_b2 python scripts/eval_score.py --generations results/b2_seed42/generations.jsonl \
  --over_refusal_labels "$HV/over_refusal_labels_b3_from_merged.json" --allow_overwrite_scored \
  || echo "FAIL score_b2" >> "$S"

# Conforming view for stats: discover() validates every scored dir under results_dir, and
# the Amendment 23 chain dirs (b3_ts{2,3}_seed42) fail its arm-name assertion by design.
V=results/posthoc_stats/view_s1
mkdir -p "$V/b3_seed42" "$V/t_seed42"
cp results/b3_seed42/scored_realsuite.jsonl "$V/b3_seed42/"
cp results/t_seed42/scored_realsuite.jsonl "$V/t_seed42/"
echo "view_s1: byte copies of results/{b3,t}_seed42/scored_realsuite.jsonl taken AFTER the hand-label re-score ($(date -u +%FT%TZ)). Reason: stats.py discover() validates every scored dir under results_dir and the Amendment 23 ts2/ts3 b3 dirs fail its arm-name assertion (apply_b3_filter stamps arm='b3'). No content modified." > "$V/README.txt"

run stats python scripts/stats.py --results_dir "$V" --treatment t --baseline b3 --seed 0 \
  --out results/stats_report_realsuite_handlabelled.json \
  || echo "FAIL stats" >> "$S"

echo "ALL_DONE $(date -u +%FT%TZ)" >> "$S"
