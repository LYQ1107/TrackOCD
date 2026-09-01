#!/usr/bin/env bash
# Phase 4Q auto-finalize: wait for all running jobs, run evals in parallel,
# convert proposals, run audits, regenerate final report.
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
OUT=$ROOT/outputs/iclr27_phase4q
OVTR_PY=/home/lwr/anaconda3/envs/ovtr/bin/python
PY=python3
LOG=$OUT/auto_finalize.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_pid() {
  local pidfile=$1 name=$2
  while [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; do
    sleep 60
  done
  log "done waiting $name"
}

wait_process_name() {
  local pat=$1 name=$2
  while pgrep -f "$pat" >/dev/null 2>&1; do
    sleep 60
  done
  log "done waiting $name"
}

eval_model() {
  local gpu=$1 ckpt=$2 mode=$3 out=$4; shift 4
  cd "$OVTR_DIR"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$OVTR_PY" eval.py \
    --config_file ./config/ovtr_lite_train_val.py \
    --dataset_file lvis_generated_img_seqs --batch_size 1 \
    --with_box_refine --two_stage --pretrain "$ckpt" \
    --score_mode "$mode" "$@" --num_workers 4 --sampler_lengths 2 \
    --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
    --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
    --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
    --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
    --output_dir "$out" --eval track \
    --result_path_track "$out/teta_results" > "$out/eval.log" 2>&1
  local rc=$?
  log "eval $out rc=$rc"
  return $rc
}

convert() {
  local results=$1 prefix=$2
  PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase4p/ovtr_main_eval.py" \
    --results-json "$results" --out-prefix "$prefix"
}

mkdir -p "$OUT"
: > "$LOG"
log "auto-finalize started"

# 1. wait for the running pilot eval (session launched separately)
wait_process_name "[e]val.py.*q2_pilot" "q2_pilot_eval"

# 2. convert pilot eval + pilot mechanism audit
convert "$OUT/q2_pilot/teta_results/tao_track.json" "$OUT/q2_pilot/proposals"
log "pilot proposals converted"

# 3. wait for Q0/Q1/Q2 long training
wait_pid "$OUT/q0_long/train.pid" q0
wait_pid "$OUT/q1_long/train.pid" q1
wait_pid "$OUT/q2_long/train.pid" q2
log "all long trainings finished"

# 4. parallel evals (3 GPUs)
eval_model 1 "$OUT/q0_long/checkpoint.pth" base "$OUT/q0_long" &
P0_EVAL=$!
eval_model 2 "$OUT/q1_long/checkpoint.pth" tco "$OUT/q1_long" \
  --tco_loss_coef 1.0 --tco_alpha 0.5 &
P1_EVAL=$!
eval_model 3 "$OUT/q2_long/checkpoint.pth" dscq "$OUT/q2_long" \
  --dscq_loss_coef 1.0 --dscq_alpha 0.5 \
  --dscq_stats_path "$OUT/q2_long/dscq_stats.json" &
P2_EVAL=$!
wait "$P0_EVAL" "$P1_EVAL" "$P2_EVAL"
log "all evals finished"

# 5. convert
convert "$OUT/q0_long/teta_results/tao_track.json" "$OUT/q0_long/proposals"
convert "$OUT/q1_long/teta_results/tao_track.json" "$OUT/q1_long/proposals"
convert "$OUT/q2_long/teta_results/tao_track.json" "$OUT/q2_long/proposals"
log "proposals converted"

# 5b. P1+ control on the new models' proposals
for model in q0_long q1_long q2_long; do
  "$PY" "$ROOT/src/iclr27_phase4q/p1plus_confirmation.py" \
    --dev-csv "$OUT/$model/proposals_dev.csv" \
    --heldout-csv "$OUT/$model/proposals_heldout.csv" \
    --out-dir "$OUT/p1plus/on_$model"
done
log "p1plus on Q0/Q1/Q2 done"

# 6. mechanism audit + comparison + report
"$PY" "$ROOT/src/iclr27_phase4q/dscq_mechanism_audit.py" \
  --dscq-stats "$OUT/q2_long/dscq_stats.json" \
  --dev-csv "$OUT/q2_long/proposals_dev.csv" \
  --heldout-csv "$OUT/q2_long/proposals_heldout.csv" \
  --out-json "$OUT/audits/dscq_mechanism.json"
"$PY" "$ROOT/src/iclr27_phase4q/q0_q1_q2_compare.py" \
  --out-json "$OUT/audits/q0_q1_q2_comparison.json"
"$PY" "$ROOT/src/iclr27_phase4q/generate_final_report.py"
log "auto-finalize done"
touch "$OUT/auto_finalize.done"
