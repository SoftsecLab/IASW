#!/usr/bin/env bash
set -Eeuo pipefail

# KGW+IAS-W whitening-rank sweep: 256, 128, 64, 32.
# Two concurrent workers: physical GPU 0 runs 256 then 64;
# physical GPU 1 runs 128 then 32. Every rank has isolated stats/results.
# No existing main-experiment files are overwritten.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCAN="$SCRIPT_DIR"
while [[ "$SCAN" != "/" && ! -f "$SCAN/src/watermarks/runner.py" ]]; do SCAN="$(dirname "$SCAN")"; done
PROJECT_DIR="${PROJECT_DIR:-$SCAN}"
[[ -f "$PROJECT_DIR/src/watermarks/runner.py" ]] || { echo "[ERROR] Set PROJECT_DIR to the IASW root" >&2; exit 1; }
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export NLTK_DATA="${NLTK_DATA:-/home/share/models/nltk_data}"

DATASET="${DATASET:-c4}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/llama3.1-8b-instruct}"
EMBEDDER_PATH="${EMBEDDER_PATH:-/home/share/models/bge-small-en-v1.5}"
TARGET_LEN="${TARGET_LEN:-200}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
NUM_CANDIDATES="${NUM_CANDIDATES:-32}"
GENERATION_SEED="${GENERATION_SEED:-20261017}"
SECRET_KEY="${SECRET_KEY:-DUAL_MACRO_2026}"
EPS="${EPS:-0.0001}"
CALIBRATION_NUM_SAMPLES="${CALIBRATION_NUM_SAMPLES:-1000}"
MAX_STATS_SENTENCES="${MAX_STATS_SENTENCES:-20000}"
MAX_CALIBRATION_TEXTS="${MAX_CALIBRATION_TEXTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_NAME="$(basename "$MODEL_PATH")"

GEN_BASELINE="${GEN_BASELINE:-$PROJECT_DIR/data/main_exp/baseline_triplets/$DATASET/$MODEL_NAME/baseline_len${TARGET_LEN}.jsonl}"
CAL_BASELINE="${CAL_BASELINE:-$PROJECT_DIR/data/calibration_set/$DATASET/$MODEL_NAME/ias_baseline_len${TARGET_LEN}_skip3000_n1000.jsonl}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/results/ablations/whitening_rank_sweep/$DATASET/$MODEL_NAME/len${TARGET_LEN}}"
STATS_ROOT="${STATS_ROOT:-$PROJECT_DIR/cache/iasw_stats/whitening_rank_sweep/$DATASET/$MODEL_NAME/len${TARGET_LEN}}"
LOG_ROOT="$OUT_ROOT/logs"

PARAPHRASE_PATH="${PARAPHRASE_PATH:-/home/share/models/tuner007_pegasus_paraphrase}"
DIPPER_PATH="${DIPPER_PATH:-/home/share/models/dipper-paraphraser-xxl}"
FWD_TRANS_PATH="${FWD_TRANS_PATH:-/home/share/models/opus-mt-en-fr}"
BWD_TRANS_PATH="${BWD_TRANS_PATH:-/home/share/models/opus-mt-fr-en}"
NLLB_PATH="${NLLB_PATH:-facebook/nllb-200-distilled-600M}"
# Keep the same nine attacks as the completed no-whitening run for direct comparison.

[[ -f "$GEN_BASELINE" ]] || { echo "[ERROR] Missing generation baseline: $GEN_BASELINE" >&2; exit 1; }
[[ -f "$CAL_BASELINE" ]] || { echo "[ERROR] Missing calibration baseline: $CAL_BASELINE" >&2; exit 1; }
[[ -f "$PROJECT_DIR/src/calibration/calibrate_empirical_iasw.py" ]] || { echo "[ERROR] Missing calibration script" >&2; exit 1; }
mkdir -p "$OUT_ROOT" "$STATS_ROOT" "$LOG_ROOT"

run_one_rank() {
  local rank="$1"
  local eps_tag="0p0001"
  local stats_file="$STATS_ROOT/ias_stats_${DATASET}_${MODEL_NAME}_len${TARGET_LEN}_rank${rank}_eps${eps_tag}_calibskip3000_calibn1000.pt"
  local unit_dir="$OUT_ROOT/rank${rank}"
  local target="$unit_dir/dual_kgw_iasw_rank${rank}.jsonl"
  local plot_dir="$unit_dir/plots"
  local plot_pdf="$plot_dir/dual_kgw_iasw_rank${rank}_roc_combined.pdf"
  local stats_key="rank${rank}_eps${eps_tag}_calibskip3000_calibn1000"
  local dual_args=(--micro_method kgw --macro_method iasw)

  mkdir -p "$unit_dir" "$plot_dir"
  if [[ -e "$target" ]]; then
    echo "[ERROR] Existing target prevents accidental overwrite: $target" >&2
    echo "Move it aside or set OUT_ROOT to a new directory." >&2
    return 1
  fi

  echo "========== RANK $rank =========="
  echo "GPU_VISIBLE=${CUDA_VISIBLE_DEVICES:-unset} DEVICE=${DEVICE:-cuda:0}"
  echo "stats=$stats_file"
  echo "target=$target"

  # Build whitening + profile-specific calibration for this rank.
  # Do not use --update_iasw_mark: the project registry must remain untouched.
  python src/calibration/calibrate_empirical_iasw.py \
    --mode full \
    --baseline_file "$CAL_BASELINE" \
    --stats_file "$stats_file" \
    --dataset_type "$DATASET" \
    --model_path "$MODEL_PATH" \
    --embedder_path "$EMBEDDER_PATH" \
    --seq_len "$TARGET_LEN" \
    --rank "$rank" \
    --eps "$EPS" \
    --num_samples "$CALIBRATION_NUM_SAMPLES" \
    --max_stats_sentences "$MAX_STATS_SENTENCES" \
    --max_calibration_texts "$MAX_CALIBRATION_TEXTS" \
    --batch_size "$BATCH_SIZE" \
    --stats_key "$stats_key" \
    --secret_key "$SECRET_KEY" \
    --device "${DEVICE:-cuda:0}"

  # Verify rank and dual profile calibration before generation.
  python -c 'import sys,torch; p,r=sys.argv[1],int(sys.argv[2]); s=torch.load(p,map_location="cpu",weights_only=False); assert int(s["components"].shape[1])==r,(s["components"].shape,r); c=s["calibration_profiles"]["dual"]["calibration"]["dp_local"]; assert int(c["n"])>0 and float(c["sigma"])>0,c; print("[OK] rank",r,"dual dp_local calibration:",c)' "$stats_file" "$rank"

  export IASW_STATS_FILE="$stats_file"
  export IASW_STATS_KEY=""
  export IASW_ALIGNMENT_MODE=dp_local
  export IAS_ALIGNMENT_MODE=dp_local
  export IASW_TARGET_LEN="$TARGET_LEN"
  export IAS_TARGET_LEN="$TARGET_LEN"

  # Generate
  python src/watermarks/runner.py \
    --task generate --baseline_file "$GEN_BASELINE" --target_file "$target" \
    --dataset_type "$DATASET" --method universal_dual "${dual_args[@]}" \
    --model_path "$MODEL_PATH" --embedder_path "$EMBEDDER_PATH" \
    --max_total_tokens "$TARGET_LEN" --max_sent_tokens 40 \
    --num_candidates "$NUM_CANDIDATES" --num_samples "$NUM_SAMPLES" \
    --generation_seed "$GENERATION_SEED" --secret_key "$SECRET_KEY" --device "${DEVICE:-cuda:0}"

  # Clean / baseline detection
  python src/watermarks/runner.py \
    --task detect --target_file "$target" --dataset_type "$DATASET" \
    --method universal_dual "${dual_args[@]}" \
    --model_path "$MODEL_PATH" --embedder_path "$EMBEDDER_PATH" \
    --max_total_tokens "$TARGET_LEN" --max_sent_tokens 40 \
    --num_candidates "$NUM_CANDIDATES" --mode baseline_eval \
    --secret_key "$SECRET_KEY" --device "${DEVICE:-cuda:0}"

  # Attacks + attacked-text detection
  for attack in "${ATTACKS[@]}"; do
    echo "[RANK $rank] attack=$attack"
    python src/attacks/run_attack.py \
      --target_file "$target" --attack_name "$attack" \
      --dipper_path "$DIPPER_PATH" --paraphrase_path "$PARAPHRASE_PATH" \
      --fwd_trans_path "$FWD_TRANS_PATH" --bwd_trans_path "$BWD_TRANS_PATH" \
      --nllb_path "$NLLB_PATH"

    python src/watermarks/runner.py \
      --task detect --target_file "$target" --dataset_type "$DATASET" \
      --method universal_dual "${dual_args[@]}" \
      --model_path "$MODEL_PATH" --embedder_path "$EMBEDDER_PATH" \
      --max_total_tokens "$TARGET_LEN" --max_sent_tokens 40 \
      --num_candidates "$NUM_CANDIDATES" --mode attack_eval --attack_name "$attack" \
      --secret_key "$SECRET_KEY" --device "${DEVICE:-cuda:0}"
  done

  # Official ROC/fixed-FPR evaluator; emits PDF and metrics JSON.
  python src/evaluation/plot_roc.py \
    --input_file "$target" \
    --output_pdf "$plot_pdf" \
    --method "dual_kgw_iasw_rank${rank}" \
    --negative_source auto

  echo "[DONE] rank=$rank record=$target"
  echo "[DONE] rank=$rank ROC=$plot_pdf"
}

run_group() {
  local gpu="$1"
  shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  export DEVICE="cuda:0"
  echo "[WORKER] physical GPU $gpu; ranks: $*"
  for rank in "$@"; do run_one_rank "$rank"; done
}

# Two groups run concurrently. Within each GPU, its two ranks run sequentially.
# run_group 0 256 128 > "$LOG_ROOT/gpu0_ranks256_128.log" 2>&1 &
run_group 0 48 > "$LOG_ROOT/gpu0_ranks48.log" 2>&1 &

PID0=$!
# run_group 1 64 32 > "$LOG_ROOT/gpu1_ranks64_32.log" 2>&1 &
run_group 1 16 > "$LOG_ROOT/gpu1_ranks16.log" 2>&1 &
PID1=$!

STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1

echo
if [[ "$STATUS" -eq 0 ]]; then
  echo "[DONE] All ranks finished."
else
  echo "[ERROR] One or more GPU workers failed. Inspect:"
  # echo "  $LOG_ROOT/gpu0_ranks256_128.log"
  # echo "  $LOG_ROOT/gpu1_ranks64_32.log"
  echo "  $LOG_ROOT/gpu0_ranks48.log"
  echo "  $LOG_ROOT/gpu1_ranks16.log"
  
  exit 1
fi
echo "Output root: $OUT_ROOT"
echo "For each rank, see rankN/plots/*_roc_combined.pdf and *_metrics.json."
