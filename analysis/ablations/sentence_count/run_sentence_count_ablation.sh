#!/usr/bin/env bash
set -Eeuo pipefail

# Run sentence-count detection on existing records for additional datasets.
# Default: Wiki + Dolly (C4 was already tested). No generation/attacks rerun.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_DIR:-}" ]]; then ROOT="$PROJECT_DIR"; else
  ROOT="$SCRIPT_DIR"
  while [[ "$ROOT" != "/" && ! -d "$ROOT/src" ]]; do ROOT="$(dirname "$ROOT")"; done
fi
[[ -d "$ROOT/src" ]] || { echo "ERROR: cannot find IASW project root containing src/. Set PROJECT_DIR." >&2; exit 1; }
PROJECT_DIR="$ROOT"; cd "$PROJECT_DIR"
export PROJECT_DIR PROJECT_ROOT="$PROJECT_DIR" PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NLTK_DATA="${NLTK_DATA:-/home/share/models/nltk_data}"

DATASETS="${DATASETS:-c4}"
MODEL_NAME="${MODEL_NAME:-llama3.1-8b-instruct}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/llama3.1-8b-instruct}"
EMBEDDER_PATH="${EMBEDDER_PATH:-/home/share/models/bge-small-en-v1.5}"
METHOD="${METHOD:-universal_dual}"
MICRO_METHOD="${MICRO_METHOD:-kgw}"
MACRO_METHOD="${MACRO_METHOD:-iasw}"
TARGET_LEN="${TARGET_LEN:-500}"
SENTENCE_COUNTS="${SENTENCE_COUNTS:-1,2,3,4,5,6,7,8,9,10}"
COHORT="${COHORT:-fixed}"                 # fixed = same records for every K
NEGATIVE_SOURCE="${NEGATIVE_SOURCE:-human}" # human or no_wm
NUM_SAMPLES="${NUM_SAMPLES:-0}"             # 0 = all records
STATS_KEY="${STATS_KEY:-rank64_eps0p0001_calibskip3000_calibn1000}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-dp_local}"
DEVICE="${DEVICE:-cuda:0}"
BOUNDARY_TPR1="${BOUNDARY_TPR1:-0.80}"
BOUNDARY_TPR5="${BOUNDARY_TPR5:-0.90}"
PY_SCRIPT="${PY_SCRIPT:-$SCRIPT_DIR/analysis_sentence_count_ablation.py}"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: Python script not found: $PY_SCRIPT" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "ERROR: model path not found: $MODEL_PATH" >&2; exit 1; }
[[ -d "$EMBEDDER_PATH" ]] || { echo "ERROR: embedder path not found: $EMBEDDER_PATH" >&2; exit 1; }

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
SUCCESS=0; SKIPPED=0
for DS in "${DATASET_LIST[@]}"; do
  DS="$(echo "$DS" | xargs)"; [[ -n "$DS" ]] || continue
  INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/results/ablations/token_length/len500/records/dual_kgw_iasw.jsonl}"
  # If INPUT_FILE was supplied explicitly, use it only for a single-dataset run.
  if [[ -n "${INPUT_FILE_OVERRIDE:-}" ]]; then INPUT_FILE="$INPUT_FILE_OVERRIDE"; fi
  OUT="$PROJECT_DIR/results/ablations/sentence_count/${DS}_${MODEL_NAME}_sentences"
  if [[ ! -f "$INPUT_FILE" ]]; then
    echo "[SKIP] $DS input not found: $INPUT_FILE"
    ((SKIPPED+=1)); unset INPUT_FILE
    continue
  fi
  mkdir -p "$OUT" "$PROJECT_DIR/results/ablations/sentence_count/logs"
  LOG="$PROJECT_DIR/results/ablations/sentence_count/logs/${DS}_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
  echo "============================================================"
  echo "Dataset       : $DS"
  echo "Input         : $INPUT_FILE"
  echo "Model         : $MODEL_PATH"
  echo "Sentence Ks   : $SENTENCE_COUNTS"
  echo "Cohort        : $COHORT"
  echo "Output        : $OUT"
  echo "============================================================"
  python "$PY_SCRIPT" \
    --input_file "$INPUT_FILE" --output_dir "$OUT" \
    --model_path "$MODEL_PATH" --embedder_path "$EMBEDDER_PATH" \
    --dataset_type "$DS" --method "$METHOD" \
    --micro_method "$MICRO_METHOD" --macro_method "$MACRO_METHOD" \
    --target_len "$TARGET_LEN" --stats_key "$STATS_KEY" \
    --alignment_mode "$ALIGNMENT_MODE" --sentence_counts "$SENTENCE_COUNTS" \
    --cohort "$COHORT" --negative_source "$NEGATIVE_SOURCE" \
    --num_samples "$NUM_SAMPLES" --device "$DEVICE" \
    --boundary_tpr1 "$BOUNDARY_TPR1" --boundary_tpr5 "$BOUNDARY_TPR5" 2>&1 | tee "$LOG"
  ((SUCCESS+=1)); unset INPUT_FILE
  if [[ -n "${INPUT_FILE_OVERRIDE:-}" ]]; then break; fi
done
if (( SUCCESS == 0 )); then echo "ERROR: no dataset was evaluated. Check dataset names and existing input JSONL files." >&2; exit 1; fi
echo "Finished. Successful datasets: $SUCCESS; skipped missing datasets: $SKIPPED"