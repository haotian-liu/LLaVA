#!/usr/bin/env bash
set -euo pipefail

ARG="${1:-}"
if [[ -z "$ARG" ]]; then
  echo "ERROR: missing TAG. Example: bash $0 A  or  bash $0 ALL"
  exit 2
fi

ALL_TAGS=(A AE ALE LEA)

if [[ "$ARG" == "ALL" ]]; then
  TAGS=("${ALL_TAGS[@]}")
else
  TAGS=("$ARG")
fi

WORK_ROOT="${WORK_ROOT:-/scratch/$USER/llava}"
LLAVA_ROOT="$WORK_ROOT/LLaVA_DL_project"
SQA_ROOT="$WORK_ROOT/datasets/ScienceQA/data/scienceqa"
BASE_REPO="${BASE_REPO:-liuhaotian/llava-v1.5-7b}"
PY="$LLAVA_ROOT/.venv/bin/python"

# Store merged checkpoints OUTSIDE runs/ (no /runs/.../latest/merged)
SCRATCH_MERGED_ROOT="${SCRATCH_MERGED_ROOT:-$WORK_ROOT/merged_models}"

export HF_HOME="${HF_HOME:-$WORK_ROOT/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORK_ROOT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$WORK_ROOT/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$LLAVA_ROOT:${PYTHONPATH:-}"

choose_output_format() {
  local tag="$1"
  if [[ "$tag" == "A" ]]; then
    echo "letter"
  elif [[ "$tag" == A* ]]; then
    echo "cot_answer_first"
  elif [[ "$tag" == *A ]]; then
    echo "cot_reason_first"
  else
    echo "cot_answer_first"
  fi
}

merged_has_weights() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ -f "$d/config.json" ]] || return 1

  # loadable safetensors layouts
  if [[ -f "$d/model.safetensors" ]]; then
    return 0
  fi
  if ls "$d"/model-*-of-*.safetensors >/dev/null 2>&1 && [[ -f "$d/model.safetensors.index.json" ]]; then
    return 0
  fi

  # legacy bin layouts
  if [[ -f "$d/pytorch_model.bin" ]]; then
    return 0
  fi
  if ls "$d"/pytorch_model-*-of-*.bin >/dev/null 2>&1 && [[ -f "$d/pytorch_model.bin.index.json" ]]; then
    return 0
  fi

  return 1
}

do_merge() {
  local base="$1"
  local adapter="$2"
  local out="$3"

  rm -rf "$out"
  mkdir -p "$out"

  HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" - <<PY
from peft import PeftModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

BASE="$base"
ADAPTER="$adapter"
OUT="$out"

model_name = get_model_name_from_path(BASE)
tokenizer, model, image_processor, _ = load_pretrained_model(
    BASE, model_base=None, model_name=model_name, device_map="cpu"
)

model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()

tokenizer.save_pretrained(OUT)

# safer writes: safetensors + sharding
model.save_pretrained(
    OUT,
    safe_serialization=True,
    max_shard_size="4GB"
)

try:
    if image_processor is not None:
        image_processor.save_pretrained(OUT)
except Exception:
    pass

print("merged ok:", OUT)
PY
}

run_one_tag() {
  local TAG="$1"
  local OUTPUT_FORMAT
  OUTPUT_FORMAT="$(choose_output_format "$TAG")"

  local RUN_LATEST="$WORK_ROOT/runs/QCM-${TAG}/latest"
  [[ -e "$RUN_LATEST" ]] || { echo "[SKIP] $TAG missing $RUN_LATEST"; return 0; }

  local RUN_REAL
  RUN_REAL="$(readlink -f "$RUN_LATEST")"
  [[ -d "$RUN_REAL" ]] || { echo "[SKIP] $TAG broken symlink"; return 0; }

  local QUESTION_FILE="$SQA_ROOT/llava_test_QCM-${TAG}.json"
  local IMAGE_DIR="$SQA_ROOT/images/test"
  [[ -f "$QUESTION_FILE" ]] || { echo "[SKIP] $TAG missing $QUESTION_FILE"; return 0; }
  [[ -d "$IMAGE_DIR" ]] || { echo "[SKIP] $TAG missing $IMAGE_DIR"; return 0; }

  local CKPT
  CKPT="$(ls -d "$RUN_REAL"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
  [[ -n "$CKPT" ]] || { echo "[SKIP] $TAG no checkpoint-*"; return 0; }

  # merged dir OUTSIDE run folder (uses real run id, not "latest")
  local RUN_ID
  RUN_ID="$(basename "$RUN_REAL")"
  local MERGED_DIR="$SCRATCH_MERGED_ROOT/QCM-${TAG}/${RUN_ID}/merged"
  mkdir -p "$(dirname "$MERGED_DIR")"

  local STAMP
  STAMP="$(date +%F_%H-%M-%S)"
  local EVAL_DIR="$RUN_REAL/artifacts/$STAMP"
  mkdir -p "$EVAL_DIR"

  local PRED_FILE="$EVAL_DIR/preds.jsonl"
  local EVAL_FILE="$EVAL_DIR/output.jsonl"
  local RESULT_FILE="$EVAL_DIR/result.json"

  local ANS_ROOT="$WORK_ROOT/answers/QCM-${TAG}"
  mkdir -p "$ANS_ROOT"

  echo "============================================================"
  echo "TAG=$TAG OUTPUT_FORMAT=$OUTPUT_FORMAT"
  echo "RUN_REAL=$RUN_REAL"
  echo "CKPT=$CKPT"
  echo "MERGED_DIR=$MERGED_DIR"
  echo "EVAL_DIR=$EVAL_DIR"
  echo "============================================================"

  cd "$LLAVA_ROOT"

  if [[ "${FORCE_REMERGE:-0}" == "1" ]]; then
    echo "[MERGE] FORCE_REMERGE=1"
    do_merge "$BASE_REPO" "$CKPT" "$MERGED_DIR"
  else
    if merged_has_weights "$MERGED_DIR"; then
      echo "[MERGE] already merged -> skip"
    else
      echo "[MERGE] merging -> $MERGED_DIR"
      do_merge "$BASE_REPO" "$CKPT" "$MERGED_DIR"
    fi
  fi

  if ! merged_has_weights "$MERGED_DIR"; then
    echo "ERROR: merged checkpoint incomplete: $MERGED_DIR"
    exit 3
  fi

  echo "[INFER] -> $PRED_FILE"
  HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -m llava.eval.model_vqa_science \
    --model-path "$MERGED_DIR" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_DIR" \
    --answers-file "$PRED_FILE" \
    --output-format "letter" \
    --temperature 0 \
    --conv-mode llava_v1

  echo "[EVAL] -> $RESULT_FILE"
  "$PY" llava/eval/eval_science_qa.py \
    --base-dir "$SQA_ROOT" \
    --result-file "$PRED_FILE" \
    --output-file "$EVAL_FILE" \
    --output-result "$RESULT_FILE"

  mkdir -p "$RUN_REAL/artifacts"
  ln -sfn "$EVAL_DIR" "$RUN_REAL/artifacts/latest_eval"
  ln -sfn "$PRED_FILE"   "$ANS_ROOT/latest_pred.jsonl"
  ln -sfn "$EVAL_FILE"   "$ANS_ROOT/latest_output.jsonl"
  ln -sfn "$RESULT_FILE" "$ANS_ROOT/latest_result.json"

  echo "[DONE] $TAG"
}

mkdir -p "$SCRATCH_MERGED_ROOT"

for t in "${TAGS[@]}"; do
  run_one_tag "$t"
done

echo "All requested tags finished: ${TAGS[*]}"
