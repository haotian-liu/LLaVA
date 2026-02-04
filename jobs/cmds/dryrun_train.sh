#!/usr/bin/env bash
set -euo pipefail
cd "$LLAVA_ROOT"

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PY="$LLAVA_ROOT/.venv/bin/python"
DS="$LLAVA_ROOT/.venv/bin/deepspeed"

# fail fast
$PY -c 'import google.protobuf, sentencepiece; print("tokenizer deps ok")'
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
$PY -c 'import deepspeed; print("deepspeed import ok")'

TRAIN_JSON="$SQA_ROOT/llava_train_QCM-LEA.json"
IMG_DIR="$SQA_ROOT/images/train"
OUTDIR="/scratch/$USER/llava_dryrun_out"
mkdir -p "$OUTDIR"

test -f "$TRAIN_JSON" || { echo "Missing: $TRAIN_JSON"; exit 2; }
test -d "$IMG_DIR"     || { echo "Missing: $IMG_DIR"; exit 2; }
test -f "./scripts/zero2.json" || { echo "Missing: ./scripts/zero2.json"; exit 2; }

"$DS" llava/train/train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path liuhaotian/llava-v1.5-7b \
  --data_path "$TRAIN_JSON" \
  --image_folder "$IMG_DIR" \
  --output_dir "$OUTDIR" \
  --num_train_epochs 1 \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --logging_steps 1 \
  --save_strategy no \
  --evaluation_strategy no \
  --bf16 True \
  --tf32 True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True
