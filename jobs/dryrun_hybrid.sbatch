#!/usr/bin/env bash
#SBATCH --job-name=llava-dryrun
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/%u/LLaVA_DL_project/logs/%x-%j.out
#SBATCH --error=/scratch/%u/LLaVA_DL_project/logs/%x-%j.err

set -euo pipefail

# ---- job-specific knobs (edit here only) ----
export MODEL_NAME="liuhaotian/llava-v1.5-7b"
export MAX_STEPS=5
export OUTDIR="/scratch/$USER/llava_dryrun_out"

# (Optional) override defaults if needed
export LLAVA_ROOT="/scratch/$USER/LLaVA_DL_project"
export SQA_ROOT="/scratch/$USER/datasets/ScienceQA/data/scienceqa"

# Run inside container via wrapper, but keep the command inline:
"$LLAVA_ROOT/scripts/llava_job_hybrid.sh" bash -s <<'EOF'
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
mkdir -p "$OUTDIR"

test -f "$TRAIN_JSON" || { echo "Missing: $TRAIN_JSON"; exit 2; }
test -d "$IMG_DIR"     || { echo "Missing: $IMG_DIR"; exit 2; }
test -f "./scripts/zero2.json" || { echo "Missing: ./scripts/zero2.json"; exit 2; }

"$DS" llava/train/train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path "$MODEL_NAME" \
  --data_path "$TRAIN_JSON" \
  --image_folder "$IMG_DIR" \
  --output_dir "$OUTDIR" \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
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
EOF
