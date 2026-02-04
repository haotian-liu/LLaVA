#!/usr/bin/env bash
#SBATCH --job-name=masking-test
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=/scratch/%u/llava/LLaVA_DL_project/logs/%x-%j.out
#SBATCH --error=/scratch/%u/llava/LLaVA_DL_project/logs/%x-%j.err

set -euo pipefail

# New workspace root
export WORK_ROOT="/scratch/$USER/llava"

export LLAVA_ROOT="$WORK_ROOT/LLaVA_DL_project"
export SQA_ROOT="$WORK_ROOT/datasets/ScienceQA/data/scienceqa"
export SIF="$LLAVA_ROOT/llava_dev.sif"

export HF_HOME="$WORK_ROOT/hf_home"
export HF_HUB_CACHE="$WORK_ROOT/hf_cache"
export TRITON_CACHE_DIR="$WORK_ROOT/triton_cache"
mkdir -p "$LLAVA_ROOT/logs" "$HF_HOME" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR"

TRAIN_JSON="$SQA_ROOT/llava_train_QCM-LEA.json"
IMG_DIR="$SQA_ROOT/images/train"
OUTDIR="$WORK_ROOT/tmp_masking_test"
mkdir -p "$OUTDIR"

apptainer exec --nv \
  -B "$WORK_ROOT":"$WORK_ROOT" \
  -B /cvmfs:/cvmfs \
  --env WORK_ROOT="$WORK_ROOT",LLAVA_ROOT="$LLAVA_ROOT",SQA_ROOT="$SQA_ROOT",HF_HOME="$HF_HOME",HF_HUB_CACHE="$HF_HUB_CACHE",TRITON_CACHE_DIR="$TRITON_CACHE_DIR",TRAIN_JSON="$TRAIN_JSON",IMG_DIR="$IMG_DIR",OUTDIR="$OUTDIR",CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  "$SIF" bash -lc '
    set -euo pipefail
    cd "$LLAVA_ROOT"
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    export DEBUG_MASK=1

    PY="$LLAVA_ROOT/.venv/bin/python"

    echo "=== ENV ==="
    echo "HF_HOME=$HF_HOME"
    echo "HF_HUB_CACHE=$HF_HUB_CACHE"
    echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    $PY -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"ngpu\", torch.cuda.device_count())"
    nvidia-smi -L || true

    # Avoid the DeepSpeed CUDA_HOME compile issue for this test
    # $PY -m pip uninstall -y deepspeed || true

    test -f "$TRAIN_JSON" || { echo "Missing: $TRAIN_JSON"; exit 2; }
    test -d "$IMG_DIR"    || { echo "Missing: $IMG_DIR"; exit 2; }

    echo "=== RUN (1 step) ==="
    $PY llava/train/train_mem.py \
      --model_name_or_path liuhaotian/llava-v1.5-7b \
      --version v1 \
      --data_path "$TRAIN_JSON" \
      --image_folder "$IMG_DIR" \
      --is_multimodal True \
      --vision_tower openai/clip-vit-large-patch14-336 \
      --mm_projector_type mlp2x_gelu \
      --mm_vision_select_layer -2 \
      --output_dir "$OUTDIR" \
      --max_steps 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --warmup_ratio 0.0 \
      --logging_steps 1 \
      --save_strategy no \
      --save_steps 0 \
      --report_to none \
      --evaluation_strategy no \
      --model_max_length 1024 \
      --lora_enable --lora_r 32 --lora_alpha 64 \
      --bf16 \
      --lazy_preprocess False
  '
