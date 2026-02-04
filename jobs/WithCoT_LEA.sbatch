#!/usr/bin/env bash
#SBATCH --job-name=llava-QCM-LEA
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a100:2
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/llava/LLaVA_DL_project/logs/%x-%j.out
#SBATCH --error=/scratch/%u/llava/LLaVA_DL_project/logs/%x-%j.err

set -euo pipefail

# Workspace root (match your working job)
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

RUNS_ROOT="$WORK_ROOT/runs/QCM-LEA"
mkdir -p "$RUNS_ROOT"

STAMP="$(date +%F_%H-%M-%S)"
JOB="${SLURM_JOB_ID:-nojob}"
RUN_NAME="${STAMP}-llava15_7b-lora_r32a64-lr1e-5-e2-job${JOB}"
OUTDIR="$RUNS_ROOT/$RUN_NAME"
mkdir -p "$OUTDIR"
ln -sfn "$OUTDIR" "$RUNS_ROOT/latest"


# If you want deterministic-ish runs:
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
apptainer exec --nv \
  -B "$WORK_ROOT":"$WORK_ROOT" \
  -B /cvmfs:/cvmfs \
  --env WORK_ROOT="$WORK_ROOT" \
  --env LLAVA_ROOT="$LLAVA_ROOT" \
  --env SQA_ROOT="$SQA_ROOT" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TRAIN_JSON="$TRAIN_JSON" \
  --env IMG_DIR="$IMG_DIR" \
  --env OUTDIR="$OUTDIR" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  "$SIF" bash -lc '
    set -euo pipefail
    cd "$LLAVA_ROOT"
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"

    PY="$LLAVA_ROOT/.venv/bin/python"
    TR="$LLAVA_ROOT/.venv/bin/torchrun"

    mkdir -p "$OUTDIR"/{logs,config,metrics}
    echo "=== CMD ===" | tee "$OUTDIR/config/command.txt"
    printf "%q " "$0" "$@" | tee -a "$OUTDIR/config/command.txt" || true

    # Record git commit if repo is a git checkout
    ( cd "$LLAVA_ROOT" && git rev-parse HEAD ) > "$OUTDIR/config/git_commit.txt" 2>/dev/null || true

    # Record python + pip freeze
    $PY -V > "$OUTDIR/config/python_version.txt"
    $PY -m pip freeze > "$OUTDIR/config/pip_freeze.txt"
    nvidia-smi > "$OUTDIR/config/nvidia_smi.txt" 2>/dev/null || true


    echo "=== ENV ==="
    echo "WORK_ROOT=$WORK_ROOT"
    echo "LLAVA_ROOT=$LLAVA_ROOT"
    echo "SQA_ROOT=$SQA_ROOT"
    echo "HF_HOME=$HF_HOME"
    echo "HF_HUB_CACHE=$HF_HUB_CACHE"
    echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    $PY -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"ngpu\", torch.cuda.device_count())"
    nvidia-smi -L || true

    test -f "$TRAIN_JSON" || { echo "Missing: $TRAIN_JSON"; exit 2; }
    test -d "$IMG_DIR"    || { echo "Missing: $IMG_DIR"; exit 2; }

    # DDP settings
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export OMP_NUM_THREADS=8
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

    export TORCH_CHECKPOINT_USE_REENTRANT=0


    # IMPORTANT: SLURM gives you 2 GPUs; use them.
    NPROC=2
    echo "=== starting DDP: nproc_per_node=$NPROC ==="

    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
      llava/train/train_mem.py \
      --model_name_or_path liuhaotian/llava-v1.5-7b \
      --version v1 \
      --data_path "$TRAIN_JSON" \
      --image_folder "$IMG_DIR" \
      --is_multimodal True \
      --vision_tower openai/clip-vit-large-patch14-336 \
      --mm_projector_type mlp2x_gelu \
      --mm_vision_select_layer -2 \
      --mm_use_im_start_end False \
      --mm_use_im_patch_token False \
      --output_dir "$OUTDIR" \
      --num_train_epochs 2 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --weight_decay 0.0 \
      --logging_steps 10 \
      --save_strategy epoch \
      --save_total_limit 3 \
      --evaluation_strategy no \
      --report_to none \
      --gradient_checkpointing True \
      --ddp_find_unused_parameters False \
      --lazy_preprocess True \
      --dataloader_num_workers 4 \
      --model_max_length 2048 \
      --bf16 True \
      --fp16 False \
      --tf32 True \
      --lora_enable True \
      --lora_r 32 \
      --lora_alpha 64 \
      --lora_dropout 0.05
  '
