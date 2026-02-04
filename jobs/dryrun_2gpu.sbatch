#!/usr/bin/env bash
#SBATCH --job-name=llava-dryrun-2g
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gpus-per-node=l40s:2
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/LLaVA_DL_project/logs/%x-%j.out
#SBATCH --error=/scratch/%u/LLaVA_DL_project/logs/%x-%j.err

set -euo pipefail

export LLAVA_ROOT="/scratch/$USER/LLaVA_DL_project"
export SQA_ROOT="/scratch/$USER/datasets/ScienceQA/data/scienceqa"
export SIF="/scratch/$USER/LLaVA_DL_project/llava_dev.sif"

export HF_HOME="/scratch/$USER/hf_home"
export HF_HUB_CACHE="/scratch/$USER/hf_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
mkdir -p "$LLAVA_ROOT/logs" "$HF_HOME" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR"

module --force purge
module load StdEnv
module load CUDA/12.1 2>/dev/null || module load cuda/12.1 2>/dev/null || module load CUDA/12.1.0 2>/dev/null
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

TRAIN_JSON="$SQA_ROOT/llava_train_QCM-LEA.json"
IMG_DIR="$SQA_ROOT/images/train"
OUTDIR="/scratch/$USER/llava_dryrun_out_2gpu"
mkdir -p "$OUTDIR"

    set -euo pipefail
    cd "$LLAVA_ROOT"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

    PY="$LLAVA_ROOT/.venv/bin/python"

    echo "=== sanity ==="
    $PY -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"ngpu\", torch.cuda.device_count())"
    command -v nvidia-smi >/dev/null && nvidia-smi -L || true

    test -f "$TRAIN_JSON" || { echo "Missing: $TRAIN_JSON"; exit 2; }
    test -d "$IMG_DIR"    || { echo "Missing: $IMG_DIR"; exit 2; }

    echo "torchrun path: $LLAVA_ROOT/.venv/bin/torchrun"
    "$LLAVA_ROOT/.venv/bin/torchrun" --version 2>/dev/null || true

   GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    if echo "$GPU_NAME" | grep -qiE "A100|A30|A40|A10|L40|L4|H100|RTX A6000"; then
        PREC_ARGS=(--bf16 True --fp16 False --tf32 True)
    else
        # V100 (Volta): use fp16 but avoid GradScaler unscale/clip path
        PREC_ARGS=(--bf16 False --fp16 False --tf32 False)
    fi
    echo "GPU=$GPU_NAME -> ${PREC_ARGS[*]}"


    # Fix DDP + grad checkpointing re-entrant backward + LoRA
    # export TORCH_CHECKPOINT_USE_REENTRANT=0

########################################################
# FOR LOOKING
########################################################
    echo "=== starting DDP ==="
    "$LLAVA_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node="$NPROC" \
      llava/train/train_mem.py \
      \
      # Enable LoRA fine-tuning (only small adapter weights are trained)
      --lora_enable True \
      \
      # LoRA rank: controls adapter capacity (8 = small, memory-efficient)
      --lora_r 32 \
      \
      # LoRA scaling factor (effective update strength = alpha / r)
      --lora_alpha 64 \
      \
      # Dropout applied inside LoRA adapters (regularization)
      --lora_dropout 0.05 \
      \
      # Base LLaVA model (LLaMA-2 7B + vision alignment)
      --model_name_or_path liuhaotian/llava-v1.5-7b \
      \
      # Vision encoder used to extract image features (CLIP ViT-L/14 @336px)
      --vision_tower openai/clip-vit-large-patch14-336 \
      \
      # Projector that maps CLIP features into LLM embedding space
      # mlp2x_gelu = 2-layer MLP with GELU activations (standard LLaVA choice)
      --mm_projector_type mlp2x_gelu \
      \
      # Which CLIP layer to take features from
      # -2 = second-to-last layer (more semantic, less noisy than final)
      --mm_vision_select_layer -2 \
      \
      # Do NOT wrap images with special <im_start> / <im_end> tokens
      --mm_use_im_start_end False \
      \
      # Do NOT insert per-patch image tokens (use pooled vision features)
      --mm_use_im_patch_token False \
      \
      # Path to ScienceQA / LLaVA JSON training file
      --data_path "$TRAIN_JSON" \
      \
      # Directory containing training images
      --image_folder "$IMG_DIR" \
      \
      # Where checkpoints and logs are written
      --output_dir "$OUTDIR" \
      \
      # Number of full passes over the dataset
      # (ignored if max_steps is set)
      --num_train_epochs 1 \
      \
      # Batch size PER GPU (LoRA → usually 1)
      --per_device_train_batch_size 1 \
      \
      # Number of steps to accumulate gradients before optimizer step
      --gradient_accumulation_steps 16 \
      \
      # AdamW learning rate (2e-5 is typical for LoRA)
      --learning_rate 1e-5 \
      \
      # Log training metrics every N steps
      --logging_steps 10 \
      \
      # Save checkpoints based on step count
      --save_strategy steps \
      \
      # Save every N steps
      --save_steps 50 \
      \
      # Keep only the most recent checkpoint
      --save_total_limit 2 \
      \
      # Disable evaluation loop entirely
      --evaluation_strategy no \
      \
      # Precision flags chosen based on GPU (bf16 on A100, fp16 on V100)
      "${PREC_ARGS[@]}" \
      \
      # Disable activation recomputation to simplify DDP graph
      # (enable later if you need memory savings)
      --gradient_checkpointing True \
      \
      # Preprocess samples lazily instead of upfront (saves RAM, faster startup)
      --lazy_preprocess True \
      \
      # Disable multi-worker dataloading (more stable on shared HPC FS)
      --dataloader_num_workers 0 \
      \
      # Required for LoRA + DDP to avoid extra autograd traversal
      --ddp_find_unused_parameters False \
      \
      # Maximum sequence length (text + image tokens)
      --model_max_length 1024
########################################################


apptainer exec --nv \
  -B /scratch/$USER:/scratch/$USER \
  -B /cvmfs:/cvmfs \
  --env LLAVA_ROOT="$LLAVA_ROOT",SQA_ROOT="$SQA_ROOT",HF_HOME="$HF_HOME",HF_HUB_CACHE="$HF_HUB_CACHE",TRITON_CACHE_DIR="$TRITON_CACHE_DIR",CUDA_HOME="$CUDA_HOME",TRAIN_JSON="$TRAIN_JSON",IMG_DIR="$IMG_DIR",OUTDIR="$OUTDIR",NPROC=2 \
  "$SIF" bash -lc '
    set -euo pipefail
    cd "$LLAVA_ROOT"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

    PY="$LLAVA_ROOT/.venv/bin/python"

    echo "=== sanity ==="
    $PY -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"ngpu\", torch.cuda.device_count())"
    command -v nvidia-smi >/dev/null && nvidia-smi -L || true

    test -f "$TRAIN_JSON" || { echo "Missing: $TRAIN_JSON"; exit 2; }
    test -d "$IMG_DIR"    || { echo "Missing: $IMG_DIR"; exit 2; }

    echo "torchrun path: $LLAVA_ROOT/.venv/bin/torchrun"
    "$LLAVA_ROOT/.venv/bin/torchrun" --version 2>/dev/null || true

   GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    if echo "$GPU_NAME" | grep -qiE "A100|A30|A40|A10|L40|L4|H100|RTX A6000"; then
        PREC_ARGS=(--bf16 True --fp16 False --tf32 True)
    else
        # V100 (Volta): use fp16 but avoid GradScaler unscale/clip path
        PREC_ARGS=(--bf16 False --fp16 False --tf32 False)
    fi
    echo "GPU=$GPU_NAME -> ${PREC_ARGS[*]}"


    # Fix DDP + grad checkpointing re-entrant backward + LoRA
    # export TORCH_CHECKPOINT_USE_REENTRANT=0


    echo "=== starting DDP ==="
    "$LLAVA_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node="$NPROC" \
      llava/train/train_mem.py \
      \
      # Enable LoRA fine-tuning (only small adapter weights are trained)
      --lora_enable True \
      \
      # LoRA rank: controls adapter capacity (8 = small, memory-efficient)
      --lora_r 8 \
      \
      # LoRA scaling factor (effective update strength = alpha / r)
      --lora_alpha 16 \
      \
      # Dropout applied inside LoRA adapters (regularization)
      --lora_dropout 0.05 \
      \
      # Base LLaVA model (LLaMA-2 7B + vision alignment)
      --model_name_or_path liuhaotian/llava-v1.5-7b \
      \
      # Vision encoder used to extract image features (CLIP ViT-L/14 @336px)
      --vision_tower openai/clip-vit-large-patch14-336 \
      \
      # Projector that maps CLIP features into LLM embedding space
      # mlp2x_gelu = 2-layer MLP with GELU activations (standard LLaVA choice)
      --mm_projector_type mlp2x_gelu \
      \
      # Which CLIP layer to take features from
      # -2 = second-to-last layer (more semantic, less noisy than final)
      --mm_vision_select_layer -2 \
      \
      # Do NOT wrap images with special <im_start> / <im_end> tokens
      --mm_use_im_start_end False \
      \
      # Do NOT insert per-patch image tokens (use pooled vision features)
      --mm_use_im_patch_token False \
      \
      # Path to ScienceQA / LLaVA JSON training file
      --data_path "$TRAIN_JSON" \
      \
      # Directory containing training images
      --image_folder "$IMG_DIR" \
      \
      # Where checkpoints and logs are written
      --output_dir "$OUTDIR" \
      \
      # Number of full passes over the dataset
      # (ignored if max_steps is set)
      --num_train_epochs 1 \
      \
      # Hard stop after N optimizer steps (used for dry-run)
      --max_steps 5 \
      \
      # Batch size PER GPU (LoRA → usually 1)
      --per_device_train_batch_size 1 \
      \
      # Number of steps to accumulate gradients before optimizer step
      --gradient_accumulation_steps 1 \
      \
      # AdamW learning rate (2e-5 is typical for LoRA)
      --learning_rate 2e-5 \
      \
      # Log training metrics every N steps
      --logging_steps 1 \
      \
      # Save checkpoints based on step count
      --save_strategy steps \
      \
      # Save every N steps
      --save_steps 5 \
      \
      # Keep only the most recent checkpoint
      --save_total_limit 1 \
      \
      # Disable evaluation loop entirely
      --evaluation_strategy no \
      \
      # Precision flags chosen based on GPU (bf16 on A100, fp16 on V100)
      "${PREC_ARGS[@]}" \
      \
      # Disable activation recomputation to simplify DDP graph
      # (enable later if you need memory savings)
      --gradient_checkpointing False \
      \
      # Preprocess samples lazily instead of upfront (saves RAM, faster startup)
      --lazy_preprocess True \
      \
      # Disable multi-worker dataloading (more stable on shared HPC FS)
      --dataloader_num_workers 0 \
      \
      # Required for LoRA + DDP to avoid extra autograd traversal
      --ddp_find_unused_parameters False \
      \
      # Maximum sequence length (text + image tokens)
      --model_max_length 768
  '
