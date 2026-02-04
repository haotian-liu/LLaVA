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

# =========================
# Paths
# =========================
export LLAVA_ROOT="/scratch/$USER/LLaVA_DL_project"
export SQA_ROOT="/scratch/$USER/datasets/ScienceQA/data/scienceqa"
export SIF="/scratch/$USER/LLaVA_DL_project/llava_dev.sif"

mkdir -p "$LLAVA_ROOT/logs"

# =========================
# Hugging Face + Triton caches (scratch)
# =========================
export HF_HOME="/scratch/$USER/hf_home"
export HF_HUB_CACHE="/scratch/$USER/hf_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR"

# =========================
# Load CUDA toolkit (for nvcc) so DeepSpeed can detect CUDA_HOME
# =========================
module --force purge
module load StdEnv
module load CUDA/12.1 2>/dev/null || module load cuda/12.1 2>/dev/null || module load CUDA/12.1.0 2>/dev/null

# Derive CUDA_HOME from nvcc
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
echo "CUDA_HOME=$CUDA_HOME"
ls -l "$CUDA_HOME/bin/nvcc"
nvcc -V | head -n 2 || true

# =========================
# Debug
# =========================
echo "JobID: $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
nvidia-smi || true

# =========================
# Run inside Apptainer
# =========================
# ---- Run inside Apptainer (robust: no $(...) command substitution) ----
apptainer exec --nv \
  -B /scratch/$USER:/scratch/$USER \
  -B /cvmfs:/cvmfs \
  --env HF_HOME="$HF_HOME",HF_HUB_CACHE="$HF_HUB_CACHE",TRITON_CACHE_DIR="$TRITON_CACHE_DIR",CUDA_HOME="$CUDA_HOME",LLAVA_ROOT="$LLAVA_ROOT",SQA_ROOT="$SQA_ROOT" \
  "$SIF" bash -s <<'EOF'
set -euo pipefail
cd "$LLAVA_ROOT"

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PY="$LLAVA_ROOT/.venv/bin/python"
DS="$LLAVA_ROOT/.venv/bin/deepspeed"

echo "Using python: $PY"
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
$PY -c 'import deepspeed; print("deepspeed import ok")'

$PY $LLAVA_ROOT/llava/train/train_mem.py --help | grep -i lora


TRAIN_JSON="$SQA_ROOT/llava_train_QCM-LEA.json"
IMG_DIR="$SQA_ROOT/images/train"
OUTDIR="/scratch/$USER/llava_dryrun_out"
mkdir -p "$OUTDIR"

test -f "$TRAIN_JSON" || { echo "Missing: $TRAIN_JSON"; exit 2; }
test -d "$IMG_DIR"     || { echo "Missing: $IMG_DIR"; exit 2; }
test -f "./scripts/zero3.json" || { echo "Missing: ./scripts/zero3.json"; exit 2; }

$PY -c "import deepspeed; print(deepspeed.__version__)"
$PY -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'


"$DS" llava/train/train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --lora_enable True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --model_name_or_path liuhaotian/llava-v1.5-7b \
  --data_path "$TRAIN_JSON" \
  --image_folder "$IMG_DIR" \
  --output_dir "$OUTDIR" \
  --num_train_epochs 1 \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 5 \
  --save_total_limit 1 \
  --evaluation_strategy no \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --dataloader_num_workers 0

EOF