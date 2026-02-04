#!/usr/bin/env bash
set -euo pipefail

# =========================
# Defaults (override by exporting before calling)
# =========================
: "${LLAVA_ROOT:=/scratch/$USER/LLaVA_DL_project}"
: "${SQA_ROOT:=/scratch/$USER/datasets/ScienceQA/data/scienceqa}"
: "${SIF:=/scratch/$USER/LLaVA_DL_project/llava_dev.sif}"

: "${HF_HOME:=/scratch/$USER/hf_home}"
: "${HF_HUB_CACHE:=/scratch/$USER/hf_cache}"
: "${TRITON_CACHE_DIR:=/scratch/$USER/triton_cache}"

mkdir -p "$LLAVA_ROOT/logs" "$HF_HOME" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR"

# =========================
# CUDA toolkit for DeepSpeed (nvcc) + CUDA_HOME
# =========================
module --force purge || true
module load StdEnv || true
module load CUDA/12.1 2>/dev/null || module load cuda/12.1 2>/dev/null || module load CUDA/12.1.0 2>/dev/null

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
else
  echo "ERROR: nvcc not found after loading CUDA module. DeepSpeed will likely fail."
  exit 2
fi

# =========================
# Info (nice in logs)
# =========================
echo "JobID: ${SLURM_JOB_ID:-na}"
echo "Node:  ${SLURMD_NODENAME:-na}"
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi || true
nvcc -V | head -n 2 || true

# =========================
# Run inside Apptainer
# =========================
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command ...>"
  echo "Example: $0 bash -lc \"echo hello\""
  exit 2
fi

exec apptainer exec --nv \
  -B /scratch/$USER:/scratch/$USER \
  -B /cvmfs:/cvmfs \
  --env LLAVA_ROOT="$LLAVA_ROOT",SQA_ROOT="$SQA_ROOT",HF_HOME="$HF_HOME",HF_HUB_CACHE="$HF_HUB_CACHE",TRITON_CACHE_DIR="$TRITON_CACHE_DIR",CUDA_HOME="$CUDA_HOME" \
  "$SIF" \
  "$@"
