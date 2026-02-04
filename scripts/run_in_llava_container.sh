#!/usr/bin/env bash
set -euo pipefail

# ---- Defaults (override by exporting before calling) ----
: "${LLAVA_ROOT:=/scratch/$USER/LLaVA_DL_project}"
: "${SQA_ROOT:=/scratch/$USER/datasets/ScienceQA/data/scienceqa}"
: "${SIF:=/scratch/$USER/LLaVA_DL_project/llava_dev.sif}"

# Caches on scratch
: "${HF_HOME:=/scratch/$USER/hf_home}"
: "${HF_HUB_CACHE:=/scratch/$USER/hf_cache}"
: "${TRITON_CACHE_DIR:=/scratch/$USER/triton_cache}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR"

# Optional: CUDA toolkit path for DeepSpeed checks (nvcc)
# If CUDA_HOME is set by the caller, we pass it through.
ENV_VARS="LLAVA_ROOT=$LLAVA_ROOT,SQA_ROOT=$SQA_ROOT,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

if [[ -n "${CUDA_HOME:-}" ]]; then
  ENV_VARS="$ENV_VARS,CUDA_HOME=$CUDA_HOME"
fi

# Safety: require a command to execute
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command ...>"
  echo "Example: $0 bash -lc 'echo hello'"
  exit 2
fi

exec apptainer exec --nv \
  -B /scratch/$USER:/scratch/$USER \
  -B /cvmfs:/cvmfs \
  --env "$ENV_VARS" \
  "$SIF" \
  "$@"
