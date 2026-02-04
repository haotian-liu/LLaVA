#!/usr/bin/env bash
#SBATCH --job-name=AE-llava-eval
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/llava/LLaVA_DL_project/logs/%x-%j.out
#SBATCH --error=/scratch/%u/llava/LLaVA_DL_project/logs/%x-%j.err

set -euo pipefail

export WORK_ROOT="/scratch/$USER/llava"
export LLAVA_ROOT="$WORK_ROOT/LLaVA_DL_project"
export SCRATCH_MERGED_ROOT="$WORK_ROOT/merged_models"
export SIF="$LLAVA_ROOT/llava_dev.sif"

export HF_HOME="$WORK_ROOT/hf_home"
export HF_HUB_CACHE="$WORK_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$WORK_ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$SCRATCH_MERGED_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$LLAVA_ROOT/logs"

apptainer exec --nv \
  -B "$WORK_ROOT":"$WORK_ROOT" \
  -B /cvmfs:/cvmfs \
  --env WORK_ROOT="$WORK_ROOT" \
  --env LLAVA_ROOT="$LLAVA_ROOT" \
  --env SCRATCH_MERGED_ROOT="$SCRATCH_MERGED_ROOT" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  "$SIF" bash -lc '
    set -euo pipefail
    cd "$LLAVA_ROOT"
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"

    echo "SCRATCH_MERGED_ROOT=$SCRATCH_MERGED_ROOT"
    nvidia-smi -L || true

    echo "=== PRECHECK ==="
    ls -lah "$LLAVA_ROOT/.venv/bin/python"
    "$LLAVA_ROOT/.venv/bin/python" -c "import llava; print(\"llava import ok\")"
    mkdir -p "$SCRATCH_MERGED_ROOT/__write_test"
    dd if=/dev/zero of="$SCRATCH_MERGED_ROOT/__write_test/test.bin" bs=1M count=64 status=none
    rm -rf "$SCRATCH_MERGED_ROOT/__write_test"

    bash ./llava/eval/run_merge_eval.sh AE
  '
