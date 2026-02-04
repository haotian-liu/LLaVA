#!/usr/bin/env bash
set -euo pipefail

export SQA_ROOT="/scratch/$USER/datasets/ScienceQA/data/scienceqa"
export LLAVA_ROOT="/scratch/$USER/LLaVA_DL_project"
export HF_HOME="/scratch/$USER/hf_home"
export HF_HUB_CACHE="/scratch/$USER/hf_cache"
export TRANSFORMERS_CACHE="/scratch/$USER/hf_cache"

apptainer exec --nv \
  -B /scratch/$USER:/scratch/$USER \
  -B /cvmfs:/cvmfs \
  --env SQA_ROOT="$SQA_ROOT",LLAVA_ROOT="$LLAVA_ROOT",HF_HOME="$HF_HOME",HF_HUB_CACHE="$HF_HUB_CACHE",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  "$@"
