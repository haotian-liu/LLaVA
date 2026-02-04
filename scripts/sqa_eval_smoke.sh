#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY"
  echo "Tip: when running apptainer, bind /scratch and /cvmfs:"
  echo "  apptainer exec --nv -B /scratch/$USER:/scratch/$USER -B /cvmfs:/cvmfs llava_dev.sif bash scripts/sqa_eval_smoke.sh"
  exit 1
fi

export HF_HOME=/scratch/$USER/hf_home
export HF_HUB_CACHE=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"  "$TRANSFORMERS_CACHE"


CHUNKS=1
IDX=0

CUDA_VISIBLE_DEVICES=0 "$PY" -m llava.eval.model_vqa_science \
  --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
  --answers-file ./smoke_test_llava-13b.jsonl \
  --question-file /scratch/$USER/datasets/ScienceQA/data/scienceqa/llava_minitest_QCM-LEA.json \
  --image-folder /scratch/$USER/datasets/ScienceQA/data/scienceqa/images/test \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX \
  --conv-mode llava_v1
