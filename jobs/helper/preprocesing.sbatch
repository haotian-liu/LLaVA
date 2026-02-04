#!/bin/bash
set -e  # exit immediately if a command fails
set -u  # error on undefined variables
set -o pipefail

PYTHON="./.venv/bin/python"
SCRIPT="scripts/convert_sqa_to_llava.py"
BASE_DIR="/scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa"

# Prompt formats you want to generate
PROMPT_FORMATS=(
  "QCM-A"
  "QCM-LEA"
  "QCM-ALE"
  "QCM-AE"
  "QCM-EA"
)

# Dataset splits
SPLITS=(
  "train"
  "val"
  "minival"
  "test"
  "minitest"
)

echo "Starting ScienceQA → LLaVA conversion"
echo "Base dir: ${BASE_DIR}"
echo

for FORMAT in "${PROMPT_FORMATS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    echo "======================================"
    echo "Format: ${FORMAT}"
    echo "Split : ${SPLIT}"
    echo "======================================"

    ${PYTHON} ${SCRIPT} \
      convert_to_llava \
      --base-dir "${BASE_DIR}" \
      --prompt-format "${FORMAT}" \
      --split "${SPLIT}"

    echo
  done
done

echo "All conversions completed successfully."
