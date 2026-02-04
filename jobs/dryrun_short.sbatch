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

/scratch/$USER/LLaVA_DL_project/scripts/llava_job.sh \
  bash -lc "/scratch/$USER/LLaVA_DL_project/jobs/cmds/dryrun_train.sh"
