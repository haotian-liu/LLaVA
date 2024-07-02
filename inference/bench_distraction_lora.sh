#!/bin/bash

### The following requests all resources on 1 DGX-1 node
#PBS -l select=1:ngpus=2:ncpus=16

### Specify amount of time required
#PBS -l walltime=20:00:00

### Specify project code
#PBS -P 12002486

### Specify name for job
#PBS -N llava_distraction_lora

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### For automatic mailing, use the following options:
#PBS -M chih0001@e.ntu.edu.sg
#PBS -m abe

### Start of commands to be run
source /home/users/ntu/chih0001/anaconda3/etc/profile.d/conda.sh
conda activate llava-test

export CUDA_VISIBLE_DEVICES=0,1

cd /home/users/ntu/chih0001/scratch/VLM/sglang/benchmark/llava_bench

python -m llava.eval.model_vqa \
    --model-path /home/users/ntu/chih0001/scratch/model/lora/llava-v1.5-7b-DSLLM-lora \
    --model-base /home/users/ntu/chih0001/scratch/model/llava-v1.5-7b \
    --question-file /home/users/ntu/chih0001/scratch/VLM/sglang/benchmark/llava_bench/questions/distraction/questions_ultrasimplified_new.jsonl \
    --image-folder /home/users/ntu/chih0001/scratch/data/distraction \
    --answers-file /home/users/ntu/chih0001/scratch/VLM/sglang/benchmark/llava_bench/answers/SAM-DD/lora_answers_SAM-DD_1.5-7b_new.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
