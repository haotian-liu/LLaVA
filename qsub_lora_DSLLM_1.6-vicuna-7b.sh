#!/bin/bash

### The following requests all resources on 1 DGX-1 node
#PBS -l select=1:ngpus=2:ncpus=16

### Specify amount of time required
#PBS -l walltime=48:00:00

### Specify project code
#PBS -P 12002486

### Specify name for job
#PBS -N lora_1.6-vicuna-7b

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### For automatic mailing, use the following options:
#PBS -M chih0001@e.ntu.edu.sg
#PBS -m abe

### Start of commands to be run
# 获取当前日期和时间，格式为 YYYYMMDD-HHMMSS
NOW=$(date +%Y%m%d-%H%M%S)

# 将标准输出和错误重定向到包含日期时间的文件名
exec > /home/users/ntu/chih0001/scratch/VLM/LLaVA/train_logs/${PBS_JOBNAME}_${NOW}.o${PBS_JOBID} 2>&1

source /home/users/ntu/chih0001/anaconda3/etc/profile.d/conda.sh
conda activate llava-test

export CUDA_VISIBLE_DEVICES=0,1

cd /home/users/ntu/chih0001/scratch/VLM/LLaVA

module load craype-accel-nvidia80

# Run deepspeed training command directly
# Always keep the global batch size the same: per_device_train_batch_size x gradient_accumulation_steps x num_gpus.
# For LoRA is per_device_train_batch_size x gradient_accumulation_steps x num_gpus = 128.

# GlobalBatchsize = 128 = per_device_train_batch_size x gradient_accumulation_steps x num_gpus
# acc step: 梯度累积，应该影响比较小。所以考虑到显存可能直接调这个是比较稳妥的方式。
# 2GPUs: 2 x 64 = 2 x 64 x 1 = 2 x 32 x 2 = 2 x 16 x 4
# 3GPUs: 3 x 44 = 3 x 44 x 1 = 3 x 22 x 2 = 132 （128不能整除3，所以采用逼近的方式，用132）
# 4GPUs: 4 x 32 = 4 x 32 x 1 = 4 x 16 x 2
# 2x32x2
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/users/ntu/chih0001/scratch/model/llava-v1.6-vicuna-7b \
    --version v1 \
    --data_path /home/users/ntu/chih0001/scratch/VLM/LLaVA/train/lora.json \
    --image_folder ./playground/data \
    --vision_tower /home/users/ntu/chih0001/scratch/model/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/users/ntu/chih0001/scratch/model/lora/globalBS/llava-v1.6-vicuna-7b-DSLLM-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

#NoGlobalBS
# deepspeed llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /home/users/ntu/chih0001/scratch/model/llava-v1.6-vicuna-7b \
#     --version v1 \
#     --data_path /home/users/ntu/chih0001/scratch/VLM/LLaVA/train/lora.json \
#     --image_folder ./playground/data \
#     --vision_tower /home/users/ntu/chih0001/scratch/model/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir /home/users/ntu/chih0001/scratch/model/lora/globalBS/llava-v1.6-vicuna-7b-DSLLM-lora \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb