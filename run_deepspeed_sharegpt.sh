#!/bin/bash

PROMPT_VERSION=v1

MODEL="30b"
PORT=26000
GPUS="0,1,2,3,4,5,6,7"

deepspeed --include=localhost:$GPUS --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed deepspeed.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/llama_hf/llama_$MODEL \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/sg_90k_clean_new_splitlong.json \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_llama-$MODEL-$PROMPT_VERSION-sg90k_clean_splitlong-finetune_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
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
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb
