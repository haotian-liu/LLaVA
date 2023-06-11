#!/bin/bash

WEIGHT_VERSION=v1-1
PROMPT_VERSION=v1
MODEL_VERSION="7b"

# Pretraining
deepspeed llava/train/train_mem.py \
    --deepspeed /path/to/deepspeed.json \
    --model_name_or_path ./checkpoints/vicuna-$MODEL_VERSION-$WEIGHT_VERSION \
    --version $WEIGHT_VERSION \
    --data_path /path/to/anno.json \
    --image_folder /path/to/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_llava-$MODEL_VERSION-$WEIGHT_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
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

# Extract projector features
python scripts/extract_mm_projector.py \
  --model_name_or_path ./checkpoints/deepspeed_llava-$MODEL_VERSION-$WEIGHT_VERSION-pretrain \
  --output ./checkpoints/mm_projector/deepspeed_llava-$MODEL_VERSION-$WEIGHT_VERSION-pretrain.bin