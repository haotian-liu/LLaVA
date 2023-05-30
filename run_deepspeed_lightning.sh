#!/bin/bash

WEIGHT_VERSION=v0

# Pretraining (2 hours)
deepspeed --include=localhost:0,1,2,3 --master_port 26002 \
    llava/train/train_mem.py \
    --deepspeed deepspeed.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v0 \
    --version $WEIGHT_VERSION \
    --data_path /Data/haotian/blip/blip_558k.json \
    --image_folder /Data/haotian/blip/blip_images_558k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_llava-lightning-7b-v0-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
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
  --model_name_or_path ./checkpoints/deepspeed_llava-lightning-7b-v0-pretrain \
  --output ./checkpoints/mm_projector/deepspeed_llava-lightning-7b-v0-pretrain.bin

# Visual instruction tuning (1 hour)
deepspeed --include=localhost:0,1,2,3 --master_port 26002 \
    llava/train/train_mem.py \
    --deepspeed deepspeed.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v0 \
    --version $WEIGHT_VERSION \
    --data_path ./playground/data/llava_instruct/conv_reason_no_overlap_80k.json \
    --image_folder /Data/haotian/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/deepspeed_llava-lightning-7b-v0-pretrain.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_llava-lightning-7b-v0-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
