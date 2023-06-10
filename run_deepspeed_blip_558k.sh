#!/bin/bash

WEIGHT_VERSION=v1-1
PROMPT_VERSION=v1

# WEIGHT_VERSION=v0
# PROMPT_VERSION=v0

PORT=26002

GPUS="0,1,2,3,4,5,6,7"

# # Pretraining
# deepspeed --include=localhost:$GPUS --master_port $PORT \
#     llava/train/train_mem.py \
#     --deepspeed deepspeed.json \
#     --model_name_or_path ./checkpoints/vicuna-13b-$WEIGHT_VERSION \
#     --version $WEIGHT_VERSION \
#     --data_path /Data/haotian/blip_6m/blip_558k.json \
#     --image_folder /Data/haotian/blip_6m/images \
#     --vision_tower openai/clip-vit-large-patch14 \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end \
#     --bf16 True \
#     --output_dir ./checkpoints/deepspeed_llava-13b-$WEIGHT_VERSION-pretrain_blip558k \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-3 \
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

# # Extract projector features
# python scripts/extract_mm_projector.py \
#   --model_name_or_path ./checkpoints/deepspeed_llava-13b-$WEIGHT_VERSION-pretrain_blip558k \
#   --output ./checkpoints/mm_projector/deepspeed_llava-13b-$WEIGHT_VERSION-pretrain_blip558k.bin

# Visual instruction tuning
deepspeed --include=localhost:$GPUS --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed deepspeed.json \
    --model_name_or_path ./checkpoints/vicuna-13b-$WEIGHT_VERSION \
    --version $WEIGHT_VERSION \
    --data_path ./playground/data/coco2014_gpt4_reasonv3_58k_detail23k_complex77k_curate_158k.json \
    --image_folder /Data/haotian/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/deepspeed_llava-13b-$WEIGHT_VERSION-pretrain_blip558k.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_llava-13b-$WEIGHT_VERSION-pretrain_blip558k-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
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









# # Visual instruction tuning (lora)
# deepspeed --include=localhost:$GPUS --master_port $PORT \
#     llava/train/train_mem.py \
#     --deepspeed deepspeed.json \
#     --lora_enable True \
#     --model_name_or_path ./checkpoints/vicuna-13b-$WEIGHT_VERSION \
#     --version $PROMPT_VERSION \
#     --data_path ./playground/data/coco2014_gpt4_reasonv3_58k_detail23k_complex77k_curate_158k.json \
#     --image_folder /Data/haotian/coco/train2017 \
#     --vision_tower openai/clip-vit-large-patch14 \
#     --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/deepspeed_llava-13b-$WEIGHT_VERSION-pretrain_blip558k.bin \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end True \
#     --bf16 True \
#     --output_dir ./checkpoints/deepspeed_llava-13b-$WEIGHT_VERSION-pretrain_blip558k-finetune_lora \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 5e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --dataloader_num_workers 4 \
#     --report_to wandb
