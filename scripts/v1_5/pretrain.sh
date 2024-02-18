#!/bin/bash

deepspeed --hostfile hostfile.txt \
    --master_port 65535 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt2/yinxie/pretrain_models/QWen/Qwen1.5-72B-Chat \
    --version plain \
    --data_path /mnt2/jiaxingchen/project/LLaVA1.5/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt2/jiaxingchen/project/LLaVA1.5/playground/data/LLaVA-Pretrain/images \
    --vision_tower /mnt2/jiaxingchen/project/LLaVA1.5/checkpoints/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-qwen-72b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --lazy_preprocess True 2>&1 | tee qwen_pretrain_log
