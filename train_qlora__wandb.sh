#!/bin/bash

set -e -x  # stop on 1st error, debug output of args used

# TODO:
# - adjust paths like '/root/' to match your system!
# - get/create the data
# - download the base model:
#     git lfs install
#     git clone https://huggingface.co/liuhaotian/llava-v1.6-7b
# - review args below, and the comments section

# Set the prompt and model versions directly in the command
deepspeed /root/LLaVA/llava/train/train_mem.py \
    --deepspeed /root/LLaVA/scripts/zero2.json \ 
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --bits 4 \
    --model_name_or_path /root/LLaVA/llava/llava-v1.6-7b \
    --version llava_llama_2 \
    --data_path /root/dataset/train/dataset.json \
    --validation_data_path /root/dataset/validation/dataset.json \
    --image_folder /root/dataset/images/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /root/LLaVA/llava/checkpoints/llama-2-7b-chat-task-qlora \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy “epoch” \ 
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
    --lazy_preprocess True

# Notes on args:
#
# zero2.json -- if not enough GPU memory, try zero3.json which offloads to CPU (slower)
# bits 4 -- qlora
#
# model_name_or_path - original had /root/LLaVA/llava/llava-v1.5-7b
#
# num_train_epochs - was 500, but blog says 10 is enough (depends on data...)
# mm_projector_lr: The separate learning rate for the multimodal projector as specified by the LLaVA authors 
# bits: This is where we specify we want to use Q-LoRA 
# lora_alpha: Following the guidelines of the LLaVA authors, we've set lora_alpha to 256. This alpha value is pivotal in preserving numerical stability and the full expressive power of the model. It's worth noting that this is an adjustment from the typical values around 16
# lora_r: The lora_r parameter represents the rank of the decomposition matrices in LoRA. We've chosen a value of 128, diverging from the common range of 8 to 64 seen in typical LLM fine-tunes. A higher rank, as in our case, can enhance the model's representational capability
# mm_projector_type: I set this to mlp2x_gelu, which is a multi-layer perceptron with GELU activation
# deepspeed: Here we specify the deepspeed zero stage 2 config for the training run
# data_path: This parameter specifies the location of the training dataset that we created earlier
# validation_data_path: Since I added intermediate evaluations between each epoch, we will need to pass the path to our validation dataset as well (note that the code assumes the images for both train and validation are in the same directory)
# image_folder: This argument points to the directory containing the images used in both the training and validation datasets.
# output_dir: This is the directory where the trained model checkpoints will be saved. It’s important to have sufficient storage space in this directory, especially when training large models like LLaVA
#
# Depending on your hardware setup, you can change the batch size to avoid memory errors. I trained on 8 NVIDIA RTX 3090’s, which had no issues with a batch size of 32. The training script has an option for monitoring using Weights & Biases using the --report_to wandb flag, providing real-time tracking of the model's progress and performance metrics.
#
# see also https://wandb.ai/byyoung3/ml-news/reports/How-to-Fine-Tune-LLaVA-on-a-Custom-Dataset--Vmlldzo2NjUwNTc1

# To infer with the QLORA layer:
#
# python run_llava.py --model-path /root/LLaVA/llava/checkpoints/llava-2-7b-chat-task-qlora/best_llava_eval_model_llava_lora 
# --model-base /root/LLaVA/llava/llava-v1.6-7b 
# --image-file /root/dataset/images/0f47c0b5-2c77-45e6-87b0-89af46e99500.jpg 
# --query “why was this photo taken?”
