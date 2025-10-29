#!/bin/bash
# ============================================================================
# CM-IBQ LLaVA 训练脚本集合 - 支持完整权重+激活量化
# ============================================================================

# ----------------------------------------------------------------------------
# 1. 单机单卡训练 - Stage 1 (Bottleneck Shaping with Weight Quantization)
# ----------------------------------------------------------------------------
cat > train_stage1_single_gpu.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 2 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_w4a4_mixed" \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage1_single_gpu.sh


# ----------------------------------------------------------------------------
# 2. 单机多卡训练 - Stage 1 (混合精度权重+激活量化)
# ----------------------------------------------------------------------------
cat > train_stage1_multi_gpu.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

NUM_GPUS=4

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 2 \
    --quantize_vision_embeddings \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_w4a4_mixed_4gpu" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage1_multi_gpu.sh


# ----------------------------------------------------------------------------
# 3. 单机多卡训练 - Stage 1 (统一精度量化版本)
# ----------------------------------------------------------------------------
cat > train_stage1_uniform_quant.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

NUM_GPUS=4

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "uniform" \
    --llm_layer_interval 1 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_w4a4_uniform" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage1_uniform_quant.sh


# ----------------------------------------------------------------------------
# 4. DeepSpeed ZeRO-2 训练 - 支持大模型
# ----------------------------------------------------------------------------
cat > train_stage1_deepspeed.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

NUM_GPUS=4

deepspeed --num_gpus=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 3 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_deepspeed" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_deepspeed \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage1_deepspeed.sh


# ----------------------------------------------------------------------------
# 5. Stage 2训练 (Alignment + LoRA) - 基于量化模型
# ----------------------------------------------------------------------------
cat > train_stage2_multi_gpu.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

NUM_GPUS=4

# 使用Stage 1的量化模型
STAGE1_CHECKPOINT="./checkpoints/stage1_w4a4_mixed_4gpu/best_model"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "$STAGE1_CHECKPOINT" \
    --model_size "7b" \
    \
    --stage 2 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 2 \
    --num_groups 8 \
    --use_ib \
    --use_lora \
    --lora_rank 16 \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage2_aligned_lora" \
    --num_epochs 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 300 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage2_multi_gpu.sh


# ----------------------------------------------------------------------------
# 6. 13B模型训练 - 多节点配置
# ----------------------------------------------------------------------------
cat > train_13b_multi_node.sh << 'EOF'
#!/bin/bash

# 节点配置
export MASTER_ADDR="主节点IP"
export MASTER_PORT=29500
export NNODES=2
export NODE_RANK=0
export GPUS_PER_NODE=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-13b" \
    --model_size "13b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 4 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_13b_w4a4" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --bf16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_13b_multi_node.sh


# ----------------------------------------------------------------------------
# 7. W&B监控训练 - 带量化统计
# ----------------------------------------------------------------------------
cat > train_with_wandb.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# W&B配置
export WANDB_PROJECT="cm-ibq-llava"
export WANDB_API_KEY="你的API密钥"

NUM_GPUS=4

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 2 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_wandb" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --use_wandb \
    --run_name "cmibq_w4a4_mixed_exp1" \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_with_wandb.sh


# ----------------------------------------------------------------------------
# 8. 快速测试脚本 - 验证量化功能
# ----------------------------------------------------------------------------
cat > train_quick_test.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

echo "Testing weight + activation quantization..."

python train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 4 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_small.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 512 \
    \
    --output_dir "./checkpoints/quick_test" \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    \
    --use_amp \
    --fp16 \
    \
    --logging_steps 5 \
    --save_epochs 1 \
    --num_workers 2 \
    --seed 42

echo "Quick test completed!"
EOF

chmod +x train_quick_test.sh


# ----------------------------------------------------------------------------
# 9. 完整两阶段训练流程 - W4A4量化
# ----------------------------------------------------------------------------
cat > train_full_pipeline.sh << 'EOF'
#!/bin/bash

set -e  # 遇到错误就退出

echo "=========================================="
echo "CM-IBQ Full Training Pipeline (W4A4)"
echo "=========================================="

# 配置
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# -------------------- Stage 1 --------------------
echo ""
echo "[Stage 1/2] Bottleneck Shaping with Mixed-Precision Quantization..."
echo "  - Weight: 2/4/8-bit mixed precision"
echo "  - Activation: 2/4/8-bit mixed precision"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 2 \
    --num_groups 8 \
    --use_ib \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    --output_dir "./checkpoints/pipeline_stage1" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --use_amp --fp16 --gradient_checkpointing \
    --logging_steps 10 --save_epochs 1 --eval_epochs 1 \
    --num_workers 4 --seed 42

echo ""
echo "Stage 1 completed! Model compressed to ~25% size."
echo ""

# -------------------- Stage 2 --------------------
echo "[Stage 2/2] Task-Aware Optimization with LoRA + Alignment..."
echo "  - Adding LoRA adapters (rank=16)"
echo "  - Visual-text alignment training"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "./checkpoints/pipeline_stage1/best_model" \
    --model_size "7b" \
    --stage 2 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --quantize_weights \
    --weight_quant_mode "mixed" \
    --llm_layer_interval 2 \
    --num_groups 8 \
    --use_ib \
    --use_lora \
    --lora_rank 16 \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    --output_dir "./checkpoints/pipeline_stage2" \
    --num_epochs 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --use_amp --fp16 --gradient_checkpointing \
    --logging_steps 10 --save_epochs 1 --eval_epochs 1 \
    --num_workers 4 --seed 42

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo ""
echo "Final model statistics:"
echo "  - Location: ./checkpoints/pipeline_stage2/best_model"
echo "  - Size: ~3.5GB (from 14GB)"
echo "  - Compression: 4x"
echo "  - Weight bits: 4.0 (mixed 2/4/8)"
echo "  - Activation bits: 4.0 (mixed 2/4/8)"
echo "=========================================="
EOF

chmod +x train_full_pipeline.sh


# ----------------------------------------------------------------------------
# 10. 对比实验脚本 - 不同量化配置
# ----------------------------------------------------------------------------
cat > train_ablation_study.sh << 'EOF'
#!/bin/bash

echo "Running ablation study on quantization configurations..."

CONFIGS=(
    "uniform_w4a4:uniform:4:4"
    "uniform_w8a4:uniform:8:4"
    "mixed_w4a4:mixed:4:4"
    "mixed_w4a8:mixed:4:8"
)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r name mode w_bits a_bits <<< "$config"
    
    echo ""
    echo "Training config: $name"
    echo "  Mode: $mode, Weight: ${w_bits}bit, Act: ${a_bits}bit"
    
    python train_cmibq.py \
        --model_path "liuhaotian/llava-v1.5-7b" \
        --model_size "7b" \
        --stage 1 \
        --target_bits_act $a_bits \
        --target_bits_weight $w_bits \
        --quantize_weights \
        --weight_quant_mode $mode \
        --llm_layer_interval 2 \
        --num_groups 8 \
        --train_data_path "./data/llava_instruct_small.json" \
        --image_folder "./data/coco/train2017" \
        --output_dir "./checkpoints/ablation_$name" \
        --num_epochs 1 \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --use_amp --fp16 \
        --logging_steps 10 \
        --seed 42
    
    echo "Config $name completed!"
done

echo ""
echo "Ablation study completed! Check ./checkpoints/ablation_* for results."
EOF

chmod +x train_ablation_study.sh


echo "=========================================="
echo "Training scripts created successfully!"
echo "=========================================="
echo ""
echo "Available scripts:"
echo "  1. train_stage1_single_gpu.sh   - 单GPU Stage 1训练 (W4A4混合精度)"
echo "  2. train_stage1_multi_gpu.sh    - 多GPU Stage 1训练 (W4A4混合精度)"
echo "  3. train_stage1_uniform_quant.sh- 统一精度量化训练"
echo "  4. train_stage1_deepspeed.sh    - DeepSpeed ZeRO训练"
echo "  5. train_stage2_multi_gpu.sh    - Stage 2 LoRA+对齐训练"
echo "  6. train_13b_multi_node.sh      - 13B模型多节点训练"
echo "  7. train_with_wandb.sh          - W&B监控训练"
echo "  8. train_quick_test.sh          - 快速功能测试"
echo "  9. train_full_pipeline.sh       - 完整两阶段流程"
echo " 10. train_ablation_study.sh      - 量化配置对比实验"
echo ""
echo "Quick start:"
echo "  ./train_quick_test.sh           # 测试环境"
echo "  ./train_full_pipeline.sh        # 运行完整训练"
echo ""
