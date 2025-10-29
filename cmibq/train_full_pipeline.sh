#!/bin/bash
#SBATCH --job-name=cmibq-pipeline
#SBATCH --time=96:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --environment=llava-pytorch25
#SBATCH --account=lp12
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

set -e  # 遇到错误就退出

echo "=========================================="
echo "CM-IBQ Full Training Pipeline (W4A4)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo ""

# 环境配置
export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/huggingface
export TRANSFORMERS_VERBOSITY=info
export OMP_NUM_THREADS=72  # 288/4 = 72 per GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# 切换到工作目录
cd $SCRATCH/cmibq-training
mkdir -p logs

# 激活虚拟环境
source ./llava-venv/bin/activate

# 打印环境信息
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count per node: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# 获取主节点地址
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Total GPUs: 8 (2 nodes × 4 GPUs)"
echo ""

# ==================== Stage 1 ====================
echo "=========================================="
echo "[Stage 1/2] Bottleneck Shaping"
echo "=========================================="
echo "  - Nodes: 2 (8 GPUs total)"
echo "  - Weight: 2/4/8-bit mixed precision"
echo "  - Activation: 2/4/8-bit mixed precision"
echo "  - Effective batch size: 256 (2×8×16)"
echo ""

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
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
    --train_data_path "$SCRATCH/data/llava_instruct_150k.json" \
    --eval_data_path "$SCRATCH/data/llava_instruct_eval.json" \
    --image_folder "$SCRATCH/data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "$SCRATCH/checkpoints/pipeline_stage1" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
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
    --num_workers 8 \
    --seed 42

STAGE1_EXIT_CODE=$?
if [ $STAGE1_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Stage 1 failed with exit code $STAGE1_EXIT_CODE"
    echo "Check logs/pipeline_${SLURM_JOB_ID}.err for details"
    exit $STAGE1_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Stage 1 completed successfully!"
echo "=========================================="
echo "Model saved to: $SCRATCH/checkpoints/pipeline_stage1/best_model"
echo "Model compressed to ~25% of original size"
echo "Completion time: $(date)"
echo ""
sleep 5

# ==================== Stage 2 ====================
echo "=========================================="
echo "[Stage 2/2] Task-Aware Optimization"
echo "=========================================="
echo "  - Loading Stage 1 quantized model"
echo "  - Adding LoRA adapters (rank=16)"
echo "  - Visual-text alignment training"
echo "  - Effective batch size: 256 (2×8×16)"
echo ""

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_cmibq.py \
    --model_path "$SCRATCH/checkpoints/pipeline_stage1/best_model" \
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
    --train_data_path "$SCRATCH/data/llava_instruct_150k.json" \
    --eval_data_path "$SCRATCH/data/llava_instruct_eval.json" \
    --image_folder "$SCRATCH/data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "$SCRATCH/checkpoints/pipeline_stage2" \
    --num_epochs 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
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
    --num_workers 8 \
    --seed 42

STAGE2_EXIT_CODE=$?
if [ $STAGE2_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Stage 2 failed with exit code $STAGE2_EXIT_CODE"
    echo "Check logs/pipeline_${SLURM_JOB_ID}.err for details"
    exit $STAGE2_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "🎉 Pipeline completed successfully!"
echo "=========================================="
echo "Completion time: $(date)"
echo ""
echo "📊 Final model statistics:"
echo "  - Location: $SCRATCH/checkpoints/pipeline_stage2/best_model"
echo "  - Size: ~3.5GB (compressed from ~14GB)"
echo "  - Compression ratio: 4x"
echo "  - Weight quantization: 4.0-bit (mixed 2/4/8)"
echo "  - Activation quantization: 4.0-bit (mixed 2/4/8)"
echo ""
echo "📁 Checkpoint locations:"
echo "  - Stage 1: $SCRATCH/checkpoints/pipeline_stage1/"
echo "  - Stage 2: $SCRATCH/checkpoints/pipeline_stage2/"
echo ""
echo "📝 Log files:"
echo "  - Output: logs/pipeline_${SLURM_JOB_ID}.out"
echo "  - Error: logs/pipeline_${SLURM_JOB_ID}.err"
echo ""
echo "🔍 To evaluate the model, run:"
echo "  python evaluate_cmibq.py --model_path $SCRATCH/checkpoints/pipeline_stage2/best_model"
echo "=========================================="