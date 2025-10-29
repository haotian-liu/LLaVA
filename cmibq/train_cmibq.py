#!/usr/bin/env python
"""
CM-IBQ LLaVA训练脚本 - 增强版
支持两阶段训练，添加了内存优化和更好的错误处理
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from pathlib import Path
import warnings
import gc

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cmibq import CMIBQQuantizedLLaVA
from cmibq.training import DistributedCMIBQTrainer
from cmibq.training.data_utils import make_supervised_data_module


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDNN的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_gpu_memory():
    """检查GPU内存状态"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            total_memory = props.total_memory / 1024**3
            
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {total_memory:.1f} GB")
            print(f"  Allocated: {memory_allocated:.1f} GB")
            print(f"  Reserved: {memory_reserved:.1f} GB")
            print(f"  Free: {total_memory - memory_reserved:.1f} GB")


def optimize_model_for_size(model, model_size: str):
    """
    根据模型大小应用优化策略
    
    Args:
        model: 模型实例
        model_size: 模型大小标识符 ('7b', '13b', '34b', '70b')
    """
    print(f"\nApplying optimizations for {model_size} model...")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # 根据模型大小应用不同的优化策略
    if model_size in ['13b', '34b', '70b']:
        print("Applying memory optimizations for large model...")
        
        # 1. 启用梯度检查点（gradient checkpointing）
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled for main model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled for base model")
        
        # 2. 对vision tower单独启用梯度检查点
        try:
            if hasattr(model, 'base_model'):
                vision_tower = model.base_model.get_vision_tower()
                if vision_tower and hasattr(vision_tower, 'gradient_checkpointing_enable'):
                    vision_tower.gradient_checkpointing_enable()
                    print("✓ Gradient checkpointing enabled for vision tower")
        except Exception as e:
            warnings.warn(f"Could not enable gradient checkpointing for vision tower: {e}")
        
        # 3. CPU offloading建议（需要手动在DeepSpeed配置中启用）
        if model_size in ['34b', '70b']:
            print("✓ Recommended: Enable CPU offloading in DeepSpeed config")
            print("  - Use ZeRO Stage 3 with offload_optimizer and offload_param")
            
        # 4. 混合精度建议
        print("✓ Recommended: Use bf16 instead of fp16 for better stability")
        
        # 5. 批量大小建议
        recommended_batch_size = {
            '13b': 2,
            '34b': 1,
            '70b': 1
        }
        print(f"✓ Recommended batch size: {recommended_batch_size.get(model_size, 1)}")
        
    # 对于7B模型的优化
    elif model_size == '7b':
        print("Applying standard optimizations for 7B model...")
        # 7B模型通常不需要激进的内存优化
        print("✓ Standard optimization applied")
        print("✓ Recommended: Use fp16 for faster training")
    
    # 清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ GPU cache cleared")
    
    return model


def validate_args(args):
    """验证和调整参数"""
    # 根据模型大小自动调整批量大小
    if args.model_size in ['34b', '70b'] and args.batch_size > 1:
        warnings.warn(f"Batch size {args.batch_size} might be too large for {args.model_size} model. "
                     f"Consider reducing to 1.")
    
    # 检查梯度累积
    if args.model_size in ['13b', '34b', '70b']:
        min_grad_accum = {'13b': 8, '34b': 16, '70b': 32}
        if args.gradient_accumulation_steps < min_grad_accum.get(args.model_size, 8):
            warnings.warn(f"Consider increasing gradient_accumulation_steps to at least "
                         f"{min_grad_accum[args.model_size]} for {args.model_size} model")
    
    # 检查学习率
    if args.stage == 2 and args.learning_rate > 1e-5:
        warnings.warn("Stage 2 typically requires lower learning rate. Consider using 1e-5 or smaller.")
    
    # 检查数据路径
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found: {args.train_data_path}")
    
    if not os.path.exists(args.image_folder):
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")
    
    return args


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CM-IBQ LLaVA Training')
    parser.add_argument('--quantize_weights', action='store_true', default=True,
                       help='是否量化权重')
    parser.add_argument('--weight_quant_mode', type=str, default='mixed',
                       choices=['uniform', 'mixed'],
                       help='权重量化模式：uniform(统一精度)或mixed(混合精度)')
    parser.add_argument('--llm_layer_interval', type=int, default=2,
                       help='LLM层量化间隔（每隔几层量化一次）')
    parser.add_argument('--quantize_vision_embeddings', action='store_true', default=False,
                       help='是否量化vision embeddings')
    parser.add_argument('--quantize_lm_head', action='store_true', default=False,
                       help='是否量化语言模型输出头')
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='LLaVA模型路径')
    parser.add_argument('--model_base', type=str, default=None,
                       help='基础模型路径（如果使用LoRA）')
    parser.add_argument('--model_size', type=str, default='7b',
                       choices=['7b', '13b', '34b', '70b'],
                       help='模型大小')
    
    # 量化参数
    parser.add_argument('--target_bits_act', type=float, default=4.0,
                       help='激活量化目标比特')
    parser.add_argument('--target_bits_weight', type=float, default=4.0,
                       help='权重量化目标比特')
    parser.add_argument('--num_groups', type=int, default=8,
                       help='量化分组数')
    parser.add_argument('--use_ib', action='store_true', default=True,
                       help='是否使用IB框架')
    parser.add_argument('--use_lora', action='store_true', default=False,
                       help='是否使用LoRA (Stage 2)')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA秩')
    
    # 训练阶段
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                       help='训练阶段: 1=Bottleneck Shaping, 2=Alignment')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='训练数据JSON路径')
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='验证数据JSON路径')
    parser.add_argument('--image_folder', type=str, required=True,
                       help='图像文件夹路径')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='每GPU的batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                       help='验证batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='warmup步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='使用混合精度训练')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='使用FP16')
    parser.add_argument('--bf16', action='store_true', default=False,
                       help='使用BF16')
    parser.add_argument('--use_deepspeed', action='store_true',
                       help='使用DeepSpeed')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='使用梯度检查点')
    parser.add_argument('--auto_optimize', action='store_true', default=True,
                       help='根据模型大小自动优化')
    
    # 日志和保存
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='日志记录步数')
    parser.add_argument('--save_epochs', type=int, default=1,
                       help='保存检查点的epoch间隔')
    parser.add_argument('--eval_epochs', type=int, default=1,
                       help='验证的epoch间隔')
    parser.add_argument('--use_wandb', action='store_true',
                       help='使用W&B记录')
    parser.add_argument('--run_name', type=str, default=None,
                       help='运行名称')
    
    # 其他
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载worker数')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='从检查点恢复')
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 验证参数
    args = validate_args(args)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查GPU状态
    print("\n" + "=" * 80)
    print("GPU Status:")
    print("=" * 80)
    check_gpu_memory()
    
    # 打印配置
    print("\n" + "=" * 80)
    print(f"CM-IBQ LLaVA Training - Stage {args.stage}")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Model size: {args.model_size}")
    print(f"Activation quantization: {args.target_bits_act} bits")
    print(f"Weight quantization: {args.target_bits_weight} bits ({args.weight_quant_mode} mode)")
    print(f"  - Quantize weights: {args.quantize_weights}")
    if args.quantize_weights:
        print(f"  - LLM layer interval: every {args.llm_layer_interval} layers")
        print(f"  - Quantize vision embeddings: {args.quantize_vision_embeddings}")
        print(f"  - Quantize LM head: {args.quantize_lm_head}")
    print(f"Use IB: {args.use_ib}")
    print(f"Use LoRA: {args.use_lora and args.stage == 2}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # 1. 加载模型
    print("\n[1/4] Loading model...")
    try:
        model = CMIBQQuantizedLLaVA(
            model_path=args.model_path,
            # 激活量化参数
            target_bits_act=args.target_bits_act,
            # 权重量化参数（新增）
            target_bits_weight=args.target_bits_weight,
            quantize_weights=args.quantize_weights,
            weight_quant_mode=args.weight_quant_mode,
            llm_layer_interval=args.llm_layer_interval,
            quantize_vision_embeddings=args.quantize_vision_embeddings,
            quantize_lm_head=args.quantize_lm_head,
            # 其他参数
            use_ib=args.use_ib,
            use_lora=args.use_lora and args.stage == 2,
            lora_rank=args.lora_rank,
            num_groups=args.num_groups,
            stage=args.stage,
            model_base=args.model_base
        )
        
        print(f"Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # 2. 应用内存优化
    if args.auto_optimize:
        model = optimize_model_for_size(model, args.model_size)
    
    # 手动启用梯度检查点（如果指定）
    if args.gradient_checkpointing and not args.auto_optimize:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Manual gradient checkpointing enabled")
    
    # 打印最终的参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFinal parameter count:")
    print(f"  Total: {total_params/1e9:.3f}B")
    print(f"  Trainable: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    
    # 3. 准备数据
    print("\n[2/4] Preparing data...")
    try:
        data_module = make_supervised_data_module(
            tokenizer=model.tokenizer,
            image_processor=model.image_processor,
            data_args={
                'train_data_path': args.train_data_path,
                'eval_data_path': args.eval_data_path,
                'image_folder': args.image_folder,
                'max_length': args.max_length,
                'use_mm_proj': True,
                'image_aspect_ratio': 'pad'
            }
        )
        
        train_dataset = data_module['train_dataset']
        eval_dataset = data_module['eval_dataset']
        
        print(f"Train samples: {len(train_dataset) if train_dataset else 0}")
        print(f"Eval samples: {len(eval_dataset) if eval_dataset else 0}")
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        raise
    
    # 4. 配置训练参数
    print("\n[3/4] Configuring trainer...")
    
    # 自动设置run_name
    if args.run_name is None:
        args.run_name = f"cmibq_stage{args.stage}_{args.model_size}_{args.target_bits_act}bit"
    
    # 根据模型大小调整精度设置
    if args.model_size in ['34b', '70b'] and not args.bf16:
        print("Note: Switching to bf16 for large model stability")
        args.bf16 = True
        args.fp16 = False
    
    training_args = {
        # 基础参数
        'output_dir': args.output_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        
        # 优化参数
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'max_grad_norm': args.max_grad_norm,
        
        # 精度设置
        'use_amp': args.use_amp or args.fp16 or args.bf16,
        'fp16': args.fp16,
        'bf16': args.bf16,
        'use_deepspeed': args.use_deepspeed,
        'gradient_checkpointing': args.gradient_checkpointing or args.auto_optimize,
        
        # 日志和保存
        'logging_steps': args.logging_steps,
        'save_epochs': args.save_epochs,
        'eval_epochs': args.eval_epochs,
        'use_wandb': args.use_wandb,
        'run_name': args.run_name,
        
        # 其他
        'num_workers': args.num_workers,
        'seed': args.seed,
        'model_size': args.model_size,
        'stage': args.stage,
        
        # DDP设置
        'find_unused_parameters': True,
        'static_graph': False
    }
    
    # 5. 创建训练器
    try:
        trainer = DistributedCMIBQTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.tokenizer
        )
        
        print("Trainer initialized successfully")
        
    except Exception as e:
        print(f"Error creating trainer: {e}")
        raise
    
    # 从检查点恢复
    if args.resume_from_checkpoint:
        print(f"\nResuming from checkpoint: {args.resume_from_checkpoint}")
        try:
            trainer.load_checkpoint(args.resume_from_checkpoint)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
    
    # 6. 开始训练前的最终内存检查
    print("\n[4/4] Pre-training memory status:")
    check_gpu_memory()
    
    # 7. 开始训练
    print("\nStarting training...")
    print("=" * 80)

    try:
        trainer.train()
        
        # 获取最终的量化统计（新增）
        if hasattr(model, 'get_quantization_stats'):
            final_stats = model.get_quantization_stats()
            print("\n" + "=" * 80)
            print("Final Quantization Statistics:")
            print("=" * 80)
            if 'weight_avg_bits' in final_stats:
                print(f"Weight average bits: {final_stats['weight_avg_bits']:.2f}")
                print(f"Weight quantized layers: {final_stats.get('weight_quantized_layers', 0)}")
            print(f"Activation average bits (vision): {final_stats.get('vision_act_avg_bits', 'N/A')}")
            print(f"Activation average bits (projector): {final_stats.get('projector_act_avg_bits', 'N/A')}")
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model')}")
        print("=" * 80)



        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted_checkpoint')
        print("Checkpoint saved")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Attempting to save emergency checkpoint...")
        try:
            trainer.save_checkpoint('emergency_checkpoint')
            print("Emergency checkpoint saved")
        except:
            print("Failed to save emergency checkpoint")
        raise
    
    finally:
        # 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        print("\nCleanup completed")


if __name__ == '__main__':
    main()
