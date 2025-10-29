#!/usr/bin/env python3
"""
CIBD训练脚本 - V2 with RDO
真正集成了率失真优化器和自适应调度器
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import argparse
from tqdm import tqdm
import logging
from typing import Optional, Dict, Any
import numpy as np

# 导入LLaVA组件
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# 导入CIBD组件（只使用V2）
from cibd.dataset import LLaVADataset, LLaVACollator
from cibd.llava_cibd_model_v2 import LlavaCIBDModelV2
from cibd.create_student_model_v2 import (
    create_compressed_student_config_v2,
    initialize_student_from_teacher_v2,
    estimate_model_params
)

# 导入RDO组件
from cibd.rate_distortion_optimizer import (
    RateDistortionOptimizer,
    AdaptiveRateScheduler
)

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class CIBDTrainerWithRDO:
    """
    CIBD训练器 - 集成RDO优化
    
    新功能：
    1. 动态调整 β (log_beta) 参数
    2. 自适应压缩率调度
    3. Pareto前沿追踪
    """
    
    def __init__(self, model, train_loader, val_loader, args, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        num_training_steps = len(train_loader) * args.num_epochs
        num_warmup_steps = int(num_training_steps * args.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # ===== 新增：RDO 优化器 =====
        self.rdo = RateDistortionOptimizer(
            initial_beta=args.initial_beta,
            target_compression=args.target_compression_rate,
            adaptation_rate=args.beta_adaptation_rate,
            window_size=100
        )
        
        # ===== 新增：自适应压缩率调度器 =====
        if args.use_adaptive_compression:
            self.compression_scheduler = AdaptiveRateScheduler(
                initial_rate=args.initial_compression_rate,
                target_rate=args.target_compression_rate,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )
        else:
            self.compression_scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 日志文件
        self.log_file = os.path.join(args.output_dir, 'training_log.txt')
        self.rdo_log_file = os.path.join(args.output_dir, 'rdo_log.txt')
        
        logger.info("\n" + "=" * 60)
        logger.info("训练器初始化完成（with RDO）")
        logger.info("=" * 60)
        logger.info(f"训练样本: {len(train_loader.dataset)}")
        logger.info(f"验证样本: {len(val_loader.dataset) if val_loader else 0}")
        logger.info(f"总步数: {num_training_steps}")
        logger.info(f"预热步数: {num_warmup_steps}")
        logger.info(f"学习率: {args.learning_rate}")
        logger.info(f"批次大小: {args.batch_size}")
        logger.info(f"梯度累积: {args.gradient_accumulation_steps}")
        logger.info(f"\n🔧 RDO 配置:")
        logger.info(f"  初始 β: {args.initial_beta}")
        logger.info(f"  目标压缩率: {args.target_compression_rate}")
        logger.info(f"  β 自适应率: {args.beta_adaptation_rate}")
        logger.info(f"  自适应压缩: {args.use_adaptive_compression}")
        logger.info("=" * 60 + "\n")
    
    def train(self):
        """主训练循环"""
        logger.info("\n" + "=" * 60)
        logger.info("开始训练（with RDO）")
        logger.info("=" * 60 + "\n")
        
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            logger.info(f"{'='*60}")
            
            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            # 记录训练指标
            self._log_metrics(epoch, train_metrics, phase='train')
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self._validate()
                self._log_metrics(epoch, val_metrics, phase='val')
                
                # 保存最佳模型
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self._save_checkpoint('best_model')
                    logger.info(f"✓ 保存最佳模型 (val_loss={self.best_val_loss:.4f})")
            
            # 定期保存checkpoint
            if (epoch + 1) % self.args.save_steps == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}')
            
            # 保存Pareto前沿
            self._save_pareto_frontier(epoch)
        
        logger.info("\n" + "=" * 60)
        logger.info("训练完成!")
        logger.info("=" * 60 + "\n")
    
    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        loss_components = {
            'task_loss': 0,
            'kl_loss': 0,
            'feature_loss': 0,
            'ib_loss': 0,
            'layer_0_loss': 0,
            'layer_1_loss': 0,
            'layer_2_loss': 0,
        }
        
        # RDO 统计
        rdo_stats = {
            'beta_values': [],
            'compression_rates': [],
            'rate_values': [],
            'distortion_values': []
        }
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            dynamic_ncols=True
        )
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # ===== 1. 自适应压缩率调度 =====
            if self.compression_scheduler is not None:
                current_compression_rate = self.compression_scheduler.get_compression_rate()
                # 可以用这个调整 IB 的瓶颈维度或其他参数
                # 这里我们将它记录下来
                rdo_stats['compression_rates'].append(current_compression_rate)
            
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # ===== 2. RDO 优化：动态调整 β =====
            if hasattr(outputs, 'loss_info'):
                task_loss = outputs.loss_info.get('task_loss', 0)
                ib_loss = outputs.loss_info.get('ib_loss', 0)
                
                # 如果有 IB 损失，使用 RDO 优化
                if ib_loss > 0 and task_loss > 0:
                    # 计算率（IB KL）和失真（任务损失）
                    rate = outputs.loss_info.get('ib_kl', ib_loss)
                    distortion = task_loss
                    
                    # RDO 步骤：计算最优 β
                    optimal_beta = self.rdo.step(
                        torch.tensor(distortion).to(self.device),
                        torch.tensor(rate).to(self.device)
                    )
                    
                    # ===== 关键：更新模型的 log_beta =====
                    if hasattr(self.model, 'visual_ib') and hasattr(self.model.visual_ib, 'log_beta'):
                        with torch.no_grad():
                            self.model.visual_ib.log_beta.data = torch.tensor(
                                np.log(optimal_beta),
                                dtype=torch.float32,
                                device=self.device
                            )
                    
                    # 记录 RDO 统计
                    rdo_stats['beta_values'].append(optimal_beta)
                    rdo_stats['rate_values'].append(rate)
                    rdo_stats['distortion_values'].append(distortion)
            
            # 梯度累积
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            
            # 累积损失
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            
            # 累积损失组件
            if hasattr(outputs, 'loss_info'):
                for key in loss_components:
                    if key in outputs.loss_info:
                        loss_components[key] += outputs.loss_info[key]
            
            # 更新参数
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 更新进度条
                current_lr = self.scheduler.get_last_lr()[0]
                current_beta = rdo_stats['beta_values'][-1] if rdo_stats['beta_values'] else 1.0
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'β': f'{current_beta:.3f}'
                })
        
        # 计算平均
        num_batches = len(self.train_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()}
        }
        
        # 添加 RDO 统计
        if rdo_stats['beta_values']:
            metrics['avg_beta'] = np.mean(rdo_stats['beta_values'])
            metrics['std_beta'] = np.std(rdo_stats['beta_values'])
            metrics['avg_rate'] = np.mean(rdo_stats['rate_values'])
            metrics['avg_distortion'] = np.mean(rdo_stats['distortion_values'])
        
        if rdo_stats['compression_rates']:
            metrics['avg_compression_rate'] = np.mean(rdo_stats['compression_rates'])
        
        # 记录 RDO 日志
        self._log_rdo_stats(rdo_stats)
        
        return metrics
    
    def _validate(self):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        loss_components = {
            'task_loss': 0,
            'kl_loss': 0,
            'feature_loss': 0,
            'ib_loss': 0,
            'layer_0_loss': 0,
            'layer_1_loss': 0,
            'layer_2_loss': 0,
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", dynamic_ncols=True):
                # 移动到设备
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                # 累积损失组件
                if hasattr(outputs, 'loss_info'):
                    for key in loss_components:
                        if key in outputs.loss_info:
                            loss_components[key] += outputs.loss_info[key]
        
        # 计算平均
        num_batches = len(self.val_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()}
        }
        
        return metrics
    
    def _log_metrics(self, epoch, metrics, phase='train'):
        """记录指标"""
        log_str = f"\n[{phase.upper()}] Epoch {epoch + 1}\n"
        log_str += f"  Total Loss: {metrics['total_loss']:.4f}\n"
        log_str += f"  Task Loss: {metrics['task_loss']:.4f}\n"
        log_str += f"  KL Loss: {metrics['kl_loss']:.4f}\n"
        log_str += f"  Feature Loss: {metrics['feature_loss']:.4f}\n"
        log_str += f"  IB Loss: {metrics['ib_loss']:.4f}\n"
        
        # RDO 指标
        if 'avg_beta' in metrics:
            log_str += f"\n  🔧 RDO Stats:\n"
            log_str += f"    Avg β: {metrics['avg_beta']:.4f} ± {metrics['std_beta']:.4f}\n"
            log_str += f"    Avg Rate: {metrics['avg_rate']:.4f}\n"
            log_str += f"    Avg Distortion: {metrics['avg_distortion']:.4f}\n"
        
        if 'avg_compression_rate' in metrics:
            log_str += f"    Compression Rate: {metrics['avg_compression_rate']:.4f}\n"
        
        logger.info(log_str)
        
        # 写入文件
        with open(self.log_file, 'a') as f:
            f.write(log_str + "\n")
    
    def _log_rdo_stats(self, rdo_stats):
        """记录 RDO 详细统计"""
        if not rdo_stats['beta_values']:
            return
        
        log_str = f"\nStep {self.global_step} RDO Stats:\n"
        log_str += f"  β: {np.mean(rdo_stats['beta_values']):.4f}\n"
        log_str += f"  Rate: {np.mean(rdo_stats['rate_values']):.4f}\n"
        log_str += f"  Distortion: {np.mean(rdo_stats['distortion_values']):.4f}\n"
        
        with open(self.rdo_log_file, 'a') as f:
            f.write(log_str)
    
    def _save_pareto_frontier(self, epoch):
        """保存Pareto前沿"""
        rates, distortions = self.rdo.get_pareto_frontier()
        
        if rates:
            pareto_data = {
                'epoch': epoch,
                'rates': rates,
                'distortions': distortions
            }
            
            pareto_file = os.path.join(
                self.args.output_dir, 
                f'pareto_frontier_epoch_{epoch}.json'
            )
            
            with open(pareto_file, 'w') as f:
                json.dump(pareto_data, f, indent=2)
            
            logger.info(f"✓ Saved Pareto frontier: {len(rates)} points")
    
    def _save_checkpoint(self, name):
        """保存checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_path)
        
        # 保存训练状态
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'rdo_beta': self.rdo.beta,
            'rdo_history': {
                'rate_history': list(self.rdo.rate_history),
                'distortion_history': list(self.rdo.distortion_history),
                'beta_history': list(self.rdo.beta_history),
            }
        }
        
        if self.compression_scheduler is not None:
            state['compression_scheduler_step'] = self.compression_scheduler.current_step
        
        torch.save(state, os.path.join(checkpoint_path, 'trainer_state.pt'))
        
        logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        logger.info(f"加载checkpoint: {checkpoint_path}")
        
        # 加载训练状态
        state_path = os.path.join(checkpoint_path, 'trainer_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.current_epoch = state['epoch']
            self.global_step = state['global_step']
            self.best_val_loss = state['best_val_loss']
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            
            # 恢复 RDO 状态
            if 'rdo_beta' in state:
                self.rdo.beta = state['rdo_beta']
            
            if 'rdo_history' in state:
                from collections import deque
                self.rdo.rate_history = deque(state['rdo_history']['rate_history'], maxlen=100)
                self.rdo.distortion_history = deque(state['rdo_history']['distortion_history'], maxlen=100)
                self.rdo.beta_history = deque(state['rdo_history']['beta_history'], maxlen=100)
            
            if 'compression_scheduler_step' in state and self.compression_scheduler is not None:
                self.compression_scheduler.current_step = state['compression_scheduler_step']
            
            logger.info(f"✓ 从epoch {self.current_epoch} 恢复训练")
            logger.info(f"✓ RDO β 恢复为: {self.rdo.beta:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="CIBD Training Script V2 with RDO")
    
    # 模型参数
    parser.add_argument("--teacher_model_path", type=str, required=True,
                       help="教师模型路径")
    parser.add_argument("--compression_ratio", type=float, default=0.5,
                       help="压缩比率 (0.3-0.7)")
    parser.add_argument("--compress_hidden", action="store_true",
                       help="是否压缩hidden_size")
    parser.add_argument("--compress_heads", action="store_true",
                       help="是否压缩attention heads")
    
    # 数据参数
    parser.add_argument("--train_data_path", type=str, required=True,
                       help="训练数据JSON路径")
    parser.add_argument("--val_data_path", type=str, default=None,
                       help="验证数据JSON路径")
    parser.add_argument("--image_folder", type=str, required=True,
                       help="图像文件夹路径")
    parser.add_argument("--auto_split", action="store_true",
                       help="自动分割训练/验证集")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="验证集比例（用于auto_split）")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # ===== RDO 参数 =====
    parser.add_argument("--initial_beta", type=float, default=1.0,
                       help="初始 β 值")
    parser.add_argument("--target_compression_rate", type=float, default=0.5,
                       help="目标压缩率（用于RDO）")
    parser.add_argument("--beta_adaptation_rate", type=float, default=0.01,
                       help="β 自适应学习率")
    parser.add_argument("--use_adaptive_compression", action="store_true",
                       help="使用自适应压缩率调度")
    parser.add_argument("--initial_compression_rate", type=float, default=0.1,
                       help="初始压缩率（用于调度器）")
    
    # 数据加载
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    
    # 保存和日志
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=1,
                       help="每N个epoch保存一次")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的checkpoint路径")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("\n" + "=" * 60)
    logger.info("CIBD Training V2 with RDO")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Compression ratio: {args.compression_ratio}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"RDO enabled: ✓")
    logger.info(f"Initial β: {args.initial_beta}")
    logger.info(f"Target compression rate: {args.target_compression_rate}")
    logger.info("=" * 60 + "\n")
    
    # 1. 加载教师模型
    logger.info("\n" + "=" * 60)
    logger.info("加载教师模型...")
    logger.info("=" * 60)
    
    tokenizer, teacher_model, image_processor, context_len = load_pretrained_model(
        model_path=args.teacher_model_path,
        model_base=None,
        model_name=args.teacher_model_path.split('/')[-1],
        device_map=None
    )
    
    # 确保vocab大小正确
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    if hasattr(teacher_model.config, 'vocab_size'):
        tokenizer_vocab_size = len(tokenizer)
        if teacher_model.config.vocab_size != tokenizer_vocab_size:
            logger.warning(f"  ⚠️  vocab_size不匹配:")
            logger.info(f"      config: {teacher_model.config.vocab_size}")
            logger.info(f"      tokenizer: {tokenizer_vocab_size}")
            teacher_model.config.vocab_size = tokenizer_vocab_size
            logger.info(f"  ✓ 已更新为: {tokenizer_vocab_size}")

    teacher_model.to(device)
    teacher_model.eval()
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"✓ 教师模型: {teacher_params/1e6:.2f}M 参数")
    
    # 2. 创建学生模型
    logger.info("\n" + "=" * 60)
    logger.info("创建学生模型...")
    logger.info("=" * 60)
    
    teacher_config = teacher_model.config
    student_config = create_compressed_student_config_v2(
        teacher_config,
        compression_ratio=args.compression_ratio,
        compress_hidden=args.compress_hidden,
        compress_heads=args.compress_heads
    )
    
    logger.info(f"\n压缩策略:")
    logger.info(f"  compress_hidden: {args.compress_hidden}")
    logger.info(f"  compress_heads: {args.compress_heads}")
    
    # 使用 V2 模型
    student_model = LlavaCIBDModelV2(student_config, teacher_model)
    
    # 检查并验证 embedding 大小
    logger.info("\n检查学生模型 embedding 层...")
    if hasattr(student_model.model, 'embed_tokens'):
        actual_embed_vocab = student_model.model.embed_tokens.num_embeddings
        logger.info(f"  embed_tokens.num_embeddings: {actual_embed_vocab}")
        
        if actual_embed_vocab != len(tokenizer):
            logger.warning(f"  ⚠️  Embedding 大小不匹配，进行 resize...")
            student_model.resize_token_embeddings(len(tokenizer))
            logger.info(f"  ✓ Resized to {len(tokenizer)}")
        else:
            logger.info(f"  ✓ Embedding 大小正确")

    if hasattr(student_model, 'lm_head'):
        actual_lmhead_vocab = student_model.lm_head.out_features
        logger.info(f"  lm_head.out_features: {actual_lmhead_vocab}")
        
        if actual_lmhead_vocab != len(tokenizer):
            logger.error(f"  ❌ LM head 大小不匹配!")
            raise ValueError("LM head size mismatch!")
    
    # 初始化权重 (使用 V2)
    logger.info("从教师模型初始化学生权重...")
    student_model = initialize_student_from_teacher_v2(
        student_model, teacher_model, student_config
    )
    
    logger.info("="*60)
    logger.info("复制视觉组件（不压缩）...")
    logger.info("="*60)

    # 复制vision tower
    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'vision_tower'):
        logger.info("复制vision tower...")
        student_model.model.vision_tower = teacher_model.model.vision_tower
        # 冻结vision tower参数
        for param in student_model.model.vision_tower.parameters():
            param.requires_grad = False
        logger.info("✓ Vision tower已复制并冻结")
    else:
        logger.error("✗ 未找到teacher的vision tower!")
        raise RuntimeError("Cannot find vision_tower in teacher model")

    # 复制mm_projector
    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'mm_projector'):
        logger.info("复制mm_projector...")
        student_model.model.mm_projector = teacher_model.model.mm_projector
        logger.info("✓ MM projector已复制")

    logger.info("="*60)
    student_model.to(device)
    
    # 最终验证
    logger.info("\n" + "=" * 60)
    logger.info("最终验证...")
    logger.info("=" * 60)
    
    logger.info(f"Tokenizer vocab: {len(tokenizer)}")
    logger.info(f"Teacher embed: {teacher_model.model.embed_tokens.num_embeddings}")
    logger.info(f"Student embed: {student_model.model.embed_tokens.num_embeddings}")
    logger.info(f"Teacher lm_head: {teacher_model.lm_head.out_features}")
    logger.info(f"Student lm_head: {student_model.lm_head.out_features}")
    
    # 测试一个样本
    logger.info("\n测试 embedding lookup...")
    try:
        test_text = "USER: <image>\nWhat is this? ASSISTANT:"
        test_ids = tokenizer(test_text, return_tensors='pt').input_ids
        logger.info(f"  Test IDs shape: {test_ids.shape}")
        logger.info(f"  Test IDs 范围: [{test_ids.min().item()}, {test_ids.max().item()}]")
        
        # 检查是否有超范围的 token
        max_id = test_ids.max().item()
        vocab_size = len(tokenizer)
        if max_id >= vocab_size:
            logger.error(f"  ❌ Token ID {max_id} >= vocab_size {vocab_size}!")
            raise ValueError("Token ID out of range!")
        
        with torch.no_grad():
            test_embed = student_model.model.embed_tokens(test_ids.to(device))
        logger.info(f"  ✓ Embedding 成功! shape: {test_embed.shape}")
    except Exception as e:
        logger.error(f"  ❌ Embedding 失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 分析压缩
    student_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"\n压缩分析:")
    logger.info(f"  教师: {teacher_params/1e9:.2f}B")
    logger.info(f"  学生: {student_params/1e9:.2f}B")
    logger.info(f"  压缩率: {(1 - student_params/teacher_params):.2%}")
    
    # 3. 准备数据
    logger.info("\n" + "=" * 60)
    logger.info("准备数据集...")
    logger.info("=" * 60)
    
    # 处理训练/验证分割
    if args.val_data_path:
        train_json = args.train_data_path
        val_json = args.val_data_path
        logger.info(f"使用提供的验证集")
    elif args.auto_split:
        logger.info(f"自动分割数据 ({args.val_split:.1%} 验证)")
        
        with open(args.train_data_path, 'r') as f:
            all_data = json.load(f)
        
        split_idx = int(len(all_data) * (1 - args.val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        os.makedirs(args.output_dir, exist_ok=True)
        train_json = os.path.join(args.output_dir, 'train_split.json')
        val_json = os.path.join(args.output_dir, 'val_split.json')
        
        with open(train_json, 'w') as f:
            json.dump(train_data, f)
        with open(val_json, 'w') as f:
            json.dump(val_data, f)
        
        logger.info(f"分割: {len(train_data)} 训练, {len(val_data)} 验证")
    else:
        train_json = args.train_data_path
        val_json = None
        logger.info("仅训练，无验证")
    
    # 创建数据集
    train_dataset = LLaVADataset(
        train_json,
        args.image_folder,
        tokenizer,
        image_processor,
        max_length=args.max_length,
        is_training=True
    )
    
    val_dataset = None
    if val_json:
        val_dataset = LLaVADataset(
            val_json,
            args.image_folder,
            tokenizer,
            image_processor,
            max_length=args.max_length,
            is_training=False
        )
    
    logger.info(f"✓ 训练集: {len(train_dataset)} 样本")
    if val_dataset:
        logger.info(f"✓ 验证集: {len(val_dataset)} 样本")
    
    # 数据加载器
    collator = LLaVACollator(tokenizer, image_processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # 4. 创建训练器（with RDO）
    trainer = CIBDTrainerWithRDO(student_model, train_loader, val_loader, args, device)
    
    # 恢复训练（如果需要）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 5. 开始训练
    trainer.train()
    
    # 6. 保存最终模型
    logger.info("\n" + "=" * 60)
    logger.info("保存最终模型...")
    logger.info("=" * 60)
    
    final_path = os.path.join(args.output_dir, 'final_model')
    student_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info(f"✓ 最终模型: {final_path}")
    
    # 保存统计信息
    student_params = sum(p.numel() for p in student_model.parameters())
    stats = {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': 1 - (student_params / teacher_params),
        'config': student_config.to_dict(),
        'final_beta': trainer.rdo.beta,
        'rdo_config': {
            'initial_beta': args.initial_beta,
            'target_compression_rate': args.target_compression_rate,
            'beta_adaptation_rate': args.beta_adaptation_rate,
        }
    }
    
    with open(os.path.join(args.output_dir, 'compression_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 训练完成!")
    logger.info(f"✓ 最终 β: {trainer.rdo.beta:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
