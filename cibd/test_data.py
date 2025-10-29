#!/usr/bin/env python3
"""
CIBD训练脚本 - V2 with RDO (Multi-GPU)
支持：
1. DataParallel (DP)
2. DistributedDataParallel (DDP) 
3. DeepSpeed
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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

# 导入CIBD组件
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


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


class CIBDTrainerMultiGPU:
    """
    CIBD训练器 - 多GPU版本
    
    支持：
    1. DataParallel (DP) - 简单多卡
    2. DistributedDataParallel (DDP) - 推荐多卡
    3. DeepSpeed - 大模型优化
    """
    
    def __init__(self, model, train_loader, val_loader, args, device, 
                 rank=0, world_size=1, local_rank=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main_process = (rank == 0)
        
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
        
        # RDO 优化器（只在主进程）
        if self.is_main_process:
            self.rdo = RateDistortionOptimizer(
                initial_beta=args.initial_beta,
                target_compression=args.target_compression_rate,
                adaptation_rate=args.beta_adaptation_rate,
                window_size=100
            )
        else:
            self.rdo = None
        
        # 自适应压缩率调度器
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
        
        # 创建输出目录（只在主进程）
        if self.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            self.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # 日志文件
            self.log_file = os.path.join(args.output_dir, 'training_log.txt')
            self.rdo_log_file = os.path.join(args.output_dir, 'rdo_log.txt')
        
        if self.is_main_process:
            logger.info("\n" + "=" * 60)
            logger.info("训练器初始化完成（Multi-GPU with RDO）")
            logger.info("=" * 60)
            logger.info(f"World Size: {world_size}")
            logger.info(f"Rank: {rank}")
            logger.info(f"Local Rank: {local_rank}")
            logger.info(f"训练样本: {len(train_loader.dataset)}")
            logger.info(f"验证样本: {len(val_loader.dataset) if val_loader else 0}")
            logger.info(f"总步数: {num_training_steps}")
            logger.info(f"预热步数: {num_warmup_steps}")
            logger.info(f"学习率: {args.learning_rate}")
            logger.info(f"批次大小: {args.batch_size} (per GPU)")
            logger.info(f"全局批次: {args.batch_size * world_size}")
            logger.info(f"梯度累积: {args.gradient_accumulation_steps}")
            logger.info(f"\n🔧 RDO 配置:")
            logger.info(f"  初始 β: {args.initial_beta}")
            logger.info(f"  目标压缩率: {args.target_compression_rate}")
            logger.info(f"  β 自适应率: {args.beta_adaptation_rate}")
            logger.info("=" * 60 + "\n")
    
    def train(self):
        """主训练循环"""
        if self.is_main_process:
            logger.info("\n" + "=" * 60)
            logger.info("开始训练（Multi-GPU with RDO）")
            logger.info("=" * 60 + "\n")
        
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            
            # 设置分布式采样器的epoch（确保每个epoch数据shuffle不同）
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            if self.is_main_process:
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
                logger.info(f"{'='*60}")
            
            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            # 同步所有进程
            if self.world_size > 1:
                dist.barrier()
            
            # 记录训练指标（只在主进程）
            if self.is_main_process:
                self._log_metrics(epoch, train_metrics, phase='train')
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self._validate()
                
                if self.is_main_process:
                    self._log_metrics(epoch, val_metrics, phase='val')
                    
                    # 保存最佳模型
                    if val_metrics['total_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['total_loss']
                        self._save_checkpoint('best_model')
                        logger.info(f"✓ 保存最佳模型 (val_loss={self.best_val_loss:.4f})")
            
            # 定期保存checkpoint
            if self.is_main_process and (epoch + 1) % self.args.save_steps == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}')
            
            # 保存Pareto前沿
            if self.is_main_process:
                self._save_pareto_frontier(epoch)
            
            # 同步所有进程
            if self.world_size > 1:
                dist.barrier()
        
        if self.is_main_process:
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
        
        # RDO 统计（只在主进程收集）
        rdo_stats = {
            'beta_values': [],
            'compression_rates': [],
            'rate_values': [],
            'distortion_values': []
        } if self.is_main_process else None
        
        self.optimizer.zero_grad()
        
        # 只在主进程显示进度条
        if self.is_main_process:
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.current_epoch + 1}",
                dynamic_ncols=True
            )
        else:
            progress_bar = self.train_loader
        
        for step, batch in enumerate(progress_bar):
            try:
                # ===== 【修复1】处理 images list =====
                # 1. 分离 images（它是 list，不能直接 .to(device)）
                images = batch.pop('images', None)
                
                # 2. 其他字段移到设备
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # 3. 处理 images list
                if images is not None and isinstance(images, list):
                    # 方案A：转换为张量（如果所有样本都有图）
                    if all(img is not None for img in images):
                        # 所有样本都有图，可以 stack
                        images = torch.stack(images).to(self.device)
                    else:
                        # 有些样本无图，保持 list 格式
                        # 将有效图片移到设备，None 保持为 None
                        images = [img.to(self.device) if img is not None else None 
                                for img in images]
                elif torch.is_tensor(images):
                    images = images.to(self.device)
                
                # 4. 放回 batch
                batch['images'] = images
                # ===== 【修复1结束】=====
                
                # ===== 【修复2】前向传播（放在 try 块内）=====
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    images=batch['images'],
                    return_dict=True
                )
                
                # 获取损失
                loss = outputs.loss
                # ===== 【修复2结束】=====
                
                # ===== 【修复3】RDO 优化（现在 outputs 已定义）=====
                if hasattr(outputs, 'loss_info'):
                    task_loss = outputs.loss_info.get('task_loss', 0)
                    ib_loss = outputs.loss_info.get('ib_loss', 0)
                    
                    if ib_loss > 0 and task_loss > 0:
                        if self.is_main_process:
                            # 主进程计算最优 β
                            rate = outputs.loss_info.get('ib_kl', ib_loss)
                            distortion = task_loss
                            
                            optimal_beta = self.rdo.step(
                                torch.tensor(distortion).to(self.device),
                                torch.tensor(rate).to(self.device)
                            )
                            
                            # 记录统计
                            rdo_stats['beta_values'].append(optimal_beta)
                            rdo_stats['rate_values'].append(rate)
                            rdo_stats['distortion_values'].append(distortion)
                        else:
                            optimal_beta = 1.0  # 非主进程的默认值
                        
                        # 广播 β 到所有进程
                        if self.world_size > 1:
                            beta_tensor = torch.tensor(optimal_beta, device=self.device)
                            dist.broadcast(beta_tensor, src=0)
                            optimal_beta = beta_tensor.item()
                        
                        # 更新模型的 log_beta
                        model_to_update = self.model.module if hasattr(self.model, 'module') else self.model
                        if hasattr(model_to_update, 'visual_ib') and hasattr(model_to_update.visual_ib, 'log_beta'):
                            with torch.no_grad():
                                model_to_update.visual_ib.log_beta.data = torch.tensor(
                                    np.log(optimal_beta),
                                    dtype=torch.float32,
                                    device=self.device
                                )
                # ===== 【修复3结束】=====
                
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
                    
                    # 更新进度条（只在主进程）
                    if self.is_main_process:
                        current_lr = self.scheduler.get_last_lr()[0]
                        current_beta = rdo_stats['beta_values'][-1] if rdo_stats['beta_values'] else 1.0
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'lr': f'{current_lr:.2e}',
                            'β': f'{current_beta:.3f}'
                        })
            
            except Exception as e:
                # ===== 【修复4】异常处理 =====
                if self.is_main_process:
                    logger.error(f"Error in training step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 清理梯度并跳过这个 batch
                self.optimizer.zero_grad()
                continue
                # ===== 【修复4结束】=====
        
        # 聚合所有GPU的损失
        if self.world_size > 1:
            # 将损失转换为tensor并求平均
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.AVG)
            total_loss = total_loss_tensor.item()
            
            # 聚合损失组件
            for key in loss_components:
                loss_tensor = torch.tensor(loss_components[key], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                loss_components[key] = loss_tensor.item()
        
        # 计算平均
        num_batches = len(self.train_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()}
        }
        
        # 添加 RDO 统计（只在主进程）
        if self.is_main_process and rdo_stats and rdo_stats['beta_values']:
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
            val_iter = tqdm(self.val_loader, desc="Validation", dynamic_ncols=True) if self.is_main_process else self.val_loader
            
            for batch in val_iter:
                # 移动到设备
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    images=batch.get('images'),
                    image_sizes=batch.get('image_sizes'),  # ← 添加这个
                    return_dict=True
                )
                loss = outputs.loss
                
                total_loss += loss.item()
                
                # 累积损失组件
                if hasattr(outputs, 'loss_info'):
                    for key in loss_components:
                        if key in outputs.loss_info:
                            loss_components[key] += outputs.loss_info[key]
        
        # 聚合所有GPU的损失
        if self.world_size > 1:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.AVG)
            total_loss = total_loss_tensor.item()
            
            for key in loss_components:
                loss_tensor = torch.tensor(loss_components[key], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                loss_components[key] = loss_tensor.item()
        
        # 计算平均
        num_batches = len(self.val_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()}
        }
        
        return metrics
    
    def _log_metrics(self, epoch, metrics, phase='train'):
        """记录指标（只在主进程）"""
        if not self.is_main_process:
            return
        
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
        if not self.is_main_process or not rdo_stats['beta_values']:
            return
        
        log_str = f"\nStep {self.global_step} RDO Stats:\n"
        log_str += f"  β: {np.mean(rdo_stats['beta_values']):.4f}\n"
        log_str += f"  Rate: {np.mean(rdo_stats['rate_values']):.4f}\n"
        log_str += f"  Distortion: {np.mean(rdo_stats['distortion_values']):.4f}\n"
        
        with open(self.rdo_log_file, 'a') as f:
            f.write(log_str)
    
    def _save_pareto_frontier(self, epoch):
        """保存Pareto前沿"""
        if not self.is_main_process or self.rdo is None:
            return
        
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
        """保存checkpoint（只在主进程）"""
        if not self.is_main_process:
            return
        
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 获取实际的模型（去除DDP wrapper）
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 保存模型
        model_to_save.save_pretrained(checkpoint_path)
        
        # 保存训练状态
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        
        if self.rdo is not None:
            state['rdo_beta'] = self.rdo.beta
            state['rdo_history'] = {
                'rate_history': list(self.rdo.rate_history),
                'distortion_history': list(self.rdo.distortion_history),
                'beta_history': list(self.rdo.beta_history),
            }
        
        if self.compression_scheduler is not None:
            state['compression_scheduler_step'] = self.compression_scheduler.current_step
        
        torch.save(state, os.path.join(checkpoint_path, 'trainer_state.pt'))
        
        logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        if self.is_main_process:
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
            
            # 恢复 RDO 状态（只在主进程）
            if self.is_main_process and self.rdo is not None:
                if 'rdo_beta' in state:
                    self.rdo.beta = state['rdo_beta']
                
                if 'rdo_history' in state:
                    from collections import deque
                    self.rdo.rate_history = deque(state['rdo_history']['rate_history'], maxlen=100)
                    self.rdo.distortion_history = deque(state['rdo_history']['distortion_history'], maxlen=100)
                    self.rdo.beta_history = deque(state['rdo_history']['beta_history'], maxlen=100)
            
            if 'compression_scheduler_step' in state and self.compression_scheduler is not None:
                self.compression_scheduler.current_step = state['compression_scheduler_step']
            
            if self.is_main_process:
                logger.info(f"✓ 从epoch {self.current_epoch} 恢复训练")
                if self.rdo is not None:
                    logger.info(f"✓ RDO β 恢复为: {self.rdo.beta:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="CIBD Training Script V2 with RDO (Multi-GPU)")
    
    # 分布式训练参数
    parser.add_argument("--distributed_backend", type=str, default="ddp",
                       choices=["dp", "ddp", "deepspeed"],
                       help="分布式训练后端: dp (DataParallel), ddp (DistributedDataParallel), deepspeed")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="DDP: local rank (自动设置)")
    
    # 模型参数
    parser.add_argument("--teacher_model_path", type=str, required=True)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--compress_hidden", action="store_true")
    parser.add_argument("--compress_heads", action="store_true")
    
    # 数据参数
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--auto_split", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.1)
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                       help="每个GPU的batch size")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # RDO 参数
    parser.add_argument("--initial_beta", type=float, default=1.0)
    parser.add_argument("--target_compression_rate", type=float, default=0.5)
    parser.add_argument("--beta_adaptation_rate", type=float, default=0.01)
    parser.add_argument("--use_adaptive_compression", action="store_true")
    parser.add_argument("--initial_compression_rate", type=float, default=0.1)
    
    # 数据加载
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    
    # 保存和日志
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    # 设置设备
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_main_process = (rank == 0)
    
    if is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info("CIBD Training V2 with RDO (Multi-GPU)")
        logger.info("=" * 60)
        logger.info(f"World Size: {world_size}")
        logger.info(f"Rank: {rank}")
        logger.info(f"Local Rank: {local_rank}")
        logger.info(f"Device: {device}")
        logger.info(f"Backend: {args.distributed_backend}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Global batch size: {args.batch_size * world_size}")
        logger.info("=" * 60 + "\n")
    
    # 1. 加载教师模型（只在主进程打印日志）
    if is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info("加载教师模型...")
        logger.info("=" * 60)
    
    tokenizer, teacher_model, image_processor, context_len = load_pretrained_model(
        model_path=args.teacher_model_path,
        model_base=None,
        model_name=args.teacher_model_path.split('/')[-1],
        device_map=None
    )
    
    if is_main_process:
        logger.info(f"✓ Image processor 类型: {type(image_processor)}")
        logger.info(f"  Size: {getattr(image_processor, 'size', 'N/A')}")
        logger.info(f"  Crop size: {getattr(image_processor, 'crop_size', 'N/A')}")
    # 确保vocab大小正确
    if is_main_process:
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    if hasattr(teacher_model.config, 'vocab_size'):
        tokenizer_vocab_size = len(tokenizer)
        if teacher_model.config.vocab_size != tokenizer_vocab_size:
            teacher_model.config.vocab_size = tokenizer_vocab_size
            if is_main_process:
                logger.info(f"  ✓ 已更新 vocab_size 为: {tokenizer_vocab_size}")
    
    teacher_model.to(device)
    teacher_model.eval()
    
    if is_main_process:
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info(f"✓ 教师模型: {teacher_params/1e6:.2f}M 参数")
    
    # 2. 创建学生模型
    if is_main_process:
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
    
    student_model = LlavaCIBDModelV2(student_config, teacher_model)
    
    # Embedding 检查
    if hasattr(student_model.model, 'embed_tokens'):
        actual_embed_vocab = student_model.model.embed_tokens.num_embeddings
        if actual_embed_vocab != len(tokenizer):
            student_model.resize_token_embeddings(len(tokenizer))
            if is_main_process:
                logger.info(f"  ✓ Embedding resized to {len(tokenizer)}")
    
    # 初始化权重
    if is_main_process:
        logger.info("从教师模型初始化学生权重...")
    
    student_model = initialize_student_from_teacher_v2(
        student_model, teacher_model, student_config
    )
    
    # 复制视觉组件
    if is_main_process:
        logger.info("复制视觉组件...")
    
    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'vision_tower'):
        student_model.model.vision_tower = teacher_model.model.vision_tower
        for param in student_model.model.vision_tower.parameters():
            param.requires_grad = False
        if is_main_process:
            logger.info("✓ Vision tower已复制并冻结")
    
    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'mm_projector'):
        student_model.model.mm_projector = teacher_model.model.mm_projector
        if is_main_process:
            logger.info("✓ MM projector已复制")
    
    # 将模型移到设备
    student_model.to(device)
    student_model = student_model.float()
    # 包装模型（多GPU）
    if world_size > 1:
        if args.distributed_backend == "ddp":
            student_model = DDP(
                student_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True  # 因为有些参数被冻结
            )
            if is_main_process:
                logger.info("✓ 使用 DistributedDataParallel")
        elif args.distributed_backend == "dp":
            student_model = nn.DataParallel(student_model)
            if is_main_process:
                logger.info("✓ 使用 DataParallel")
    else:
        if is_main_process:
            logger.info("✓ 使用单GPU训练")
    
    if is_main_process:
        student_params = sum(p.numel() for p in student_model.parameters())
        logger.info(f"\n压缩分析:")
        logger.info(f"  教师: {teacher_params/1e9:.2f}B")
        logger.info(f"  学生: {student_params/1e9:.2f}B")
        logger.info(f"  压缩率: {(1 - student_params/teacher_params):.2%}")
    
    # 3. 准备数据
    if is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info("准备数据集...")
        logger.info("=" * 60)
    
    # 处理数据分割
    if args.val_data_path:
        train_json = args.train_data_path
        val_json = args.val_data_path
    elif args.auto_split:
        with open(args.train_data_path, 'r') as f:
            all_data = json.load(f)
        
        split_idx = int(len(all_data) * (1 - args.val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        if is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            train_json = os.path.join(args.output_dir, 'train_split.json')
            val_json = os.path.join(args.output_dir, 'val_split.json')
            
            with open(train_json, 'w') as f:
                json.dump(train_data, f)
            with open(val_json, 'w') as f:
                json.dump(val_data, f)
            
            logger.info(f"分割: {len(train_data)} 训练, {len(val_data)} 验证")
        
        # 同步所有进程
        if world_size > 1:
            dist.barrier()
            # 非主进程也需要知道文件路径
            if not is_main_process:
                train_json = os.path.join(args.output_dir, 'train_split.json')
                val_json = os.path.join(args.output_dir, 'val_split.json')
    else:
        train_json = args.train_data_path
        val_json = None
    
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
    
    if is_main_process:
        logger.info(f"✓ 训练集: {len(train_dataset)} 样本")
        if val_dataset:
            logger.info(f"✓ 验证集: {len(val_dataset)} 样本")
    
    # 数据加载器（使用DistributedSampler）
    collator = LLaVACollator(tokenizer, image_processor)
    
    if world_size > 1 and args.distributed_backend == "ddp":
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        if world_size > 1 and args.distributed_backend == "ddp":
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            val_sampler = None
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # 4. 创建训练器
    trainer = CIBDTrainerMultiGPU(
        student_model, train_loader, val_loader, args, device,
        rank=rank, world_size=world_size, local_rank=local_rank
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 5. 开始训练
    trainer.train()
    
    # 6. 保存最终模型（只在主进程）
    if is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info("保存最终模型...")
        logger.info("=" * 60)
        
        final_path = os.path.join(args.output_dir, 'final_model')
        
        # 获取实际的模型
        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
        model_to_save.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        logger.info(f"✓ 最终模型: {final_path}")
        
        # 保存统计信息
        student_params = sum(p.numel() for p in model_to_save.parameters())
        stats = {
            'teacher_params': teacher_params,
            'student_params': student_params,
            'compression_ratio': 1 - (student_params / teacher_params),
            'config': student_config.to_dict(),
            'final_beta': trainer.rdo.beta if trainer.rdo else None,
            'world_size': world_size,
        }
        
        with open(os.path.join(args.output_dir, 'compression_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ 训练完成!")
        if trainer.rdo:
            logger.info(f"✓ 最终 β: {trainer.rdo.beta:.4f}")
        logger.info("=" * 60)
    
    # 清理分布式环境
    cleanup_distributed()


if __name__ == "__main__":
    main()