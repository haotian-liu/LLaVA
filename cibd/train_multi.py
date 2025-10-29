#!/usr/bin/env python3
"""
CIBD训练脚本 - V2 with RDO (Multi-GPU) - FIXED
支持：
1. DataParallel (DP)
2. DistributedDataParallel (DDP) 
3. DeepSpeed

修复：
1. loss_info 读取问题（ModelOutput 自定义属性）
2. 损失累积逻辑（使用原始损失值）
3. DDP 同步（所有分项损失）
4. 调试输出（第一个batch）
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

# ===== PyTorch 2.0+ 混合精度支持 =====
from torch.amp import autocast, GradScaler

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


# ===== 【修复1】安全读取 loss_info =====
def safe_get_loss_info(outputs: Any) -> Dict[str, float]:
    """
    安全地从 outputs 中提取 loss_info
    兼容 ModelOutput 自定义属性和 dict 两种格式
    """
    # 方法1：作为属性读取（ModelOutput）
    loss_info = getattr(outputs, 'loss_info', None)
    
    # 方法2：作为字典读取（dict）
    if loss_info is None and isinstance(outputs, dict):
        loss_info = outputs.get('loss_info', None)
    
    # 默认返回空字典
    if loss_info is None:
        return {}
    
    # 转换所有值为 float
    def to_float(x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)
    
    return {k: to_float(v) for k, v in loss_info.items()}


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
        
        # ===== 设置混合精度 =====
        if args.precision == 'bf16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            self.use_amp = True
            self.scaler = GradScaler('cuda', enabled=False)  # BF16 不需要 scaler
            if self.is_main_process:
                logger.info("✓ 使用 BF16 混合精度（推荐用于 H100）")
        elif args.precision == 'fp16':
            self.dtype = torch.float16
            self.use_amp = True
            self.scaler = GradScaler('cuda', enabled=True)  # FP16 需要 scaler
            if self.is_main_process:
                logger.info("✓ 使用 FP16 混合精度")
        else:
            self.dtype = torch.float32
            self.use_amp = False
            self.scaler = GradScaler('cuda', enabled=False)
            if self.is_main_process:
                logger.info("✓ 使用 FP32 全精度")
        # ===========================
        
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            if self.is_main_process:
                logger.info("✓ 使用 8-bit AdamW")
        except ImportError:
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
        
        # ===== KD Warmup 配置 =====
        self.kd_warmup_steps = args.kd_warmup_steps
        self.initial_kd_weight = args.kd_weight
        if self.is_main_process and self.kd_warmup_steps > 0:
            logger.info(f"✓ KD Warmup: {self.kd_warmup_steps} steps (0.0 → {args.kd_weight})")
        # ===========================
        
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
            logger.info("训练器初始化完成（Multi-GPU with RDO - FIXED）")
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
            
            # ===== 【新增】检查模型状态 =====
            model_to_check = self.model.module if hasattr(self.model, 'module') else self.model
            logger.info(f"\n🔍 模型状态检查:")
            logger.info(f"  model.training: {model_to_check.training}")
            if hasattr(model_to_check, 'teacher_model'):
                logger.info(f"  teacher_model exists: {model_to_check.teacher_model is not None}")
                if model_to_check.teacher_model is not None:
                    teacher_frozen = not any(p.requires_grad for p in model_to_check.teacher_model.parameters())
                    logger.info(f"  teacher frozen: {teacher_frozen}")
            logger.info(f"  visual_ib exists: {hasattr(model_to_check, 'visual_ib')}")
            logger.info(f"  multi_layer_distill exists: {hasattr(model_to_check, 'multi_layer_distill') and model_to_check.multi_layer_distill is not None}")
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
        """训练一个epoch - 修复版"""
        self.model.train()
        
        # ===== 【修复2】使用原始损失累积 =====
        total_loss_sum = 0
        num_valid_batches = 0  # 记录实际处理的批次数
        
        loss_components_sum = {
            'task_loss': 0,
            'kl_loss': 0,
            'feature_loss': 0,  # ← 注意：是 feature_loss 不是 feat_loss
            'ib_loss': 0,
            'layer_0_loss': 0,
            'layer_1_loss': 0,
            'layer_2_loss': 0,
            # 额外的细节项
            'ib_kl': 0,
            'ib_recon': 0,
            'ib_beta': 0,
            'ib_weight': 0,
            'kd_weight': 0,
            'feat_weight': 0,
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
                # ===== 处理 images list =====
                images = batch.pop('images', None)
                
                # 其他字段移到设备
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # 处理 images
                if images is not None:
                    if isinstance(images, list):
                        # 过滤并处理 list
                        processed_images = []
                        valid_count = 0
                        
                        for img in images:
                            if img is not None and torch.is_tensor(img):
                                # 兜底检查：确保是 3 通道
                                if img.dim() == 3:
                                    if img.shape[0] == 4:
                                        img = img[:3, :, :]
                                        if self.is_main_process and valid_count == 0:
                                            logger.warning(f"Step {step}: 发现4通道图片，已裁剪")
                                    elif img.shape[0] != 3:
                                        if self.is_main_process:
                                            logger.error(f"Step {step}: 异常通道数 {img.shape[0]}")
                                        raise ValueError(f"Invalid channel count: {img.shape[0]}")
                                
                                processed_images.append(img.to(self.device))
                                valid_count += 1
                            else:
                                processed_images.append(None)
                        
                        # ===== 【DDP 修复】不要跳过 batch，即使没有有效图片 =====
                        # 如果没有有效图片，创建一个 dummy tensor 确保所有 rank 都执行
                        if valid_count == 0:
                            if self.is_main_process:
                                logger.warning(f"Step {step}: 没有有效图片，使用 dummy tensor")
                            # 创建 dummy image tensor（3, 224, 224）
                            dummy_img = torch.zeros(3, 224, 224, device=self.device)
                            processed_images = [dummy_img] * len(images)
                            valid_count = 1
                        
                        # 根据情况转换格式
                        if all(img is not None for img in processed_images):
                            images = torch.stack(processed_images)
                        else:
                            images = processed_images
                        
                    elif torch.is_tensor(images):
                        if images.dim() == 4 and images.shape[1] == 4:
                            images = images[:, :3, :, :]
                            if self.is_main_process:
                                logger.warning(f"Step {step}: 发现4通道批次，已裁剪")
                        elif images.dim() == 4 and images.shape[1] != 3:
                            if self.is_main_process:
                                logger.error(f"Step {step}: 异常通道数批次 {images.shape[1]}")
                            raise ValueError(f"Invalid channel count: {images.shape[1]}")
                        
                        images = images.to(self.device)
                
                batch['images'] = images
                
                # ===== KD Warmup 逻辑 =====
                if self.kd_warmup_steps > 0 and self.global_step < self.kd_warmup_steps:
                    # 线性 warmup：从 0.0 到 initial_kd_weight
                    current_kd_weight = self.initial_kd_weight * (self.global_step / self.kd_warmup_steps)
                    # 动态设置模型的 KD 权重
                    if hasattr(self.model, 'module'):  # DDP
                        self.model.module.log_kd_weight.data.fill_(torch.log(torch.tensor(current_kd_weight + 1e-8)))
                    else:
                        self.model.log_kd_weight.data.fill_(torch.log(torch.tensor(current_kd_weight + 1e-8)))
                # =============================
                
                # ===== 前向传播（使用 autocast）=====
                if self.use_amp:
                    with autocast('cuda', dtype=self.dtype):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                            images=batch['images'],
                            return_dict=True
                        )
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        images=batch['images'],
                        return_dict=True
                    )
                
                # ===== 【修复2】在缩放前记录原始损失 =====
                loss = outputs.loss
                original_loss_value = loss.detach().float().item()  # ← 避免 AMP 下的精度问题
                
                # ===== 【修复1】使用 safe_get_loss_info 读取分项损失 =====
                loss_info = safe_get_loss_info(outputs)
                
                # ===== 【调试】第一个batch打印详细信息 =====
                if step == 0 and self.is_main_process:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"DEBUG: 第一个batch的损失详情")
                    logger.info(f"{'='*60}")
                    logger.info(f"outputs.loss = {original_loss_value:.4f}")
                    logger.info(f"loss_info 内容:")
                    for k, v in sorted(loss_info.items()):
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info(f"模型训练状态: {self.model.training}")
                    logger.info(f"{'='*60}\n")
                
                # ===== RDO 优化（修复 DDP 问题）=====
                # 【修复】去掉条件判断，确保每个 batch 都执行，避免 DDP unused parameters 错误
                task_loss = loss_info.get('task_loss', 0)
                ib_loss = loss_info.get('ib_loss', 0)
                
                # 总是执行 RDO，使用默认值避免除零
                if self.is_main_process:
                    # 使用 max(loss, 1e-8) 确保非零值
                    rate = loss_info.get('ib_kl', max(ib_loss, 1e-8))
                    distortion = max(task_loss, 1e-8)
                    
                    optimal_beta = self.rdo.step(
                        torch.tensor(distortion).to(self.device),
                        torch.tensor(rate).to(self.device)
                    )
                    
                    rdo_stats['beta_values'].append(optimal_beta)
                    rdo_stats['rate_values'].append(rate)
                    rdo_stats['distortion_values'].append(distortion)
                else:
                    optimal_beta = 1.0
                
                # 总是同步 beta（多 GPU）
                if self.world_size > 1:
                    beta_tensor = torch.tensor(optimal_beta, device=self.device)
                    dist.broadcast(beta_tensor, src=0)
                    optimal_beta = beta_tensor.item()
                
                # 总是更新 visual_ib.log_beta，确保参数被使用（避免 DDP unused parameters）
                model_to_update = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(model_to_update, 'visual_ib') and hasattr(model_to_update.visual_ib, 'log_beta'):
                    with torch.no_grad():
                        model_to_update.visual_ib.log_beta.data = torch.tensor(
                            np.log(max(optimal_beta, 1e-8)),  # 确保非零
                            dtype=torch.float32,
                            device=self.device
                        )
                
                # ===== 梯度累积 =====
                loss = loss / self.args.gradient_accumulation_steps
                
                # ===== 反向传播（使用 GradScaler）=====
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # ===== 【修复2】累积原始损失（未缩放的）=====
                total_loss_sum += original_loss_value
                num_valid_batches += 1
                
                # ===== 【修复1】累积所有分项损失 =====
                for key in loss_components_sum.keys():
                    if key in loss_info:
                        loss_components_sum[key] += loss_info[key]
                
                # ===== 参数更新 =====
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.args.max_grad_norm > 0:
                        if self.scaler.is_enabled():
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.max_grad_norm
                        )
                    
                    # 优化器步进
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # 更新进度条
                    if self.is_main_process:
                        current_lr = self.scheduler.get_last_lr()[0]
                        current_beta = rdo_stats['beta_values'][-1] if rdo_stats and rdo_stats['beta_values'] else 1.0
                        progress_bar.set_postfix({
                            'loss': f'{original_loss_value:.4f}',  # 使用原始损失
                            'lr': f'{current_lr:.2e}',
                            'β': f'{current_beta:.3f}'
                        })
            
            except Exception as e:
                # ===== 【DDP 修复】不要单独 continue，让异常向上传播 =====
                # 如果一个 rank 异常，应该让所有 rank 都知道并一起停止
                if self.is_main_process:
                    logger.error(f"Error in training step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 不要 continue！让异常传播，或者做全局同步的处理
                # 这里我们选择让异常传播，终止训练
                raise
        
        # ===== 【修复3】DDP 同步所有GPU的损失 =====
        if self.world_size > 1:
            total_loss_tensor = torch.tensor(total_loss_sum, device=self.device)
            num_batches_tensor = torch.tensor(num_valid_batches, device=self.device)
            
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            
            total_loss_sum = total_loss_tensor.item()
            num_valid_batches = num_batches_tensor.item()
            
            # ===== 【关键】对每个损失分项也做 all_reduce =====
            for key in loss_components_sum.keys():
                loss_tensor = torch.tensor(loss_components_sum[key], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_components_sum[key] = loss_tensor.item()
        
        # ===== 【修复2】使用实际处理的批次数计算平均 =====
        if num_valid_batches > 0:
            metrics = {
                'total_loss': total_loss_sum / num_valid_batches,
                **{k: v / num_valid_batches for k, v in loss_components_sum.items()}
            }
        else:
            metrics = {
                'total_loss': 0,
                **{k: 0 for k in loss_components_sum}
            }
        
        # 添加 RDO 统计
        if self.is_main_process and rdo_stats and rdo_stats['beta_values']:
            metrics['avg_beta'] = np.mean(rdo_stats['beta_values'])
            metrics['std_beta'] = np.std(rdo_stats['beta_values'])
            metrics['avg_rate'] = np.mean(rdo_stats['rate_values'])
            metrics['avg_distortion'] = np.mean(rdo_stats['distortion_values'])
            
            if rdo_stats['compression_rates']:
                metrics['avg_compression_rate'] = np.mean(rdo_stats['compression_rates'])
            
            self._log_rdo_stats(rdo_stats)
        
        return metrics
    
    def _validate(self):
        """验证 - 修复版"""
        self.model.eval()
        
        # ===== 【修复2】使用原始损失累积 =====
        total_loss_sum = 0
        num_valid_batches = 0
        
        loss_components_sum = {
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
                
                # ===== 前向传播（使用 autocast）=====
                if self.use_amp:
                    with autocast('cuda', dtype=self.dtype):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                            images=batch.get('images'),
                            image_sizes=batch.get('image_sizes'),
                            return_dict=True
                        )
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        images=batch.get('images'),
                        image_sizes=batch.get('image_sizes'),
                        return_dict=True
                    )
                
                loss = outputs.loss
                total_loss_sum += loss.detach().float().item()  # ← 避免 AMP 下的精度问题
                num_valid_batches += 1
                
                # ===== 【修复1】使用 safe_get_loss_info =====
                loss_info = safe_get_loss_info(outputs)
                
                for key in loss_components_sum.keys():
                    if key in loss_info:
                        loss_components_sum[key] += loss_info[key]
        
        # ===== 【修复3】DDP 同步 =====
        if self.world_size > 1:
            total_loss_tensor = torch.tensor(total_loss_sum, device=self.device)
            num_batches_tensor = torch.tensor(num_valid_batches, device=self.device)
            
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            
            total_loss_sum = total_loss_tensor.item()
            num_valid_batches = num_batches_tensor.item()
            
            # 对每个分项也做 all_reduce
            for key in loss_components_sum.keys():
                loss_tensor = torch.tensor(loss_components_sum[key], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_components_sum[key] = loss_tensor.item()
        
        # 计算平均
        if num_valid_batches > 0:
            metrics = {
                'total_loss': total_loss_sum / num_valid_batches,
                **{k: v / num_valid_batches for k, v in loss_components_sum.items()}
            }
        else:
            metrics = {
                'total_loss': 0,
                **{k: 0 for k in loss_components_sum}
            }
        
        return metrics
    
    def _log_metrics(self, epoch, metrics, phase='train'):
        """记录指标（只在主进程）- 修复版"""
        if not self.is_main_process:
            return
        
        log_str = f"\n[{phase.upper()}] Epoch {epoch + 1}\n"
        log_str += f"  Total Loss: {metrics.get('total_loss', 0):.4f}\n"
        log_str += f"  Task Loss: {metrics.get('task_loss', 0):.4f}\n"
        log_str += f"  KL Loss: {metrics.get('kl_loss', 0):.4f}\n"
        log_str += f"  Feature Loss: {metrics.get('feature_loss', 0):.4f}\n"  # ← 注意键名
        log_str += f"  IB Loss: {metrics.get('ib_loss', 0):.4f}\n"
        
        # 层级蒸馏损失
        if any(k in metrics for k in ['layer_0_loss', 'layer_1_loss', 'layer_2_loss']):
            log_str += f"  Layer Losses: "
            log_str += f"{metrics.get('layer_0_loss', 0):.4f} / "
            log_str += f"{metrics.get('layer_1_loss', 0):.4f} / "
            log_str += f"{metrics.get('layer_2_loss', 0):.4f}\n"
        
        # IB 细节
        if 'ib_kl' in metrics and metrics.get('ib_kl', 0) > 0:
            log_str += f"\n  IB Details:\n"
            log_str += f"    KL: {metrics.get('ib_kl', 0):.4f}\n"
            log_str += f"    Recon: {metrics.get('ib_recon', 0):.4f}\n"
            log_str += f"    Beta: {metrics.get('ib_beta', 1.0):.4f}\n"
        
        # 损失权重
        if 'kd_weight' in metrics and metrics.get('kd_weight', 0) > 0:
            log_str += f"\n  Loss Weights:\n"
            log_str += f"    KD: {metrics.get('kd_weight', 1.0):.4f}\n"
            log_str += f"    Feature: {metrics.get('feat_weight', 0.5):.4f}\n"
            log_str += f"    IB: {metrics.get('ib_weight', 0.1):.4f}\n"
        
        # RDO 指标
        if 'avg_beta' in metrics:
            log_str += f"\n  🔧 RDO Stats:\n"
            log_str += f"    Avg β: {metrics['avg_beta']:.4f} ± {metrics.get('std_beta', 0):.4f}\n"
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
    parser = argparse.ArgumentParser(description="CIBD Training Script V2 with RDO (Multi-GPU) - FIXED")
    
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
    
    # 混合精度参数
    parser.add_argument("--precision", type=str, default="bf16",
                       choices=["fp32", "fp16", "bf16"],
                       help="训练精度: fp32 (全精度), fp16 (FP16混合精度), bf16 (BF16混合精度, 推荐用于H100)")
    
    # RDO 参数
    parser.add_argument("--initial_beta", type=float, default=1.0)
    parser.add_argument("--target_compression_rate", type=float, default=0.5)
    parser.add_argument("--beta_adaptation_rate", type=float, default=0.01)
    parser.add_argument("--use_adaptive_compression", action="store_true")
    parser.add_argument("--initial_compression_rate", type=float, default=0.1)
    
    # ===== 知识蒸馏参数 =====
    parser.add_argument("--kd_temperature", type=float, default=4.0,
                       help="知识蒸馏温度系数 (2-5, 推荐 4.0)")
    parser.add_argument("--kd_weight", type=float, default=0.5,
                       help="KD 损失权重 (0.1-0.7)")
    parser.add_argument("--feat_weight", type=float, default=0.2,
                       help="特征蒸馏权重")
    parser.add_argument("--ib_weight", type=float, default=0.1,
                       help="信息瓶颈权重")
    parser.add_argument("--kd_warmup_steps", type=int, default=500,
                       help="KD warmup 步数 (0 表示不使用 warmup)")
    
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
        logger.info("CIBD Training V2 with RDO (Multi-GPU) - PyTorch 2.0+")
        logger.info("=" * 60)
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"World Size: {world_size}")
        logger.info(f"Rank: {rank}")
        logger.info(f"Local Rank: {local_rank}")
        logger.info(f"Device: {device}")
        logger.info(f"Backend: {args.distributed_backend}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Global batch size: {args.batch_size * world_size}")
        logger.info(f"\n⚡ 混合精度配置:")
        logger.info(f"  Precision: {args.precision.upper()}")
        if args.precision == 'bf16':
            logger.info(f"  BF16 支持: {torch.cuda.is_bf16_supported()}")
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
    
    # ===== 创建 CIBD 学生模型（使用命令行参数）=====
    student_model = LlavaCIBDModelV2(
        student_config, 
        teacher_model,
        temperature=args.kd_temperature,  # 从命令行传入
        kd_weight=args.kd_weight,         # 从命令行传入
        feat_weight=args.feat_weight,     # 从命令行传入
        ib_weight=args.ib_weight,         # 从命令行传入
    )
    
    if is_main_process:
        logger.info("✓ CIBD 蒸馏配置:")
        logger.info(f"  Temperature: {args.kd_temperature} (KD 温度)")
        logger.info(f"  KD Weight: {args.kd_weight}")
        logger.info(f"  Feat Weight: {args.feat_weight}")
        logger.info(f"  IB Weight: {args.ib_weight}")
        if args.kd_warmup_steps > 0:
            logger.info(f"  KD Warmup Steps: {args.kd_warmup_steps}")
    
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
    
    # 将模型移到设备并保持 FP32 精度
    # 注意：我们不强制转换模型精度，而是依赖 autocast 自动混合精度
    # 这样 LayerNorm、Embedding 等会自动保持 FP32，更稳定
    student_model.to(device)
    teacher_model.to(device)
    student_model = student_model.float()
    teacher_model = teacher_model.float()
    
    if is_main_process:
        if args.precision == 'bf16':
            logger.info("✓ 模型参数: FP32, 前向传播自动使用 BF16 (推荐)")
        elif args.precision == 'fp16':
            logger.info("✓ 模型参数: FP32, 前向传播自动使用 FP16 (推荐)")
        else:
            logger.info("✓ 模型参数: FP32")
    
    # ===== 启用梯度检查点（节省显存）=====
    if hasattr(student_model.model, 'gradient_checkpointing_enable'):
        # 使用推荐的 use_reentrant=False（PyTorch 2.0+ 新方式）
        try:
            student_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if is_main_process:
                logger.info("✓ 启用梯度检查点（use_reentrant=False）")
        except TypeError:
            # 旧版本不支持参数，使用默认方式
            student_model.model.gradient_checkpointing_enable()
            if is_main_process:
                logger.info("✓ 启用梯度检查点（默认模式）")
    
    # 包装模型（多GPU）
    if world_size > 1:
        if args.distributed_backend == "ddp":
            # ===== 修复 DDP 问题 =====
            # 1. 确保所有参数都需要梯度或者被冻结
            # 2. 使用静态图模式（PyTorch 2.0+）
            student_model = DDP(
                student_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,  # ← 改为 False，使用静态图
                broadcast_buffers=True,
                gradient_as_bucket_view=True   # 优化显存
            )
            
            # ===== 启用静态图模式（PyTorch 2.0+）=====
            # 这告诉 DDP：模型的计算图是固定的
            if hasattr(student_model, '_set_static_graph'):
                student_model._set_static_graph()
                if is_main_process:
                    logger.info("✓ 使用 DistributedDataParallel (静态图模式)")
            else:
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