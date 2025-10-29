"""
CM-IBQ 分布式训练器 - 改进版
支持单机多卡、多机多卡训练，包含对齐损失
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import deepspeed
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, Any
import wandb
from tqdm import tqdm
import time
from datetime import timedelta
import json

class DistributedCMIBQTrainer:
    """
    分布式训练器，支持多种并行策略
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: Dict[str, Any],
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        compute_metrics=None
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        
        # 训练阶段
        self.stage = args.get('stage', 1)
        
        # 初始化分布式环境
        self._setup_distributed()
        
        # 设置设备
        self.device = torch.device(f'cuda:{self.local_rank}')
        torch.cuda.set_device(self.local_rank)
        
        # 模型设置
        self.model = model.to(self.device)
        
        # 选择并行策略
        if args.get('use_deepspeed', False):
            self._setup_deepspeed()
        else:
            self._setup_ddp()
        
        # 数据加载器
        self._setup_dataloaders(train_dataset, eval_dataset)
        
        # 优化器和调度器
        self._setup_optimization()
        
        # 混合精度训练
        self.use_amp = args.get('use_amp', True)
        if self.use_amp and not args.get('use_deepspeed', False):
            self.scaler = GradScaler()
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        
        # 日志设置
        if self.is_main_process:
            self._setup_logging()
    
    def _setup_distributed(self):
        """
        初始化分布式环境
        """
        # 从环境变量获取分布式参数
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        
        if self.world_size > 1:
            # 初始化进程组
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank,
                    timeout=timedelta(minutes=60)
                )
            
            # 设置NCCL参数以优化通信
            os.environ['NCCL_DEBUG'] = 'WARN'
            os.environ['NCCL_TREE_THRESHOLD'] = '0'
            
            # 对于大模型，优化NCCL设置
            if self.args.get('model_size', '7b') in ['13b', '34b', '70b']:
                os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
                os.environ['NCCL_SOCKET_NTHREADS'] = '8'
        
        self.is_main_process = self.rank == 0
        
        if self.is_main_process:
            print(f"Initialized distributed training: {self.world_size} GPUs")
            print(f"Backend: {'DeepSpeed' if self.args.get('use_deepspeed') else 'DDP'}")
            print(f"Training Stage: {self.stage}")
    
    def _setup_ddp(self):
        """
        设置DistributedDataParallel
        """
        # DDP配置
        ddp_config = {
            'device_ids': [self.local_rank],
            'output_device': self.local_rank,
            'find_unused_parameters': self.args.get('find_unused_parameters', True),
            'gradient_as_bucket_view': True,
            'broadcast_buffers': False,
            'static_graph': self.args.get('static_graph', False)
        }
        
        # 如果模型很大，使用梯度分片
        if self.args.get('use_gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
        
        # 包装模型
        self.model = DDP(self.model, **ddp_config)
        
        if self.is_main_process:
            print(f"Model wrapped with DDP on {self.world_size} GPUs")
    
    def _setup_deepspeed(self):
        """
        设置DeepSpeed (支持ZeRO优化)
        """
        # DeepSpeed配置
        ds_config = self._get_deepspeed_config()
        
        # 初始化DeepSpeed
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self._create_optimizer(),
            config=ds_config,
            dist_init_required=False
        )
        
        if self.is_main_process:
            print(f"DeepSpeed initialized with ZeRO stage {ds_config['zero_optimization']['stage']}")
    
    def _get_deepspeed_config(self):
        """
        获取DeepSpeed配置
        """
        # 根据模型大小选择ZeRO阶段
        model_size = self.args.get('model_size', '7b')
        
        if model_size in ['70b', '34b']:
            zero_stage = 3  # ZeRO-3 for very large models
        elif model_size == '13b':
            zero_stage = 2  # ZeRO-2 for 13B
        else:
            zero_stage = 2  # ZeRO-2 for 7B
        
        config = {
            "train_batch_size": self.args['batch_size'] * self.world_size,
            "train_micro_batch_size_per_gpu": self.args['batch_size'],
            "gradient_accumulation_steps": self.args.get('gradient_accumulation_steps', 1),
            "steps_per_print": 10,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.args['learning_rate'],
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": self.args.get('weight_decay', 0.01)
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.args['learning_rate'],
                    "warmup_num_steps": self.args.get('warmup_steps', 500),
                    "total_num_steps": self.args.get('max_steps', 10000)
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if zero_stage == 3 else "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu" if zero_stage == 3 else "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "gradient_clipping": self.args.get('max_grad_norm', 1.0),
            "fp16": {
                "enabled": self.args.get('fp16', True),
                "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "consecutive_hysteresis": False,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": self.args.get('bf16', False)
            },
            "activation_checkpointing": {
                "partition_activations": self.args.get('gradient_checkpointing', False),
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            }
        }
        
        # 保存配置
        if self.is_main_process:
            config_path = os.path.join(self.args['output_dir'], 'ds_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"DeepSpeed config saved to {config_path}")
        
        return config
    
    def _setup_dataloaders(self, train_dataset, eval_dataset):
        """
        设置分布式数据加载器
        """
        # 训练数据加载器
        if train_dataset:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
                seed=self.args.get('seed', 42)
            )
            
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args['batch_size'],
                sampler=train_sampler,
                num_workers=self.args.get('num_workers', 4),
                pin_memory=True,
                persistent_workers=True if self.args.get('num_workers', 4) > 0 else False,
                prefetch_factor=2 if self.args.get('num_workers', 4) > 0 else None
            )
            
            self.train_sampler = train_sampler
        
        # 验证数据加载器
        if eval_dataset:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
            
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.args.get('eval_batch_size', self.args['batch_size']),
                sampler=eval_sampler,
                num_workers=self.args.get('num_workers', 4),
                pin_memory=True
            )
            
            self.eval_sampler = eval_sampler
    
    def _setup_optimization(self):
        """
        设置优化器和学习率调度器
        """
        if not self.args.get('use_deepspeed', False):
            self.optimizer = self._create_optimizer()
            
            # 学习率调度器
            num_training_steps = len(self.train_dataloader) * self.args['num_epochs']
            num_warmup_steps = self.args.get('warmup_steps', int(0.1 * num_training_steps))
            
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
    
    def _create_optimizer(self):
        """
        创建优化器 - 改进版：更细粒度的参数分组
        """
        # 参数分组
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        
        # 根据训练阶段和模块类型设置不同的学习率
        param_groups = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 确定weight decay
            if any(nd in name for nd in no_decay):
                weight_decay = 0.0
            else:
                weight_decay = self.args.get('weight_decay', 0.01)
            
            # 根据模块类型确定学习率倍率
            lr_scale = 1.0
            
            if self.stage == 1:
                # Stage 1: 主要训练量化模块
                if 'quantization_modules' in name:
                    if 'importance' in name or 'bit_allocator' in name:
                        lr_scale = 1.0  # 重要性估计和比特分配：正常学习率
                    elif 'quantizer' in name:
                        lr_scale = 0.5  # 量化器：较低学习率
                    elif 'ib' in name.lower():
                        lr_scale = 0.8  # IB层：中等学习率
                    else:
                        lr_scale = 0.3
                else:
                    lr_scale = 0.1  # 其他参数：很低学习率
                    
            elif self.stage == 2:
                # Stage 2: 主要训练LoRA、对齐模块和微调量化
                if 'lora' in name.lower():
                    lr_scale = 1.0  # LoRA：正常学习率
                elif 'proj_head' in name:  # 对齐投影头
                    lr_scale = 0.8  # 较高学习率
                elif 'alignment_temperature' in name:  # 温度参数
                    lr_scale = 0.1  # 低学习率，避免剧烈变化
                elif 'quantization_modules' in name:
                    lr_scale = 0.3  # 量化模块：微调
                else:
                    lr_scale = 0.1  # 其他：很低学习率
            
            param_groups.append({
                'params': param,
                'lr': self.args['learning_rate'] * lr_scale,
                'weight_decay': weight_decay,
                'param_name': name  # 保存参数名用于调试
            })
        
        # 创建优化器
        optimizer = AdamW(
            param_groups,
            lr=self.args['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 打印参数分组信息
        if self.is_main_process:
            print(f"\nOptimizer parameter groups (Stage {self.stage}):")
            lr_groups = {}
            for group in param_groups:
                lr = group['lr']
                if lr not in lr_groups:
                    lr_groups[lr] = 0
                lr_groups[lr] += 1
            for lr, count in sorted(lr_groups.items(), reverse=True):
                print(f"  LR {lr:.2e}: {count} parameters")
        
        return optimizer
    
    def _setup_logging(self):
        """
        设置日志记录
        """
        if self.args.get('use_wandb', False):
            wandb.init(
                project='cm-ibq-llava',
                name=f"{self.args.get('run_name', 'cmibq')}_{self.world_size}gpu_stage{self.stage}",
                config=self.args
            )
    
    def train(self):
        """
        主训练循环
        """
        if self.is_main_process:
            print(f"Starting training for {self.args['num_epochs']} epochs")
            print(f"Total training steps: {len(self.train_dataloader) * self.args['num_epochs']}")
            print(f"Stage {self.stage}: {'Bottleneck Shaping' if self.stage == 1 else 'Task-Aware Optimization with Alignment'}")
        
        for epoch in range(self.args['num_epochs']):
            self.current_epoch = epoch
            
            # 设置epoch for sampler
            if hasattr(self, 'train_sampler'):
                self.train_sampler.set_epoch(epoch)
            
            # 训练一个epoch
            train_loss, train_metrics = self._train_epoch()
            
            # 验证
            if hasattr(self, 'eval_dataloader') and (epoch + 1) % self.args.get('eval_epochs', 1) == 0:
                eval_loss, eval_metrics = self._evaluate()
                
                if self.is_main_process:
                    log_str = f"Epoch {epoch+1}/{self.args['num_epochs']} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}"
                    if self.stage == 2 and 'alignment_loss' in train_metrics:
                        log_str += f", Align Loss: {train_metrics['alignment_loss']:.4f}"
                    print(log_str)
                    
                    # 保存最佳模型
                    if eval_loss < self.best_metric:
                        self.best_metric = eval_loss
                        self.save_checkpoint('best_model')
            else:
                if self.is_main_process:
                    log_str = f"Epoch {epoch+1}/{self.args['num_epochs']} - Train Loss: {train_loss:.4f}"
                    if self.stage == 2 and 'alignment_loss' in train_metrics:
                        log_str += f", Align Loss: {train_metrics['alignment_loss']:.4f}"
                    print(log_str)
            
            # 定期保存检查点
            if (epoch + 1) % self.args.get('save_epochs', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
        
        # 训练结束，保存最终模型
        if self.is_main_process:
            self.save_checkpoint('final_model')
            print("Training completed!")
    
    def _train_epoch(self):
        """
        训练一个epoch - 改进版：添加量化统计和对齐损失
        """
        self.model.train()
        total_loss = 0
        alignment_loss_sum = 0
        alignment_acc_v2t_sum = 0
        alignment_acc_t2v_sum = 0
        quantization_stats = {}
        num_batches = 0
        
        # 进度条（只在主进程显示）
        if self.is_main_process:
            pbar = tqdm(total=len(self.train_dataloader), 
                       desc=f"Epoch {self.current_epoch+1} Stage {self.stage}")
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 移动数据到设备
            batch = self._prepare_batch(batch)
            
            # 混合精度训练
            if self.use_amp and not self.args.get('use_deepspeed', False):
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                    
                    # Stage 2: 提取对齐损失和准确率
                    if self.stage == 2 and hasattr(outputs, 'aux_losses'):
                        if 'alignment_loss' in outputs.aux_losses:
                            alignment_loss_sum += outputs.aux_losses['alignment_loss'].item()
                
                # 梯度累积
                loss = loss / self.args.get('gradient_accumulation_steps', 1)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度累积步骤
                if (batch_idx + 1) % self.args.get('gradient_accumulation_steps', 1) == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.get('max_grad_norm', 1.0)
                    )
                    
                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # 收集统计
                    model_to_check = self.model.module if hasattr(self.model, 'module') else self.model
                    if hasattr(model_to_check, 'get_quantization_stats'):
                        stats = model_to_check.get_quantization_stats()
                        for k, v in stats.items():
                            if k not in quantization_stats:
                                quantization_stats[k] = []
                            quantization_stats[k].append(v)
                        
                        # Stage 2: 收集对齐准确率
                        if self.stage == 2:
                            if 'acc_v2t' in stats:
                                alignment_acc_v2t_sum += stats['acc_v2t']
                            if 'acc_t2v' in stats:
                                alignment_acc_t2v_sum += stats['acc_t2v']
            
            elif self.args.get('use_deepspeed', False):
                # DeepSpeed处理
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                # Stage 2: 提取对齐损失
                if self.stage == 2 and hasattr(outputs, 'aux_losses'):
                    if 'alignment_loss' in outputs.aux_losses:
                        alignment_loss_sum += outputs.aux_losses['alignment_loss'].item()
                
                self.model.backward(loss)
                self.model.step()
                grad_norm = self.model.get_global_grad_norm()
                
                # 收集统计
                if hasattr(self.model.module, 'get_quantization_stats'):
                    stats = self.model.module.get_quantization_stats()
                    for k, v in stats.items():
                        if k not in quantization_stats:
                            quantization_stats[k] = []
                        quantization_stats[k].append(v)
                    
                    # Stage 2: 收集对齐准确率
                    if self.stage == 2:
                        if 'acc_v2t' in stats:
                            alignment_acc_v2t_sum += stats['acc_v2t']
                        if 'acc_t2v' in stats:
                            alignment_acc_t2v_sum += stats['acc_t2v']
            
            else:
                # 标准训练
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                # Stage 2: 提取对齐损失
                if self.stage == 2 and hasattr(outputs, 'aux_losses'):
                    if 'alignment_loss' in outputs.aux_losses:
                        alignment_loss_sum += outputs.aux_losses['alignment_loss'].item()
                
                loss = loss / self.args.get('gradient_accumulation_steps', 1)
                loss.backward()
                
                if (batch_idx + 1) % self.args.get('gradient_accumulation_steps', 1) == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.get('max_grad_norm', 1.0)
                    )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # 收集统计
                    model_to_check = self.model.module if hasattr(self.model, 'module') else self.model
                    if hasattr(model_to_check, 'get_quantization_stats'):
                        stats = model_to_check.get_quantization_stats()
                        for k, v in stats.items():
                            if k not in quantization_stats:
                                quantization_stats[k] = []
                            quantization_stats[k].append(v)
                        
                        # Stage 2: 收集对齐准确率
                        if self.stage == 2:
                            if 'acc_v2t' in stats:
                                alignment_acc_v2t_sum += stats['acc_v2t']
                            if 'acc_t2v' in stats:
                                alignment_acc_t2v_sum += stats['acc_t2v']
            
            # 累积损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            if self.is_main_process:
                pbar.update(1)
                if batch_idx % self.args.get('logging_steps', 10) == 0:
                    avg_loss = total_loss / num_batches
                    
                    # 计算统计平均值
                    avg_stats = {}
                    for k, v_list in quantization_stats.items():
                        if v_list:
                            avg_stats[k] = sum(v_list) / len(v_list)
                    
                    # 更新进度条显示
                    postfix = {
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{self.get_lr():.2e}'
                    }
                    
                    # Stage 2: 添加对齐相关显示
                    if self.stage == 2:
                        if alignment_loss_sum > 0:
                            avg_align_loss = alignment_loss_sum / num_batches
                            postfix['align'] = f'{avg_align_loss:.4f}'
                        
                        if alignment_acc_v2t_sum > 0:
                            avg_acc_v2t = alignment_acc_v2t_sum / num_batches
                            postfix['acc_v2t'] = f'{avg_acc_v2t:.2%}'
                        
                        if alignment_acc_t2v_sum > 0:
                            avg_acc_t2v = alignment_acc_t2v_sum / num_batches
                            postfix['acc_t2v'] = f'{avg_acc_t2v:.2%}'
                    
                    # 添加关键量化统计
                    if 'vision_avg_bits' in avg_stats:
                        postfix['v_bits'] = f"{avg_stats['vision_avg_bits']:.2f}"
                    if 'projector_avg_bits' in avg_stats:
                        postfix['p_bits'] = f"{avg_stats['projector_avg_bits']:.2f}"
                    
                    pbar.set_postfix(postfix)
                    
                    # 记录到wandb
                    if self.args.get('use_wandb', False):
                        log_dict = {
                            'train/loss': avg_loss,
                            'train/learning_rate': self.get_lr(),
                            'train/global_step': self.global_step,
                            'train/epoch': self.current_epoch + batch_idx / len(self.train_dataloader)
                        }
                        
                        # Stage 2: 添加对齐相关指标
                        if self.stage == 2:
                            if alignment_loss_sum > 0:
                                log_dict['train/alignment_loss'] = alignment_loss_sum / num_batches
                            if alignment_acc_v2t_sum > 0:
                                log_dict['alignment/acc_v2t'] = alignment_acc_v2t_sum / num_batches
                            if alignment_acc_t2v_sum > 0:
                                log_dict['alignment/acc_t2v'] = alignment_acc_t2v_sum / num_batches
                            if 'temperature' in avg_stats:
                                log_dict['alignment/temperature'] = avg_stats['temperature']
                        
                        # 添加量化统计
                        for k, v in avg_stats.items():
                            if k not in ['acc_v2t', 'acc_t2v', 'temperature']:
                                log_dict[f'quantization/{k}'] = v
                        
                        # 如果有梯度范数
                        if 'grad_norm' in locals():
                            log_dict['train/grad_norm'] = grad_norm
                        
                        wandb.log(log_dict)
        
        if self.is_main_process:
            pbar.close()
        
        # 同步所有进程的损失
        avg_loss = total_loss / num_batches
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        
        # 准备返回的指标
        metrics = {}
        if self.stage == 2:
            if alignment_loss_sum > 0:
                metrics['alignment_loss'] = alignment_loss_sum / num_batches
            if alignment_acc_v2t_sum > 0:
                metrics['acc_v2t'] = alignment_acc_v2t_sum / num_batches
            if alignment_acc_t2v_sum > 0:
                metrics['acc_t2v'] = alignment_acc_t2v_sum / num_batches
        
        return avg_loss, metrics
    
    def _evaluate(self):
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0
        alignment_loss_sum = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_batch(batch)
                
                if self.use_amp and not self.args.get('use_deepspeed', False):
                    with autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                total_loss += loss.item()
                
                # Stage 2: 收集对齐损失
                if self.stage == 2 and hasattr(outputs, 'aux_losses'):
                    if 'alignment_loss' in outputs.aux_losses:
                        alignment_loss_sum += outputs.aux_losses['alignment_loss'].item()
                
                num_batches += 1
        
        # 同步所有进程的损失
        avg_loss = total_loss / num_batches
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        
        metrics = {}
        if self.stage == 2 and alignment_loss_sum > 0:
            metrics['alignment_loss'] = alignment_loss_sum / num_batches
        
        return avg_loss, metrics
    
    def _prepare_batch(self, batch):
        """
        准备批次数据
        """
        prepared_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared_batch[k] = v.to(self.device)
            else:
                prepared_batch[k] = v
        return prepared_batch
    
    def get_lr(self):
        """
        获取当前学习率
        """
        if self.args.get('use_deepspeed', False):
            return self.model.get_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self, name):
        """
        保存检查点 - 改进版：包含量化配置和对齐统计
        """
        if not self.is_main_process:
            return
        
        checkpoint_dir = os.path.join(self.args['output_dir'], name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.args.get('use_deepspeed', False):
            # DeepSpeed保存
            self.model.save_checkpoint(checkpoint_dir)
            
            # 保存额外信息
            client_state = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_metric': self.best_metric,
                'stage': self.stage,
                'args': self.args
            }
            
            with open(os.path.join(checkpoint_dir, 'client_state.json'), 'w') as f:
                json.dump(client_state, f, indent=2)
        else:
            # 标准保存
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            
            checkpoint = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_metric': self.best_metric,
                'stage': self.stage,
                'args': self.args
            }
            
            # 如果有量化统计，也保存
            if hasattr(model_to_save, 'get_quantization_stats'):
                checkpoint['quantization_stats'] = model_to_save.get_quantization_stats()
            
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        """
        if self.args.get('use_deepspeed', False):
            # DeepSpeed加载
            _, client_sd = self.model.load_checkpoint(checkpoint_path)
            
            # 加载额外信息
            client_state_path = os.path.join(checkpoint_path, 'client_state.json')
            if os.path.exists(client_state_path):
                with open(client_state_path, 'r') as f:
                    client_state = json.load(f)
                self.current_epoch = client_state.get('epoch', 0)
                self.global_step = client_state.get('global_step', 0)
                self.best_metric = client_state.get('best_metric', float('inf'))
        else:
            # 标准加载
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        print(f"Checkpoint loaded from {checkpoint_path}")