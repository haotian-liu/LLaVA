import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class UniversalImportanceEstimation(nn.Module):
    """
    通用的重要性估计，不依赖具体任务
    修复版：正确生成per-channel重要性并进行组级聚合
    """
    def __init__(
        self,
        feature_dim: int = 768,
        num_groups: int = 8,
        window_size: int = 7,
        hidden_dim: int = 256,
        use_cross_modal: bool = True,
        use_global_stats: bool = True,  # 是否融合全局统计
        global_weight: float = 0.3  # 全局统计的权重
    ):
        super().__init__()
        
        assert feature_dim % num_groups == 0, f"feature_dim {feature_dim} must be divisible by num_groups {num_groups}"
        
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.group_size = feature_dim // num_groups
        self.use_cross_modal = use_cross_modal
        self.use_global_stats = use_global_stats
        
        # 1. 基于信息熵的per-channel重要性
        self.entropy_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),  # 输出 [B,T,D]
            nn.Sigmoid()
        )
        
        # 2. 基于稀疏性的per-channel重要性
        self.sparsity_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),  # 输出 [B,T,D]
            nn.Sigmoid()
        )
        
        # 3. 基于局部上下文的per-channel重要性
        self.local_context = nn.Conv1d(
            feature_dim, 
            feature_dim, 
            kernel_size=window_size,
            padding=window_size // 2,
            groups=feature_dim  # Depthwise
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),  # 输出 [B,T,D]
            nn.Sigmoid()
        )
        
        # 4. 跨模态per-channel重要性
        if use_cross_modal:
            self.cross_modal_mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feature_dim),  # 输出 [B,T,D]
                nn.Sigmoid()
            )
        
        # 组合权重（可学习的）
        n_components = 4 if use_cross_modal else 3
        self.combination_weights = nn.Parameter(torch.ones(n_components))
        
        # 组级聚合器 - 现在输入是有意义的per-channel值
        self.group_aggregator = nn.Sequential(
            nn.Linear(self.group_size, self.group_size // 2),
            nn.GELU(),
            nn.Linear(self.group_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 如果使用全局统计，初始化统计buffer和融合权重
        if use_global_stats:
            self.register_buffer('global_importance', torch.ones(1, 1, feature_dim) / feature_dim)
            self.global_fusion_weight = nn.Parameter(torch.tensor(global_weight))
    
    def compute_entropy_importance(self, features: torch.Tensor) -> torch.Tensor:
        """
        基于信息熵的per-channel重要性
        """
        # 归一化
        feat_mean = features.mean(dim=1, keepdim=True)
        feat_std = features.std(dim=1, keepdim=True)
        normalized = (features - feat_mean) / (feat_std + 1e-8)
        
        # 生成per-channel重要性分数
        entropy_importance = self.entropy_mlp(normalized)  # [B, T, D]
        
        return entropy_importance
    
    def compute_sparsity_importance(self, features: torch.Tensor) -> torch.Tensor:
        """
        基于稀疏性的per-channel重要性
        """
        # 计算绝对值（激活强度）
        features_abs = torch.abs(features)
        
        # 生成per-channel重要性分数
        sparsity_importance = self.sparsity_mlp(features_abs)  # [B, T, D]
        
        return sparsity_importance
    
    def compute_context_importance(self, features: torch.Tensor) -> torch.Tensor:
        """
        基于局部上下文的per-channel重要性
        """
        batch_size, seq_len, _ = features.shape
        
        # 转换为conv1d格式: [B, D, T]
        feat_transposed = features.transpose(1, 2)
        
        # 局部上下文
        local_feat = self.local_context(feat_transposed).transpose(1, 2)  # [B, T, D]
        
        # 计算与局部上下文的差异
        diff_features = torch.cat([features, features - local_feat], dim=-1)
        
        # 生成per-channel重要性分数
        context_importance = self.context_mlp(diff_features)  # [B, T, D]
        
        return context_importance
    
    def compute_cross_modal_importance(
        self, 
        features: torch.Tensor,
        other_modal_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        跨模态per-channel重要性
        """
        if not self.use_cross_modal or other_modal_features is None:
            return torch.zeros_like(features)
        
        # 拼接多模态特征
        combined = torch.cat([features, other_modal_features], dim=-1)
        
        # 生成per-channel重要性分数
        cross_modal_importance = self.cross_modal_mlp(combined)  # [B, T, D]
        
        return cross_modal_importance
    
    def aggregate_to_groups(self, channel_importance: torch.Tensor) -> torch.Tensor:
        """
        将channel级重要性聚合到组级
        
        Args:
            channel_importance: [B, T, D] per-channel重要性
        
        Returns:
            group_importance: [B, T, num_groups] 组级重要性
        """
        batch_size, seq_len, _ = channel_importance.shape
        
        # Reshape到组: [B, T, num_groups, group_size]
        importance_grouped = channel_importance.view(
            batch_size, seq_len, self.num_groups, self.group_size
        )
        
        # 通过可学习的聚合器处理每组
        group_importance = []
        for g in range(self.num_groups):
            group_channels = importance_grouped[:, :, g, :]  # [B, T, group_size]
            # 通过MLP聚合
            group_score = self.group_aggregator(group_channels)  # [B, T, 1]
            group_importance.append(group_score)
        
        # 合并所有组
        group_importance = torch.cat(group_importance, dim=-1)  # [B, T, num_groups]
        
        # 归一化
        group_importance = F.softmax(group_importance, dim=-1)
        
        return group_importance
    
    def update_global_stats(self, channel_importance: torch.Tensor):
        """更新全局重要性统计"""
        if self.training and self.use_global_stats:
            with torch.no_grad():
                # 计算batch的平均重要性
                batch_importance = channel_importance.mean(dim=[0, 1], keepdim=True)  # [1, 1, D]
                # EMA更新
                momentum = 0.9
                self.global_importance.mul_(momentum).add_(batch_importance, alpha=1-momentum)
    
    def forward(
        self,
        features: torch.Tensor,
        other_modal_features: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算通用的重要性分数
        
        Args:
            features: [B, T, D] 输入特征
            other_modal_features: [B, T, D] 其他模态特征（可选）
            return_components: 是否返回各个组件的重要性
        """
        # 1. 计算各种per-channel重要性 [B, T, D]
        entropy_imp = self.compute_entropy_importance(features)
        sparsity_imp = self.compute_sparsity_importance(features)
        context_imp = self.compute_context_importance(features)
        cross_modal_imp = self.compute_cross_modal_importance(features, other_modal_features)
        
        # 2. 加权组合per-channel重要性
        weights = F.softmax(self.combination_weights, dim=0)
        
        channel_importance = (
            weights[0] * entropy_imp +
            weights[1] * sparsity_imp +
            weights[2] * context_imp
        )
        
        if self.use_cross_modal:
            channel_importance = channel_importance + weights[3] * cross_modal_imp
        
        # 3. 融合全局统计（如果启用）
        if self.use_global_stats:
            # 更新全局统计
            self.update_global_stats(channel_importance)
            
            # 融合样本级和全局级重要性
            lambda_weight = torch.sigmoid(self.global_fusion_weight)
            channel_importance = (
                lambda_weight * channel_importance + 
                (1 - lambda_weight) * self.global_importance.expand_as(channel_importance)
            )
        
        # 4. 归一化channel重要性
        channel_importance = F.softmax(
            channel_importance.view(-1, self.feature_dim), 
            dim=-1
        ).view_as(features)
        
        # 5. 聚合到组级
        group_importance = self.aggregate_to_groups(channel_importance)
        
        # 准备输出
        aux_outputs = {
            'channel_importance': channel_importance,  # [B, T, D]
            'group_importance': group_importance,      # [B, T, num_groups]
            'combination_weights': weights,
            'mean_importance': channel_importance.mean().item(),
            'std_importance': channel_importance.std().item()
        }
        
        if self.use_global_stats:
            aux_outputs['global_importance'] = self.global_importance.squeeze()
            aux_outputs['fusion_weight'] = lambda_weight.item()
        
        if return_components:
            aux_outputs.update({
                'entropy_importance': entropy_imp,
                'sparsity_importance': sparsity_imp,
                'context_importance': context_imp,
                'cross_modal_importance': cross_modal_imp
            })
        
        return channel_importance, aux_outputs


class HybridImportanceEstimation(nn.Module):
    """
    混合方法：结合样本级动态重要性和全局静态重要性
    """
    def __init__(
        self,
        feature_dim: int = 768,
        num_groups: int = 8,
        momentum: float = 0.9
    ):
        super().__init__()
        
        assert feature_dim % num_groups == 0, f"feature_dim {feature_dim} must be divisible by num_groups {num_groups}"
        
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.group_size = feature_dim // num_groups
        self.momentum = momentum
        
        # 动态重要性估计器
        self.dynamic_estimator = UniversalImportanceEstimation(
            feature_dim=feature_dim,
            num_groups=num_groups,
            use_global_stats=False  # 不使用内部的全局统计
        )
        
        # 全局统计buffer
        self.register_buffer('global_channel_mean', torch.zeros(1, 1, feature_dim))
        self.register_buffer('global_channel_var', torch.ones(1, 1, feature_dim))
        self.register_buffer('global_channel_max', torch.ones(1, 1, feature_dim))
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
    
    def update_global_statistics(self, features: torch.Tensor):
        """更新全局统计量"""
        if self.training:
            with torch.no_grad():
                batch_mean = features.mean(dim=[0, 1], keepdim=True)
                batch_var = features.var(dim=[0, 1], keepdim=True)
                batch_max = features.abs().max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
                
                self.global_channel_mean.mul_(self.momentum).add_(batch_mean, alpha=1-self.momentum)
                self.global_channel_var.mul_(self.momentum).add_(batch_var, alpha=1-self.momentum)
                self.global_channel_max.mul_(self.momentum).add_(batch_max, alpha=1-self.momentum)
    
    def compute_static_importance(self) -> torch.Tensor:
        """基于全局统计计算静态重要性"""
        # 方差大的channel更重要
        var_importance = self.global_channel_var / (self.global_channel_var.mean() + 1e-8)
        
        # 最大值大的channel更重要
        max_importance = self.global_channel_max / (self.global_channel_max.mean() + 1e-8)
        
        # 组合
        static_importance = (var_importance + max_importance) / 2
        static_importance = F.softmax(static_importance.squeeze(0), dim=-1).unsqueeze(0)  # [1, 1, D]
        
        return static_importance
    
    def forward(
        self,
        features: torch.Tensor,
        other_modal_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        混合动态和静态重要性估计
        """
        # 更新全局统计
        self.update_global_statistics(features)
        
        # 1. 动态重要性（样本级）
        dynamic_importance, dynamic_aux = self.dynamic_estimator(
            features, 
            other_modal_features
        )
        
        # 2. 静态重要性（全局级）
        static_importance = self.compute_static_importance()
        
        # 3. 融合
        lambda_weight = torch.sigmoid(self.fusion_weight)
        final_importance = (
            lambda_weight * dynamic_importance + 
            (1 - lambda_weight) * static_importance.expand_as(dynamic_importance)
        )
        
        # 4. 归一化
        final_importance = F.softmax(
            final_importance.view(-1, self.feature_dim), 
            dim=-1
        ).view_as(features)
        
        # 5. 聚合到组级（复用dynamic estimator的方法）
        group_importance = self.dynamic_estimator.aggregate_to_groups(final_importance)
        
        aux_outputs = {
            'channel_importance': final_importance,
            'group_importance': group_importance,
            'dynamic_importance': dynamic_importance,
            'static_importance': static_importance.squeeze(),
            'fusion_weight': lambda_weight.item(),
            **dynamic_aux  # 包含其他辅助信息
        }
        
        return final_importance, aux_outputs


