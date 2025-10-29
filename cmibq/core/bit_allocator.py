# bit_allocator.py - 增强版，添加了预算验证和更好的约束处理
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings

class BitAllocationNetwork(nn.Module):
    def __init__(self, feature_dim, min_bits=2.0, max_bits=8.0, 
                 target_bits=4.0, num_groups=8, bit_levels=[2, 4, 8],
                 strict_budget=True, budget_tolerance=0.1):
        """
        改进的比特分配网络
        
        Args:
            strict_budget: 是否严格执行预算约束
            budget_tolerance: 预算容忍度（仅在strict_budget=True时使用）
        """
        super().__init__()
        
        assert feature_dim % num_groups == 0, "feature_dim must be divisible by num_groups"
        
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.group_size = feature_dim // num_groups
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.target_bits = target_bits
        self.strict_budget = strict_budget
        self.budget_tolerance = budget_tolerance
        
        # 改进的预测器：接受组级输入
        self.bit_predictor = nn.Sequential(
            nn.Linear(num_groups, num_groups * 2),
            nn.GELU(),
            nn.LayerNorm(num_groups * 2),
            nn.Linear(num_groups * 2, num_groups)
        )
        
        # 可学习的位嵌入
        self.bit_embeddings = nn.Parameter(
            torch.randn(len(bit_levels), self.group_size)
        )
        
        self.register_buffer('bit_levels', torch.tensor(bit_levels, dtype=torch.float32))
        self.register_buffer('temperature', torch.tensor(1.0))
        
        # 统计buffer，用于监控
        self.register_buffer('budget_violations', torch.tensor(0))
        self.register_buffer('total_allocations', torch.tensor(0))
    
    def allocate_bits_with_constraints(self, bit_probs):
        """根据概率分配比特，满足约束"""
        batch_size = bit_probs.size(0)
        
        # 基于概率分配比特
        total_bit_budget = self.target_bits * self.num_groups
        bit_allocation = self.min_bits + bit_probs * (self.max_bits - self.min_bits)
        
        # 归一化以满足预算
        current_total = bit_allocation.sum(dim=-1, keepdim=True)
        bit_allocation = bit_allocation * (total_bit_budget / current_total)
        
        # 确保在范围内
        bit_allocation = torch.clamp(bit_allocation, self.min_bits, self.max_bits)
        
        return bit_allocation
    
    def discretize_bits(self, bit_allocation):
        """将连续比特分配离散化到允许的级别"""
        # 找到最近的离散级别
        distances = torch.abs(
            bit_allocation.unsqueeze(-1) - self.bit_levels.view(1, 1, -1)
        )
        discrete_indices = distances.argmin(dim=-1)
        discrete_bits = self.bit_levels[discrete_indices]
        
        # Straight-through gradient
        if self.training:
            discrete_bits = discrete_bits + (bit_allocation - bit_allocation.detach())
        
        return discrete_bits
    
    def discretize_with_budget(
        self, 
        bit_assignment: torch.Tensor,
        target_budget: float,
        bit_levels: list
    ) -> torch.Tensor:
        """
        离散化比特分配到允许的级别并强制执行预算约束
        增强版：添加了验证和更好的约束处理
        
        Args:
            bit_assignment: [B, num_groups] 连续的比特分配
            target_budget: 目标平均比特数
            bit_levels: 允许的离散比特级别 (e.g., [2, 4, 8])
        
        Returns:
            discrete_bits: [B, num_groups] 离散化后的比特分配
        """
        batch_size, num_groups = bit_assignment.shape
        bit_levels_tensor = torch.tensor(bit_levels, device=bit_assignment.device, dtype=torch.float32)
        
        # Step 1: 找到每个分配最近的离散级别
        distances = torch.abs(bit_assignment.unsqueeze(-1) - bit_levels_tensor.view(1, 1, -1))
        discrete_bits = bit_levels_tensor[distances.argmin(dim=-1)]
        
        # Step 2: 逐样本调整以满足预算约束
        for b in range(batch_size):
            current_budget = discrete_bits[b].mean()
            iterations = 0
            max_iterations = num_groups * 3  # 增加最大迭代次数
            
            # 记录初始预算差距
            initial_gap = abs(current_budget - target_budget)
            
            while abs(current_budget - target_budget) > self.budget_tolerance and iterations < max_iterations:
                iterations += 1
                
                if current_budget > target_budget:
                    # 需要减少比特
                    # 找到可以减少的组
                    reducible = discrete_bits[b] > min(bit_levels)
                    if not reducible.any():
                        break
                    
                    # 优先减少比特数最高的组
                    reducible_indices = torch.where(reducible)[0]
                    
                    # 计算每个组减少后对预算的影响
                    impacts = []
                    for idx in reducible_indices:
                        current_level_idx = (discrete_bits[b, idx] == bit_levels_tensor).nonzero()[0, 0]
                        if current_level_idx > 0:
                            new_level = bit_levels_tensor[current_level_idx - 1]
                            impact = (discrete_bits[b, idx] - new_level) / num_groups
                            impacts.append((idx, impact))
                    
                    # 选择影响最接近需要调整量的组
                    if impacts:
                        gap = current_budget - target_budget
                        best_idx = min(impacts, key=lambda x: abs(x[1] - gap))[0]
                        
                        current_level_idx = (discrete_bits[b, best_idx] == bit_levels_tensor).nonzero()[0, 0]
                        discrete_bits[b, best_idx] = bit_levels_tensor[current_level_idx - 1]
                    
                else:
                    # 需要增加比特
                    # 找到可以增加的组
                    increasable = discrete_bits[b] < max(bit_levels)
                    if not increasable.any():
                        break
                    
                    # 优先增加比特数最低的组
                    increasable_indices = torch.where(increasable)[0]
                    
                    # 计算每个组增加后对预算的影响
                    impacts = []
                    for idx in increasable_indices:
                        current_level_idx = (discrete_bits[b, idx] == bit_levels_tensor).nonzero()[0, 0]
                        if current_level_idx < len(bit_levels) - 1:
                            new_level = bit_levels_tensor[current_level_idx + 1]
                            impact = (new_level - discrete_bits[b, idx]) / num_groups
                            impacts.append((idx, impact))
                    
                    # 选择影响最接近需要调整量的组
                    if impacts:
                        gap = target_budget - current_budget
                        best_idx = min(impacts, key=lambda x: abs(x[1] - gap))[0]
                        
                        current_level_idx = (discrete_bits[b, best_idx] == bit_levels_tensor).nonzero()[0, 0]
                        discrete_bits[b, best_idx] = bit_levels_tensor[current_level_idx + 1]
                
                current_budget = discrete_bits[b].mean()
            
            # 验证最终分配是否满足约束
            final_budget = discrete_bits[b].mean()
            budget_error = abs(final_budget - target_budget)
            
            if self.strict_budget and budget_error > self.budget_tolerance:
                # 记录违反次数
                self.budget_violations += 1
                
                if budget_error > 0.5:  # 严重违反
                    warnings.warn(f"Severe budget violation: {final_budget:.2f} vs target {target_budget:.2f}")
                    
                    # 尝试强制修正
                    if final_budget > target_budget:
                        # 强制降低最高比特的组
                        max_indices = (discrete_bits[b] == discrete_bits[b].max()).nonzero().squeeze()
                        if max_indices.numel() > 0:
                            if max_indices.numel() == 1:
                                max_indices = [max_indices.item()]
                            else:
                                max_indices = max_indices.tolist()
                            
                            for idx in max_indices[:int(len(max_indices)/2)]:  # 降低一半的最高比特组
                                current_level_idx = (discrete_bits[b, idx] == bit_levels_tensor).nonzero()[0, 0]
                                if current_level_idx > 0:
                                    discrete_bits[b, idx] = bit_levels_tensor[current_level_idx - 1]
                                    
                                new_budget = discrete_bits[b].mean()
                                if abs(new_budget - target_budget) <= self.budget_tolerance:
                                    break
            
            self.total_allocations += 1
        
        # 最终验证和断言（仅在strict_budget模式下）
        if self.strict_budget:
            final_avg_budget = discrete_bits.mean()
            budget_violation = abs(final_avg_budget - target_budget)
            
            if budget_violation > self.budget_tolerance:
                violation_rate = (self.budget_violations.float() / max(1, self.total_allocations.float())).item()
                
                if violation_rate > 0.1:  # 超过10%的分配违反预算
                    warnings.warn(
                        f"High budget violation rate: {violation_rate:.1%}. "
                        f"Current: {final_avg_budget:.2f} vs Target: {target_budget:.2f}"
                    )
                
                # 在训练时使用软约束，推理时使用硬约束
                if not self.training and budget_violation > 0.5:
                    raise AssertionError(
                        f"Budget constraint severely violated: "
                        f"{final_avg_budget:.2f} vs {target_budget:.2f} "
                        f"(tolerance: {self.budget_tolerance})"
                    )
        
        return discrete_bits
    
    def generate_bit_embeddings(self, discrete_bits):
        """生成比特嵌入"""
        batch_size = discrete_bits.size(0)
        
        # 为每个比特级别创建one-hot或soft分配
        bit_weights = []
        for level in self.bit_levels:
            weight = torch.exp(-torch.abs(discrete_bits - level))
            bit_weights.append(weight)
        
        bit_weights = torch.stack(bit_weights, dim=-1)  # [B, num_groups, num_levels]
        bit_weights = F.softmax(bit_weights, dim=-1)
        
        # 加权组合嵌入
        embeddings = []
        for g in range(self.num_groups):
            group_weights = bit_weights[:, g, :]  # [B, num_levels]
            # 加权求和
            group_embedding = torch.matmul(group_weights, self.bit_embeddings)  # [B, group_size]
            embeddings.append(group_embedding)
        
        embeddings = torch.stack(embeddings, dim=1)  # [B, num_groups, group_size]
        return embeddings
    
    def forward(self, importance_scores, features=None):
        """
        前向传播
        
        Args:
            importance_scores: 可以是 [B, D] 或 [B, T, D]
            features: 可选的特征输入
        
        Returns:
            包含比特分配结果的字典
        """
        # 处理不同的输入维度
        if importance_scores.dim() == 3:  # [B, T, D]
            # 在时间维度上聚合
            importance_scores = importance_scores.mean(dim=1)  # [B, D]
        
        batch_size = importance_scores.size(0)
        
        # 1. 计算组级重要性
        importance_reshaped = importance_scores.view(batch_size, self.num_groups, -1)
        group_importance = importance_reshaped.mean(dim=-1)  # [B, num_groups]
        
        # 2. 预测比特分配
        bit_logits = self.bit_predictor(group_importance)
        
        # 3. Gumbel-Softmax（用于可微分采样）
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(bit_logits) + 1e-8) + 1e-8)
            bit_logits = bit_logits + gumbel_noise
        
        bit_probs = F.softmax(bit_logits / self.temperature, dim=-1)
        
        # 4. 分配比特（带约束）
        bit_assignment = self.allocate_bits_with_constraints(bit_probs)
        
        # 5. 离散化（使用增强的预算约束）
        discrete_bits = self.discretize_with_budget(
            bit_assignment, 
            self.target_bits, 
            self.bit_levels.tolist()
        )
        
        # 6. 生成比特嵌入
        bit_embeddings = self.generate_bit_embeddings(discrete_bits)
        
        # 7. 生成group_indices（重要！）
        group_indices = torch.arange(self.feature_dim, device=importance_scores.device)
        group_indices = group_indices // self.group_size
        group_indices = group_indices.unsqueeze(0).expand(batch_size, -1)
        
        # 8. 计算损失
        # 比特率损失（鼓励接近目标比特）
        actual_avg_bits = discrete_bits.mean()
        bitrate_loss = F.mse_loss(
            actual_avg_bits, 
            torch.tensor(self.target_bits, device=discrete_bits.device)
        )
        
        # 多样性奖励（鼓励使用不同的比特级别）
        entropy = -(bit_probs * torch.log(bit_probs + 1e-8)).sum(dim=-1).mean()
        diversity_bonus = entropy * 0.1  # 缩放因子
        
        # 预算违反惩罚（仅在strict_budget模式下）
        budget_penalty = torch.tensor(0.0, device=discrete_bits.device)
        if self.strict_budget:
            budget_error = abs(actual_avg_bits - self.target_bits)
            if budget_error > self.budget_tolerance:
                budget_penalty = budget_error * 10.0  # 高惩罚
        
        # 组合总损失
        total_auxiliary_loss = bitrate_loss + budget_penalty - diversity_bonus
        
        # 返回结果
        result = {
            'bit_assignment': discrete_bits,
            'group_indices': group_indices,
            'bit_embeddings': bit_embeddings,
            'bitrate_loss': bitrate_loss,
            'diversity_bonus': diversity_bonus,
            'budget_penalty': budget_penalty,
            'total_auxiliary_loss': total_auxiliary_loss,
            'bit_probs': bit_probs,
            # 统计信息
            'actual_avg_bits': actual_avg_bits.item(),
            'budget_error': abs(actual_avg_bits - self.target_bits).item(),
            'violation_rate': (self.budget_violations.float() / max(1, self.total_allocations.float())).item()
        }
        
        return result
    
    def get_statistics(self):
        """获取分配统计信息"""
        stats = {
            'budget_violations': self.budget_violations.item(),
            'total_allocations': self.total_allocations.item(),
            'violation_rate': (self.budget_violations.float() / max(1, self.total_allocations.float())).item(),
            'target_bits': self.target_bits,
            'min_bits': self.min_bits,
            'max_bits': self.max_bits,
            'num_groups': self.num_groups,
            'temperature': self.temperature.item()
        }
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.budget_violations.zero_()
        self.total_allocations.zero_()
