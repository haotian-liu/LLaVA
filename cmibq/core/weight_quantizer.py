# cmibq/core/weight_quantizer.py
"""
Weight Quantization Module for CM-IBQ
支持uniform和mixed精度权重量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class QuantizedLinear(nn.Module):
    """
    量化的Linear层
    支持uniform和adaptive混合精度量化
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 4,
        per_channel: bool = True,
        symmetric: bool = True,
        num_groups: int = 8,
        use_adaptive: bool = False,
        granularity: str = 'per_channel'  # Add default parameter
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.num_groups = num_groups
        self.use_adaptive = use_adaptive
        self.granularity = granularity  # Store the granularity
        
        # 存储全精度权重
        self.weight_fp = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 量化参数
        if granularity == 'per_channel':
            self.register_buffer('scale', torch.ones(out_features, 1))
            self.register_buffer('zero_point', torch.zeros(out_features, 1))
        elif granularity == 'per_tensor':
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('zero_point', torch.zeros(1))
        else:  # per_group
            groups = min(num_groups, out_features)
            group_size = out_features // groups
            self.register_buffer('scale', torch.ones(groups, 1))
            self.register_buffer('zero_point', torch.zeros(groups, 1))
            self.group_size = group_size
        
        # 用于自适应量化的比特分配
        if use_adaptive:
            self.register_buffer('bit_assignment', None)
        else:
            self.bit_assignment = None
        
        # 量化后的权重（用于推理）
        self.register_buffer('weight_quantized', None)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 使用Kaiming初始化
        nn.init.kaiming_uniform_(self.weight_fp, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_fp)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize_weight(self, bit_assignment: Optional[torch.Tensor] = None):
        """
        执行权重量化
        
        Args:
            bit_assignment: 可选的per-channel比特分配
        """
        if self.use_adaptive and bit_assignment is not None:
            self.bit_assignment = bit_assignment
            self.weight_quantized = self._adaptive_quantize(self.weight_fp, bit_assignment)
        else:
            self.weight_quantized = self._uniform_quantize(self.weight_fp, self.num_bits)
    
    def _uniform_quantize(self, weight: torch.Tensor, num_bits: int) -> torch.Tensor:
        """
        统一精度量化
        """
        # 确保scale和zero_point在正确的设备上
        device = weight.device
        if self.scale.device != device:
            self.scale = self.scale.to(device)
        if self.zero_point.device != device:
            self.zero_point = self.zero_point.to(device)
        
        if self.symmetric:
            # 对称量化
            qmax = 2 ** (num_bits - 1) - 1
            qmin = -qmax
            
            if self.granularity == 'per_channel':
                # Per-channel量化
                abs_max = weight.abs().max(dim=1, keepdim=True)[0]
                self.scale = abs_max / qmax
                self.scale = self.scale.clamp(min=1e-8)
                self.zero_point.zero_()
            elif self.granularity == 'per_tensor':
                # Per-tensor量化
                abs_max = weight.abs().max()
                self.scale = (abs_max / qmax).unsqueeze(0)
                self.scale = self.scale.clamp(min=1e-8)
                self.zero_point.zero_()
            else:  # per_group
                # Per-group量化
                weight_reshaped = weight.view(-1, self.group_size, weight.shape[1])
                abs_max = weight_reshaped.abs().max(dim=(1, 2), keepdim=True)[0]
                self.scale = (abs_max / qmax).squeeze(-1)
                self.scale = self.scale.clamp(min=1e-8)
                self.zero_point.zero_()
        else:
            # 非对称量化
            qmax = 2 ** num_bits - 1
            qmin = 0
            
            if self.granularity == 'per_channel':
                w_min = weight.min(dim=1, keepdim=True)[0]
                w_max = weight.max(dim=1, keepdim=True)[0]
                self.scale = (w_max - w_min) / (qmax - qmin)
                self.scale = self.scale.clamp(min=1e-8)
                self.zero_point = qmin - w_min / self.scale
            elif self.granularity == 'per_tensor':
                w_min = weight.min()
                w_max = weight.max()
                self.scale = ((w_max - w_min) / (qmax - qmin)).unsqueeze(0)
                self.scale = self.scale.clamp(min=1e-8)
                self.zero_point = (qmin - w_min / self.scale).unsqueeze(0)
            else:  # per_group
                weight_reshaped = weight.view(-1, self.group_size, weight.shape[1])
                w_min = weight_reshaped.min(dim=(1, 2), keepdim=True)[0]
                w_max = weight_reshaped.max(dim=(1, 2), keepdim=True)[0]
                self.scale = ((w_max - w_min) / (qmax - qmin)).squeeze(-1)
                self.scale = self.scale.clamp(min=1e-8)
                self.zero_point = (qmin - w_min.squeeze(-1) / self.scale)
        
        # 量化和反量化
        if self.granularity == 'per_group':
            # 处理per-group的情况
            weight_reshaped = weight.view(-1, self.group_size, weight.shape[1])
            scale_expanded = self.scale.unsqueeze(1).unsqueeze(2)
            zp_expanded = self.zero_point.unsqueeze(1).unsqueeze(2)
            
            weight_q = torch.round(weight_reshaped / scale_expanded + zp_expanded)
            
            if self.symmetric:
                qmax = 2 ** (num_bits - 1) - 1
                weight_q = torch.clamp(weight_q, -qmax, qmax)
            else:
                qmax = 2 ** num_bits - 1
                weight_q = torch.clamp(weight_q, 0, qmax)
            
            weight_dq = (weight_q - zp_expanded) * scale_expanded
            weight_dq = weight_dq.view(weight.shape)
        else:
            weight_q = torch.round(weight / self.scale + self.zero_point)
            
            if self.symmetric:
                qmax = 2 ** (num_bits - 1) - 1
                weight_q = torch.clamp(weight_q, -qmax, qmax)
            else:
                qmax = 2 ** num_bits - 1
                weight_q = torch.clamp(weight_q, 0, qmax)
            
            weight_dq = (weight_q - self.zero_point) * self.scale
        
        return weight_dq
    
    def _adaptive_quantize(self, weight: torch.Tensor, bit_assignment: torch.Tensor) -> torch.Tensor:
        """
        自适应混合精度量化
        每个通道可以有不同的比特数
        """
        weight_quantized = torch.zeros_like(weight)
        
        # 对每个唯一的比特数进行分组量化
        unique_bits = bit_assignment.unique()
        
        for bits in unique_bits:
            mask = (bit_assignment == bits).squeeze()
            if mask.sum() > 0:
                # 选择对应的权重通道
                weight_subset = weight[mask]
                
                # 对这些通道进行统一量化
                if self.symmetric:
                    qmax = 2 ** (bits.item() - 1) - 1
                    abs_max = weight_subset.abs().max(dim=1, keepdim=True)[0]
                    scale = abs_max / qmax
                    scale = scale.clamp(min=1e-8)
                    
                    weight_q = torch.round(weight_subset / scale)
                    weight_q = torch.clamp(weight_q, -qmax, qmax)
                    weight_dq = weight_q * scale
                else:
                    qmax = 2 ** bits.item() - 1
                    w_min = weight_subset.min(dim=1, keepdim=True)[0]
                    w_max = weight_subset.max(dim=1, keepdim=True)[0]
                    scale = (w_max - w_min) / qmax
                    scale = scale.clamp(min=1e-8)
                    zero_point = -w_min / scale
                    
                    weight_q = torch.round(weight_subset / scale + zero_point)
                    weight_q = torch.clamp(weight_q, 0, qmax)
                    weight_dq = (weight_q - zero_point) * scale
                
                weight_quantized[mask] = weight_dq
        
        return weight_quantized
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        训练时使用STE，推理时使用量化权重
        """
        if self.training:
            # 训练时：使用straight-through estimator
            if self.weight_quantized is None:
                self.quantize_weight(self.bit_assignment)
            
            # STE: 前向使用量化权重，反向使用全精度梯度
            weight = self.weight_fp + (self.weight_quantized - self.weight_fp).detach()
        else:
            # 推理时：直接使用量化权重
            if self.weight_quantized is None:
                self.quantize_weight(self.bit_assignment)
            weight = self.weight_quantized
        
        return F.linear(input, weight, self.bias)
    
    def extra_repr(self) -> str:
        s = f'in_features={self.in_features}, out_features={self.out_features}'
        s += f', num_bits={self.num_bits}'
        if self.use_adaptive:
            s += ', adaptive=True'
            if self.bit_assignment is not None:
                avg_bits = self.bit_assignment.mean().item()
                s += f', avg_bits={avg_bits:.2f}'
        s += f', granularity={self.granularity}'
        return s


class WeightQuantizer(nn.Module):
    """
    标准权重量化器
    用于将现有的Linear层转换为量化版本
    """
    
    def __init__(
        self,
        num_bits: int = 4,
        per_channel: bool = True,
        symmetric: bool = True,
        granularity: str = 'per_channel'
    ):
        super().__init__()
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.granularity = granularity
    
    def quantize_layer(self, layer: nn.Linear) -> QuantizedLinear:
        """
        将标准Linear层转换为量化版本
        """
        quantized_layer = QuantizedLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=(layer.bias is not None),
            num_bits=self.num_bits,
            per_channel=self.per_channel,
            symmetric=self.symmetric,
            use_adaptive=False,
            granularity=self.granularity
        )
        
        # 复制权重
        quantized_layer.weight_fp.data = layer.weight.data.clone()
        if layer.bias is not None:
            quantized_layer.bias.data = layer.bias.data.clone()
        
        # 执行初始量化
        quantized_layer.quantize_weight()
        
        return quantized_layer
    
    @torch.no_grad()
    def quantize_model(self, model: nn.Module, skip_modules: Optional[list] = None) -> Dict[str, Any]:
        """
        量化整个模型的Linear层
        
        Args:
            model: 要量化的模型
            skip_modules: 要跳过的模块名称列表
        
        Returns:
            量化统计信息
        """
        skip_modules = skip_modules or []
        stats = {
            'total_layers': 0,
            'quantized_layers': 0,
            'skipped_layers': 0,
            'compression_ratio': 1.0
        }
        
        original_size = 0
        quantized_size = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                stats['total_layers'] += 1
                
                # 检查是否需要跳过
                if any(skip_name in name for skip_name in skip_modules):
                    stats['skipped_layers'] += 1
                    original_size += module.weight.numel() * 32  # FP32
                    quantized_size += module.weight.numel() * 32
                    continue
                
                # 量化层
                quantized_layer = self.quantize_layer(module)
                
                # 替换原始层
                parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
                parent_module = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                
                setattr(parent_module, attr_name, quantized_layer)
                stats['quantized_layers'] += 1
                
                # 计算压缩
                original_size += module.weight.numel() * 32  # FP32
                quantized_size += module.weight.numel() * self.num_bits
        
        if original_size > 0:
            stats['compression_ratio'] = original_size / quantized_size
        
        return stats


class IBWeightQuantizer(nn.Module):
    """
    基于信息瓶颈的自适应权重量化器
    根据重要性动态分配比特
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_groups: int = 8,
        min_bits: int = 2,
        max_bits: int = 8,
        target_bits: float = 4.0
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.target_bits = target_bits
        
        # 重要性估计网络
        self.importance_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # 比特分配网络
        self.bit_allocation_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
    
    def estimate_importance(self, weight: torch.Tensor) -> torch.Tensor:
        """
        估计权重通道的重要性
        
        Args:
            weight: [out_features, in_features]
        
        Returns:
            importance: [out_features]
        """
        # 基于权重统计计算重要性特征
        weight_magnitude = weight.abs().mean(dim=1)
        weight_variance = weight.var(dim=1)
        
        # 组合特征
        features = torch.stack([weight_magnitude, weight_variance], dim=1)
        
        # 通过网络估计重要性
        importance = self.importance_net(weight_magnitude)
        
        return importance
    
    def allocate_bits(self, importance: torch.Tensor) -> torch.Tensor:
        """
        根据重要性分配比特
        
        Args:
            importance: [out_features]
        
        Returns:
            bit_assignment: [out_features]
        """
        # 通过网络生成初始分配
        raw_allocation = self.bit_allocation_net(importance)
        
        # 映射到比特范围
        bit_range = self.max_bits - self.min_bits
        bit_assignment = self.min_bits + bit_range * raw_allocation
        
        # 确保平均比特数接近目标
        current_mean = bit_assignment.mean()
        scaling = self.target_bits / (current_mean + 1e-8)
        bit_assignment = bit_assignment * scaling
        
        # 限制在有效范围内
        bit_assignment = torch.clamp(bit_assignment, self.min_bits, self.max_bits)
        
        # 四舍五入到整数比特
        bit_assignment = torch.round(bit_assignment)
        
        return bit_assignment
    
    def quantize_layer_adaptive(self, layer: nn.Linear) -> QuantizedLinear:
        """
        使用自适应比特分配量化层
        """
        # 创建量化层
        quantized_layer = QuantizedLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=(layer.bias is not None),
            num_bits=int(self.target_bits),
            per_channel=True,
            symmetric=True,
            num_groups=self.num_groups,
            use_adaptive=True,
            granularity='per_channel'
        )
        
        # 复制权重
        quantized_layer.weight_fp.data = layer.weight.data.clone()
        if layer.bias is not None:
            quantized_layer.bias.data = layer.bias.data.clone()
        
        # 估计重要性并分配比特
        with torch.no_grad():
            importance = self.estimate_importance(layer.weight)
            bit_assignment = self.allocate_bits(importance)
        
        # 执行自适应量化
        quantized_layer.quantize_weight(bit_assignment)
        
        return quantized_layer
    
    def forward(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：估计重要性并分配比特
        
        Args:
            weight: 权重张量
        
        Returns:
            importance: 重要性分数
            bit_assignment: 比特分配
        """
        importance = self.estimate_importance(weight)
        bit_assignment = self.allocate_bits(importance)
        return importance, bit_assignment


# 辅助函数
def get_quantization_stats(model: nn.Module) -> Dict[str, Any]:
    """
    获取模型的量化统计信息
    """
    stats = {
        'total_params': 0,
        'quantized_params': 0,
        'avg_bits': 0,
        'layers': {}
    }
    
    total_bits = 0
    num_quantized_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            param_count = module.weight_fp.numel()
            stats['total_params'] += param_count
            stats['quantized_params'] += param_count
            
            if module.use_adaptive and module.bit_assignment is not None:
                layer_bits = module.bit_assignment.mean().item()
            else:
                layer_bits = module.num_bits
            
            stats['layers'][name] = {
                'params': param_count,
                'bits': layer_bits,
                'adaptive': module.use_adaptive
            }
            
            total_bits += layer_bits
            num_quantized_layers += 1
        elif isinstance(module, nn.Linear):
            stats['total_params'] += module.weight.numel()
    
    if num_quantized_layers > 0:
        stats['avg_bits'] = total_bits / num_quantized_layers
    
    if stats['total_params'] > 0:
        # 假设未量化参数使用FP32
        unquantized_params = stats['total_params'] - stats['quantized_params']
        total_bits_original = stats['total_params'] * 32
        total_bits_quantized = (stats['quantized_params'] * stats['avg_bits'] + 
                               unquantized_params * 32)
        stats['compression_ratio'] = total_bits_original / total_bits_quantized
    
    return stats