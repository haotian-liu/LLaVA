import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

class IBQuantizedLayer(nn.Module):
    """
    Information Bottleneck Quantized Layer with target bitrate control
    """
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        target_bits: float = 8.0,
        beta: float = 1.0,
        temperature: float = 1.0,
        use_learnable_prior: bool = False,
        normalize_kl: bool = True,
        adaptive_beta: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.target_bits = target_bits
        self.beta = beta
        self.temperature = nn.Parameter(torch.tensor(temperature))  # 可学习的温度
        self.normalize_kl = normalize_kl
        self.adaptive_beta = adaptive_beta
        
        # 将目标比特转换为目标KL (bits = KL / ln(2))
        self.target_kl = target_bits * math.log(2) * bottleneck_dim
        
        # Encoder: X -> Z (添加前置归一化)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),  # 输入归一化，稳定训练
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim * 2)
        )
        
        # 分离mean和logvar的输出头
        self.encoder_mean = nn.Linear(bottleneck_dim * 2, bottleneck_dim)
        self.encoder_logvar = nn.Linear(bottleneck_dim * 2, bottleneck_dim)
        
        # Decoder: Z -> X
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim * 2),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, input_dim)
        )
        
        # Prior parameters
        if use_learnable_prior:
            # 可学习的先验分布
            self.prior_mean = nn.Parameter(torch.zeros(1, bottleneck_dim))
            self.prior_logvar = nn.Parameter(torch.zeros(1, bottleneck_dim))
        else:
            self.register_buffer('prior_mean', torch.zeros(1, bottleneck_dim))
            self.register_buffer('prior_logvar', torch.zeros(1, bottleneck_dim))
        
        # 自适应beta的动量参数
        if adaptive_beta:
            self.register_buffer('beta_momentum', torch.tensor(0.99))
            self.register_buffer('running_kl', torch.tensor(0.0))
            self.register_buffer('beta_param', torch.tensor(beta))
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入到潜在分布参数"""
        features = self.encoder(x)
        mean = self.encoder_mean(features)
        logvar = self.encoder_logvar(features)
        
        # 限制logvar范围，防止数值不稳定
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # 应用温度退火到方差
        logvar = logvar + 2 * torch.log(self.temperature)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            # 推理时直接使用均值（确定性）
            return mean
    
    def compute_kl_divergence(
        self, 
        mean: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """计算KL散度，支持可学习先验"""
        # KL(q(z|x) || p(z))
        # = 0.5 * sum(1 + log(σ²_q) - log(σ²_p) - (σ²_q + (μ_q - μ_p)²) / σ²_p)
        
        prior_var = torch.exp(self.prior_logvar)
        posterior_var = torch.exp(logvar)
        
        kl = 0.5 * torch.sum(
            self.prior_logvar - logvar - 1 
            + posterior_var / prior_var
            + (mean - self.prior_mean).pow(2) / prior_var,
            dim=-1
        )
        
        return kl
    
    def compute_target_loss(self, kl: torch.Tensor) -> torch.Tensor:
        """计算目标码率损失"""
        # 每个样本的KL
        kl_per_sample = kl.mean()
        
        # 目标KL约束损失
        target_loss = (kl_per_sample - self.target_kl).pow(2)
        
        return target_loss
    
    def update_beta(self, kl: torch.Tensor):
        """自适应调整beta以匹配目标码率"""
        if not self.adaptive_beta or not self.training:
            return
        
        with torch.no_grad():
            current_kl = kl.mean()
            
            # 更新运行均值
            if self.running_kl == 0:
                self.running_kl.copy_(current_kl)
            else:
                self.running_kl.mul_(self.beta_momentum).add_(
                    current_kl, alpha=1 - self.beta_momentum
                )
            
            # 根据KL与目标的差距调整beta
            kl_ratio = self.running_kl / self.target_kl
            
            # 如果KL太高，增加beta；如果太低，减少beta
            if kl_ratio > 1.1:  # KL超出目标10%
                self.beta_param.mul_(1.01)  # 缓慢增加
            elif kl_ratio < 0.9:  # KL低于目标10%
                self.beta_param.mul_(0.99)  # 缓慢减少
            
            # 限制beta范围
            self.beta_param.clamp_(min=1e-4, max=10.0)
    
    def get_importance_scores(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算特征重要性分数，用于连接BitAllocationNetwork
        基于潜在变量的方差或激活强度
        """
        # 方法1：使用激活值的绝对值
        importance = torch.abs(z)
        
        # 方法2：如果有多个样本，可以计算方差
        if z.size(0) > 1:
            importance = importance + z.var(dim=0, keepdim=True).expand_as(z)
        
        return importance
    
    def forward(
        self, 
        x: torch.Tensor,
        return_importance: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征
            return_importance: 是否返回重要性分数（用于量化）
        """
        # 编码
        mean, logvar = self.encode(x)
        
        # 重参数化采样
        z = self.reparameterize(mean, logvar)
        
        # 解码
        x_recon = self.decoder(z)
        
        # 计算损失
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL散度
        kl = self.compute_kl_divergence(mean, logvar)
        
        # KL损失归一化
        if self.normalize_kl:
            # 归一化到每个维度
            kl_loss = kl.sum() / (x.size(0) * self.bottleneck_dim)
        else:
            kl_loss = kl.mean()
        
        # 目标码率损失
        target_loss = self.compute_target_loss(kl)
        
        # 自适应更新beta
        if self.adaptive_beta:
            self.update_beta(kl)
            beta = self.beta_param
        else:
            beta = self.beta
        
        # 总损失：重建 + β*KL + λ*目标码率
        total_loss = recon_loss + beta * kl_loss + 0.1 * target_loss
        
        result = {
            'z': z,
            'x_recon': x_recon,
            'mean': mean,
            'logvar': logvar,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'target_loss': target_loss,
            'actual_bits': kl.sum() / (x.size(0) * math.log(2)),  # 实际比特数
            'beta': beta if self.adaptive_beta else self.beta
        }
        
        # 返回重要性分数（用于量化）
        if return_importance:
            result['importance_scores'] = self.get_importance_scores(z)
        
        return result



    