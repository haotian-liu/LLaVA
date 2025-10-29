import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InformationBottleneck(nn.Module):
    """
    率失真理论的信息瓶颈实现
    将输入特征压缩到瓶颈维度，同时最小化信息损失
    """
    def __init__(self, input_dim, bottleneck_dim, beta=1.0, use_vae=True):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_vae = use_vae
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, bottleneck_dim * 2 if use_vae else bottleneck_dim)
        )
        
        # 解码器网络（用于重构）
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim)
        )
        
        # 可学习的率失真权衡参数
        self.log_beta = nn.Parameter(torch.tensor(np.log(beta)))
        
        # 统计量追踪
        self.register_buffer('rate_ema', torch.tensor(0.0))
        self.register_buffer('distortion_ema', torch.tensor(0.0))
        
    def reparameterize(self, mu, log_var):
        """VAE重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.use_vae:
            # 变分自编码器方式
            stats = self.encoder(x)
            mu, log_var = stats.chunk(2, dim=-1)
            
            # 重参数化采样
            z = self.reparameterize(mu, log_var)
            
            # 计算率（KL散度）
            # KL(q(z|x)||p(z)) where p(z) = N(0,I)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
            rate = kl_loss.mean()
            
        else:
            # 确定性编码器
            z = self.encoder(x)
            # 使用L2正则化作为率的代理
            rate = 0.5 * torch.sum(z.pow(2), dim=-1).mean()
            
        # 重构以计算失真
        x_recon = self.decoder(z)
        distortion = F.mse_loss(x_recon, x, reduction='mean')
        
        # 更新统计量（用于监控）
        self.rate_ema = 0.9 * self.rate_ema + 0.1 * rate.detach()
        self.distortion_ema = 0.9 * self.distortion_ema + 0.1 * distortion.detach()
        
        # 率失真损失
        beta = torch.exp(self.log_beta)
        ib_loss = beta * rate + distortion
        
        return z, {
            'ib_loss': ib_loss,
            'rate': rate,
            'distortion': distortion,
            'beta': beta
        }


class HierarchicalIB(nn.Module):
    """
    分层信息瓶颈：在不同层级应用不同的压缩率
    """
    def __init__(self, dims, compression_ratios, base_beta=1.0):
        super().__init__()
        assert len(dims) == len(compression_ratios) + 1
        
        self.layers = nn.ModuleList()
        for i in range(len(compression_ratios)):
            input_dim = dims[i]
            bottleneck_dim = int(dims[i+1] * compression_ratios[i])
            
            # 深层使用更小的beta（更少压缩）
            beta = base_beta * (0.5 ** i)
            
            self.layers.append(
                InformationBottleneck(input_dim, bottleneck_dim, beta=beta)
            )
    
    def forward(self, x):
        ib_losses = []
        rates = []
        distortions = []
        
        for layer in self.layers:
            x, ib_info = layer(x)
            ib_losses.append(ib_info['ib_loss'])
            rates.append(ib_info['rate'])
            distortions.append(ib_info['distortion'])
            
        total_ib_loss = sum(ib_losses) / len(ib_losses)
        avg_rate = sum(rates) / len(rates)
        avg_distortion = sum(distortions) / len(distortions)
        
        return x, {
            'ib_loss': total_ib_loss,
            'rate': avg_rate,
            'distortion': avg_distortion,
            'layer_rates': rates,
            'layer_distortions': distortions
        }