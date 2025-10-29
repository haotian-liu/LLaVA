import torch
import torch.nn as nn
import torch.nn.functional as F
from .information_bottleneck import InformationBottleneck


class CrossModalCompressor(nn.Module):
    """
    跨模态信息压缩器
    联合优化视觉和文本的信息保留，同时维持跨模态对齐
    """
    def __init__(self, vis_dim, text_dim, hidden_dim, 
                 compression_ratio=0.5, beta_init=1.0):
        super().__init__()
        
        self.vis_dim = vis_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # 视觉信息瓶颈
        self.vis_ib = InformationBottleneck(
            vis_dim, 
            int(hidden_dim * compression_ratio),
            beta=beta_init
        )
        
        # 文本信息瓶颈
        self.text_ib = InformationBottleneck(
            text_dim,
            int(hidden_dim * compression_ratio),
            beta=beta_init
        )
        
        # 跨模态对齐投影
        compressed_dim = int(hidden_dim * compression_ratio)
        self.vis_proj = nn.Linear(compressed_dim, hidden_dim)
        self.text_proj = nn.Linear(compressed_dim, hidden_dim)
        
        # 跨模态融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 对齐预测头（用于对比学习）
        self.alignment_head = nn.Linear(hidden_dim, hidden_dim)
        
    def compute_alignment_loss(self, z_v, z_t, temperature=0.07):
        """计算跨模态对齐损失（InfoNCE）"""
        batch_size = z_v.size(0)
        
        # 归一化
        z_v = F.normalize(z_v, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(z_v, z_t.t()) / temperature
        
        # 对比学习损失
        labels = torch.arange(batch_size, device=z_v.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2
    
    def forward(self, vis_features, text_features, return_features=False):
        """
        前向传播
        Args:
            vis_features: 视觉特征 [batch, seq_len, vis_dim]
            text_features: 文本特征 [batch, seq_len, text_dim]
            return_features: 是否返回中间特征
        """
        batch_size = vis_features.size(0)
        
        # 1. 信息瓶颈压缩
        z_v, vis_ib_info = self.vis_ib(vis_features.mean(dim=1))  # 池化后压缩
        z_t, text_ib_info = self.text_ib(text_features.mean(dim=1))
        
        # 2. 投影到共同空间
        v_proj = self.vis_proj(z_v)
        t_proj = self.text_proj(z_t)
        
        # 3. 计算对齐损失
        alignment_loss = self.compute_alignment_loss(v_proj, t_proj)
        
        # 4. 跨模态融合
        fused = torch.cat([v_proj, t_proj], dim=-1)
        output = self.fusion(fused)
        
        # 5. 扩展回序列长度（用于后续transformer层）
        seq_len = max(vis_features.size(1), text_features.size(1))
        output_expanded = output.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 计算总损失
        total_ib_loss = vis_ib_info['ib_loss'] + text_ib_info['ib_loss']
        total_rate = vis_ib_info['rate'] + text_ib_info['rate']
        total_distortion = vis_ib_info['distortion'] + text_ib_info['distortion']
        
        # 率失真 + 对齐损失
        rd_loss = total_ib_loss + alignment_loss
        
        info = {
            'rd_loss': rd_loss,
            'ib_loss': total_ib_loss,
            'alignment_loss': alignment_loss,
            'rate': total_rate,
            'distortion': total_distortion,
            'vis_rate': vis_ib_info['rate'],
            'text_rate': text_ib_info['rate'],
            'vis_distortion': vis_ib_info['distortion'],
            'text_distortion': text_ib_info['distortion']
        }
        
        if return_features:
            info['compressed_features'] = {
                'z_v': z_v,
                'z_t': z_t,
                'v_proj': v_proj,
                't_proj': t_proj
            }
        
        return output_expanded, info


class AdaptiveCrossModalCompressor(CrossModalCompressor):
    """
    自适应跨模态压缩器
    根据输入动态调整压缩率
    """
    def __init__(self, vis_dim, text_dim, hidden_dim, 
                 min_compression=0.3, max_compression=0.7):
        super().__init__(vis_dim, text_dim, hidden_dim)
        
        self.min_compression = min_compression
        self.max_compression = max_compression
        
        # 压缩率预测器
        self.rate_predictor = nn.Sequential(
            nn.Linear(vis_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def predict_compression_rate(self, vis_features, text_features):
        """根据输入特征预测最优压缩率"""
        # 拼接池化特征
        vis_pooled = vis_features.mean(dim=1)
        text_pooled = text_features.mean(dim=1)
        combined = torch.cat([vis_pooled, text_pooled], dim=-1)
        
        # 预测压缩率
        rate = self.rate_predictor(combined)
        rate = self.min_compression + rate * (self.max_compression - self.min_compression)
        
        return rate
    
    def forward(self, vis_features, text_features):
        # 动态压缩率
        compression_rate = self.predict_compression_rate(vis_features, text_features)
        
        # 调整信息瓶颈的beta参数
        self.vis_ib.log_beta.data = torch.log(compression_rate)
        self.text_ib.log_beta.data = torch.log(compression_rate)
        
        # 标准前向传播
        output, info = super().forward(vis_features, text_features)
        info['compression_rate'] = compression_rate.mean()
        
        return output, info