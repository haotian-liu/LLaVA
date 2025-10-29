import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import copy

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class LlavaCIBD(LlavaLlamaForCausalLM):
    """
    CIBD压缩模型：
    1. 学生模型参数量真正减少（通过config控制）
    2. 信息瓶颈用于视觉特征压缩
    3. 保留原始文本序列，不覆盖
    4. Teacher-Student正确分路
    """
    
    def __init__(self, config, teacher_model=None):
        # 创建较小的学生模型
        super().__init__(config)
        
        self.teacher_model = teacher_model
        self.config = config
        
        if teacher_model:
            # 冻结教师模型
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        
        # 信息瓶颈模块（只用于视觉特征）
        self.visual_compressor = VisualInformationBottleneck(
            input_dim=config.hidden_size,
            bottleneck_dim=int(config.hidden_size * 0.6),  
            output_dim=config.hidden_size
        )
        
        # 特征对齐投影（如果学生和教师维度不同）
        if teacher_model:
            teacher_hidden = teacher_model.config.hidden_size
            student_hidden = config.hidden_size
            
            if teacher_hidden != student_hidden:
                # 需要投影来对齐维度进行蒸馏
                self.feature_projector = nn.Linear(student_hidden, teacher_hidden)
                # 反向投影（可选）
                self.reverse_projector = nn.Linear(teacher_hidden, student_hidden)
            else:
                self.feature_projector = nn.Identity()
                self.reverse_projector = nn.Identity()
        
        # 率失真权衡参数（可学习）
        self.log_beta = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1
        
    def encode_images_with_ib(self, images):
        """使用信息瓶颈编码图像"""
        # 先用原始方法编码
        image_features = self.encode_images(images)
        
        # 然后通过信息瓶颈压缩
        # 注意：只压缩视觉特征，保持序列长度不变
        batch_size, seq_len, hidden_dim = image_features.shape
        
        # Reshape for bottleneck
        image_features_flat = image_features.view(-1, hidden_dim)
        compressed_flat, ib_loss = self.visual_compressor(image_features_flat)
        
        # Reshape back
        compressed_features = compressed_flat.view(batch_size, seq_len, hidden_dim)
        
        return compressed_features, ib_loss
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # 初始化损失
        ib_loss = 0
        distill_loss = 0
        
        # 处理输入
        if images is not None:
            # 学生路径：使用压缩的视觉特征
            compressed_images, ib_loss = self.encode_images_with_ib(images)
            
            # 准备学生的输入（使用压缩特征）
            (
                student_input_ids,
                student_position_ids,
                student_attention_mask,
                student_past_key_values,
                student_inputs_embeds,
                student_labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                compressed_images,  # 压缩后的视觉特征
                image_sizes
            )
        else:
            # 没有图像，直接使用原始输入
            student_input_ids = input_ids
            student_position_ids = position_ids
            student_attention_mask = attention_mask
            student_past_key_values = past_key_values
            student_inputs_embeds = inputs_embeds
            student_labels = labels
        
        # 学生模型前向传播
        outputs = super().forward(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            position_ids=student_position_ids,
            past_key_values=student_past_key_values,
            inputs_embeds=student_inputs_embeds,
            labels=student_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 需要hidden states用于蒸馏
            return_dict=True
        )
        
        # 基础任务损失
        total_loss = outputs.loss if outputs.loss is not None else 0
        
        # 添加信息瓶颈损失（带权重）
        if ib_loss > 0:
            beta = torch.exp(self.log_beta)
            total_loss = total_loss + beta * ib_loss
        
        # 知识蒸馏（如果有教师模型）
        if self.training and self.teacher_model is not None and labels is not None:
            with torch.no_grad():
                # 教师路径：使用原始输入（未压缩）
                if images is not None:
                    # 教师使用原始图像编码
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        images=images,  # 原始图像
                        image_sizes=image_sizes,
                        labels=labels,
                        output_hidden_states=True,
                        return_dict=True
                    )
                else:
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=labels,
                        output_hidden_states=True,
                        return_dict=True
                    )
            
            # 1. Logits蒸馏（KL散度）
            temperature = 4.0
            
            # 对齐序列长度（如果需要）
            student_logits = outputs.logits
            teacher_logits = teacher_outputs.logits
            
            min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_seq_len, :]
            teacher_logits = teacher_logits[:, :min_seq_len, :]
            
            # KL散度损失
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # 2. 特征蒸馏（隐藏状态）
            # 选择最后一层
            student_hidden = outputs.hidden_states[-1]
            teacher_hidden = teacher_outputs.hidden_states[-1]
            
            # 对齐维度和序列长度
            min_seq_len = min(student_hidden.size(1), teacher_hidden.size(1))
            student_hidden = student_hidden[:, :min_seq_len, :]
            teacher_hidden = teacher_hidden[:, :min_seq_len, :]
            
            # 如果维度不同，需要投影
            if student_hidden.size(-1) != teacher_hidden.size(-1):
                student_hidden_proj = self.feature_projector(student_hidden)
                feature_loss = F.mse_loss(student_hidden_proj, teacher_hidden)
            else:
                feature_loss = F.mse_loss(student_hidden, teacher_hidden)
            
            # 组合蒸馏损失
            distill_loss = 0.7 * kl_loss + 0.3 * feature_loss
            total_loss = total_loss + distill_loss
        
        if return_dict:
            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        return (total_loss, outputs.logits)


class VisualInformationBottleneck(nn.Module):
    """
    视觉信息瓶颈模块
    VAE风格的压缩，带率失真优化
    """
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.LayerNorm(bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )
        
        self.bottleneck_dim = bottleneck_dim
        
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码为均值和方差
        stats = self.encoder(x)
        mu, log_var = stats.chunk(2, dim=-1)
        
        # 限制log_var的范围，避免数值不稳定
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # 重参数化采样
        z = self.reparameterize(mu, log_var)
        
        # 解码
        x_recon = self.decoder(z)
        
        # 计算损失
        # KL散度（信息率）
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # 重构损失（失真）
        recon_loss = F.mse_loss(x_recon, x)
        
        # 总的IB损失
        ib_loss = kl_loss + recon_loss
        
        return x_recon, ib_loss