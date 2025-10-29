#!/usr/bin/env python3
"""
优化版 CIBD 模型 - 支持Hidden Size压缩
改进点：
1. IB应用到纯视觉特征（在mm_projector之后）
2. 多层特征蒸馏（浅层、中层、深层）
3. 余弦相似度损失替代MSE
4. ✨ 支持hidden_size压缩（添加可学习的适配器）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging

logger = logging.getLogger(__name__)


class ImprovedVisualIB(nn.Module):
    """
    改进的视觉信息瓶颈
    应用在纯视觉特征上（mm_projector输出之后）
    """
    def __init__(self, input_dim, compression_ratio=0.7):
        super().__init__()
        
        bottleneck_dim = int(input_dim * compression_ratio)
        
        # VAE编码器（输出mu和log_var）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, bottleneck_dim * 2)
        )
        
        # VAE解码器
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        
        # 可学习的beta
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, visual_features):
        """
        Args:
            visual_features: [batch, num_patches, hidden_dim]
        Returns:
            compressed_features, ib_loss
        """
        batch_size, num_patches, hidden_dim = visual_features.shape
        
        # Flatten for processing
        x = visual_features.view(-1, hidden_dim)
        
        # Encode
        stats = self.encoder(x)
        mu, log_var = stats.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Reshape back
        compressed_features = x_recon.view(batch_size, num_patches, hidden_dim)
        
        # Compute IB loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        recon_loss = F.mse_loss(x_recon, x)
        
        beta = torch.exp(self.log_beta)
        ib_loss = beta * kl_loss + recon_loss
        
        return compressed_features, {
            'ib_loss': ib_loss,
            'kl_loss': kl_loss,
            'recon_loss': recon_loss,
            'beta': beta
        }


class MultiLayerDistillation(nn.Module):
    """
    多层特征蒸馏 - 支持Hidden Size不匹配
    在浅层、中层、深层分别进行蒸馏
    ✨ 当学生hidden_size < 教师hidden_size时，使用可学习的投影层
    """
    def __init__(self, student_hidden, teacher_hidden, num_student_layers, num_teacher_layers):
        super().__init__()
        
        self.num_student_layers = num_student_layers
        self.num_teacher_layers = num_teacher_layers
        self.student_hidden = student_hidden
        self.teacher_hidden = teacher_hidden
        
        # 选择要对齐的层（浅层、中层、深层）
        self.student_layer_indices = [
            num_student_layers // 4,      # 浅层
            num_student_layers // 2,      # 中层
            3 * num_student_layers // 4   # 深层
        ]
        
        self.teacher_layer_indices = [
            num_teacher_layers // 4,
            num_teacher_layers // 2,
            3 * num_teacher_layers // 4
        ]
        
        # ===== 关键修复：正确初始化投影层 =====
        self.projectors = nn.ModuleList()
        if student_hidden != teacher_hidden:
            logger.info(f"✨ 创建Hidden维度适配器: {student_hidden} -> {teacher_hidden}")
            for i in range(3):  # 3个层级
                proj = nn.Linear(student_hidden, teacher_hidden, bias=False)
                # Xavier初始化，保证训练初期稳定
                nn.init.xavier_uniform_(proj.weight)
                self.projectors.append(proj)
                logger.info(f"   - 层级{i}适配器已初始化")
        else:
            # 维度相同，使用Identity
            for _ in range(3):
                self.projectors.append(nn.Identity())
            logger.info(f"✓ Hidden维度相同，无需适配器")
        
        logger.info(f"多层蒸馏配置:")
        logger.info(f"  学生层: {self.student_layer_indices}")
        logger.info(f"  教师层: {self.teacher_layer_indices}")
        logger.info(f"  维度: student={student_hidden}, teacher={teacher_hidden}")
    
    def align_hidden(self, student_h, teacher_h, adapter_idx):
        """
        安全对齐hidden维度
        
        Args:
            student_h: [B, S, Hs] 学生hidden states
            teacher_h: [B, S, Ht] 教师hidden states
            adapter_idx: 使用哪个适配器（0, 1, 2）
        
        Returns:
            aligned_student_h: [B, S, Ht] 对齐后的学生特征
        """
        if student_h.size(-1) == teacher_h.size(-1):
            # 维度已匹配，直接返回
            return student_h
        
        # 使用对应的投影层
        projector = self.projectors[adapter_idx]
        aligned_h = projector(student_h)
        
        return aligned_h
    
    def cosine_similarity_loss(self, student_feat, teacher_feat):
        """
        余弦相似度损失
        鼓励特征方向对齐，而不是值对齐
        """
        # Normalize
        student_norm = F.normalize(student_feat, dim=-1)
        teacher_norm = F.normalize(teacher_feat, dim=-1)
        
        # Cosine similarity: 1 - cos(theta)
        cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
        loss = 1.0 - cos_sim
        
        return loss
    
    def forward(self, student_hidden_states, teacher_hidden_states, attention_mask=None):
        """
        Args:
            student_hidden_states: List of [batch, seq_len, student_hidden]
            teacher_hidden_states: List of [batch, seq_len, teacher_hidden]
            attention_mask: [batch, seq_len]
        Returns:
            total_loss, loss_dict
        """
        total_loss = 0
        loss_dict = {}
        
        for i, (s_idx, t_idx) in enumerate(zip(self.student_layer_indices, self.teacher_layer_indices)):
            # 获取对应层的hidden states
            student_hidden = student_hidden_states[s_idx]
            teacher_hidden = teacher_hidden_states[t_idx]
            
            # 对齐序列长度
            min_len = min(student_hidden.size(1), teacher_hidden.size(1))
            student_hidden = student_hidden[:, :min_len, :]
            teacher_hidden = teacher_hidden[:, :min_len, :]
            
            # ===== 关键修复：使用安全对齐函数 =====
            student_hidden = self.align_hidden(student_hidden, teacher_hidden, adapter_idx=i)
            
            # 应用attention mask（如果提供）
            if attention_mask is not None:
                # 显式转换mask为hidden同dtype（避免AMP/加速器下的类型提升问题）
                mask = attention_mask[:, :min_len].unsqueeze(-1).to(dtype=student_hidden.dtype)
                student_hidden = student_hidden * mask
                teacher_hidden = teacher_hidden * mask
            
            # 计算余弦相似度损失
            layer_loss = self.cosine_similarity_loss(student_hidden, teacher_hidden)
            
            total_loss += layer_loss
            loss_dict[f'layer_{i}_loss'] = layer_loss.item()
        
        # 平均
        total_loss = total_loss / len(self.student_layer_indices)
        
        return total_loss, loss_dict


class LlavaCIBDModelV2(LlavaLlamaForCausalLM):
    """
    优化版 CIBD 模型
    
    改进：
    1. IB应用到纯视觉特征（encode_images之后）
    2. 多层特征蒸馏（3个层级）
    3. 余弦相似度损失
    4. ✨ 支持hidden_size压缩（可学习适配器）
    5. 更好的损失权重平衡
    """
    
    def __init__(self, config, teacher_model=None, **kwargs):
        super().__init__(config)
        
        self.teacher_model = teacher_model
        self.config = config
        
        # ===== 接受蒸馏参数 =====
        self.temperature = kwargs.get('temperature', 4.0)
        initial_kd_weight = kwargs.get('kd_weight', 0.5)
        self.log_kd_weight = nn.Parameter(torch.log(torch.tensor(initial_kd_weight)))
        
        initial_feat_weight = kwargs.get('feat_weight', 0.2)
        self.log_feat_weight = nn.Parameter(torch.log(torch.tensor(initial_feat_weight)))
        
        initial_ib_weight = kwargs.get('ib_weight', 0.1)
        self.log_ib_weight = nn.Parameter(torch.log(torch.tensor(initial_ib_weight)))
        
        logger.info(f"✓ 蒸馏参数初始化:")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  KD Weight: {initial_kd_weight}")
        logger.info(f"  Feat Weight: {initial_feat_weight}")
        logger.info(f"  IB Weight: {initial_ib_weight}")
        # ===========================
        
        if teacher_model is not None:
            # 冻结教师模型
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            
            logger.info("✓ 教师模型已冻结")
        
        # ===== 1. 视觉信息瓶颈（应用在纯视觉特征上）=====
        self.visual_ib = ImprovedVisualIB(
            input_dim=config.hidden_size,
            compression_ratio=0.7  # 保留70%的信息
        )
        logger.info(f"✓ 视觉IB模块: {config.hidden_size} -> {int(config.hidden_size * 0.7)}")
        
        # ===== 2. 多层特征蒸馏 =====
        if teacher_model is not None:
            self.multi_layer_distill = MultiLayerDistillation(
                student_hidden=config.hidden_size,
                teacher_hidden=teacher_model.config.hidden_size,
                num_student_layers=config.num_hidden_layers,
                num_teacher_layers=teacher_model.config.num_hidden_layers
            )
            logger.info("✓ 多层蒸馏模块已创建")
        else:
            self.multi_layer_distill = None
        
        # ===== 3. Logits投影层（如果维度不同）=====
        if teacher_model and config.hidden_size != teacher_model.config.hidden_size:
            self.logits_projector = nn.Linear(config.hidden_size, teacher_model.config.hidden_size)
            logger.info(f"✓ Logits投影: {config.hidden_size} -> {teacher_model.config.hidden_size}")
        else:
            self.logits_projector = None
        
        # 注意：损失权重已在 __init__ 开头初始化
    
    def encode_images_with_ib(self, images):
        """
        带信息瓶颈的图像编码
        IB应用在纯视觉特征上
        """
        # 1. 标准视觉编码（vision_tower + mm_projector）
        image_features = self.encode_images(images)
        
        # 2. 维度适配（如果需要）
        if image_features.shape[-1] != self.config.hidden_size:
            # 创建适配层（只在第一次调用时创建）
            if not hasattr(self, 'visual_adapter'):
                self.visual_adapter = nn.Linear(
                    image_features.shape[-1], 
                    self.config.hidden_size
                ).to(image_features.device).to(image_features.dtype)
                logger.info(f"创建视觉适配层: {image_features.shape[-1]} -> {self.config.hidden_size}")
            
            image_features = self.visual_adapter(image_features)
        
        # 3. 应用信息瓶颈压缩
        compressed_features, ib_info = self.visual_ib(image_features)
        
        return compressed_features, ib_info
    
    def encode_images(self, images):
        """
        覆盖父类的图像编码方法，统一兼容各种输入：
        - list 原图 / list 特征 -> 自动合批
        - 3D 特征 [B, L, D] -> 直接返回（D 可与 student hidden 不同，后续 IB 会适配）
        - 原图 Tensor [B, 3, H, W] -> 走视觉塔
        """
        if images is None:
            return None

        # 1) 已是特征：3D Tensor，直接返回（D 不用等于 self.config.hidden_size）
        if torch.is_tensor(images) and images.dim() == 3:
            return images

        # 2) list 输入：先合批成 Tensor
        if isinstance(images, list):
            if len(images) == 0:
                raise ValueError("images list is empty in encode_images().")
            first = images[0]
            if not torch.is_tensor(first):
                raise TypeError(f"Unsupported element type in images list: {type(first)}")

            # 检查list中元素类型一致性
            first_dim = first.dim()
            for i, im in enumerate(images):
                if im.dim() != first_dim:
                    raise ValueError(
                        f"Mixed tensor dimensions in images list: "
                        f"element 0 has dim={first_dim}, element {i} has dim={im.dim()}. "
                        f"Cannot mix raw images [3,H,W] with features [L,D]."
                    )

            # 2.a) list 原图：每个 [3,H,W] -> stack 到 [B,3,H,W]
            if first.dim() == 3:
                device = first.device
                dtype = first.dtype
                images = torch.stack([im.to(device=device, dtype=dtype) for im in images], dim=0)

            # 2.b) list 特征：每个 [L,D]（L 可能不同）-> pad 到同一 L 后 stack 成 [B,L_max,D]
            elif first.dim() == 2:
                D = first.size(-1)
                L_max = max(im.size(0) for im in images)
                device = first.device
                dtype = first.dtype
                padded = []
                
                # 记录padding mask（True表示真实token，False表示padding）
                feature_masks = []
                
                for im in images:
                    if im.size(-1) != D:
                        raise ValueError("All feature elements in images list must share the same last dim D.")
                    L = im.size(0)
                    
                    # 创建mask
                    mask = torch.ones(L_max, dtype=torch.bool, device=device)
                    
                    if L < L_max:
                        # Padding
                        pad = torch.zeros(L_max - L, D, device=device, dtype=dtype)
                        im = torch.cat([im, pad], dim=0)
                        # 标记padding位置为False
                        mask[L:] = False
                    
                    padded.append(im.to(device=device, dtype=dtype))
                    feature_masks.append(mask)
                
                images = torch.stack(padded, dim=0)
                
                # 保存padding mask供后续使用（如feature KD时的mask）
                if hasattr(self, '_last_feature_padding_mask'):
                    self._last_feature_padding_mask = torch.stack(feature_masks, dim=0)
                else:
                    # 第一次创建，设置为buffer（不会被保存到state_dict）
                    self.register_buffer(
                        '_last_feature_padding_mask',
                        torch.stack(feature_masks, dim=0),
                        persistent=False
                    )
                
                logger.debug(f"List特征padding: {len(images)}个样本, L_max={L_max}, "
                           f"真实tokens: {feature_masks[0].sum().item()}/{L_max}")
            else:
                raise TypeError(f"Unsupported tensor shape in images list: dim={first.dim()}")

            # 若上面已合成 3D 特征，直接返回；若是 4D 原图，继续走下面视觉塔
            if images.dim() == 3:
                return images

        # 3) 到这里：应是原图 Tensor [B,3,H,W]，准备走视觉塔
        # 一些 LLaVA 版本的 CLIPVisionTower 懒加载，需要先 ensure load
        try:
            vt = self.get_model().get_vision_tower()
        except Exception:
            vt = None
        if vt is not None and hasattr(vt, "load_model") and getattr(vt, "vision_tower", None) is None:
            vt.load_model()  # 懒加载，使 vt.vision_tower 生效

        # 4) 调用父类标准流程（vision_tower -> mm_projector）
        image_features = super().encode_images(images)
        
        # 5) 【关键】维度适配：teacher vision输出4096，student需要3072
        # 这个投影层在训练和eval时都需要！
        if image_features is not None and image_features.shape[-1] != self.config.hidden_size:
            if not hasattr(self, 'vision_dim_projector'):
                # 创建投影层
                self.vision_dim_projector = nn.Linear(
                    image_features.shape[-1],    # in = 4096 (teacher projector输出)
                    self.config.hidden_size,     # out = 3072 (student hidden)
                    bias=False
                ).to(image_features.device).to(image_features.dtype)
                
                # 截断单位矩阵初始化：保留前k个通道的信息
                with torch.no_grad():
                    self.vision_dim_projector.weight.zero_()
                    k = min(self.vision_dim_projector.out_features,
                            self.vision_dim_projector.in_features)
                    self.vision_dim_projector.weight[:k, :k].copy_(
                        torch.eye(k, device=image_features.device, dtype=image_features.dtype)
                    )
                
                logger.info(f"✨ 创建vision维度投影: {image_features.shape[-1]} -> {self.config.hidden_size}")
                logger.info(f"   截断单位初始化: 保留前{k}个通道")
            
            image_features = self.vision_dim_projector(image_features)
        
        return image_features
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
    ):
        """
        优化的前向传播
        """
        
        # ===== 推理模式：统一走IB路径（内部已做4096->3072投影）=====
        if not self.training or self.teacher_model is None:
            # 如果有图像，统一通过encode_images_with_ib处理
            if images is not None:
                # 总是走IB路径（内部会先调encode_images做维度投影，再做IB压缩）
                compressed_images, _ = self.encode_images_with_ib(images)
                
                # ⭐ 准备inputs_embeds，接住返回的labels（对齐图像token位置）
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    _,                  # past_key_values占位
                    inputs_embeds,
                    labels              # ⭐ 接住对齐后的labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    None,
                    labels,             # ⭐ 传入原始labels
                    compressed_images,
                    image_sizes
                )
                
                # 清空images，避免父类重复处理
                images = None
                image_sizes = None
            
            # ⭐ 调用父类，使用对齐后的labels
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,          # ⭐ 使用对齐后的labels
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                images=None,            # 已处理
                image_sizes=None,       # 已处理
                return_dict=True
            )
        
        
        # ===== 训练模式：完整蒸馏 =====
        
        # 保存原始输入（用于教师模型）
        original_input_ids = input_ids.clone() if input_ids is not None else None
        original_attention_mask = attention_mask.clone() if attention_mask is not None else None
        original_position_ids = position_ids.clone() if position_ids is not None else None
        original_labels = labels.clone() if labels is not None else None
        
        # ===== 学生路径：使用压缩的视觉特征 =====
        ib_info = None
        if images is not None:
            # 应用IB压缩
            compressed_images, ib_info = self.encode_images_with_ib(images)
            
            # 准备多模态输入（使用压缩后的视觉特征）
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                compressed_images,  # 使用压缩后的特征
                image_sizes
            )
        elif inputs_embeds is None:
            # 纯文本，准备输入
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                None,
                None
            )
        
        # 学生模型前向
        student_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 需要hidden states
            return_dict=True
        )
        
        # 初始化损失
        task_loss = student_outputs.loss if student_outputs.loss is not None else 0
        total_loss = task_loss
        
        # 损失记录
        loss_info = {
            'task_loss': task_loss.item() if torch.is_tensor(task_loss) else task_loss,
            'kl_loss': 0,
            'feature_loss': 0,
            'ib_loss': 0,
            'layer_0_loss': 0,
            'layer_1_loss': 0,
            'layer_2_loss': 0,
        }
        
        # ===== 1. 信息瓶颈损失 =====
        if ib_info is not None:
            ib_weight = torch.exp(self.log_ib_weight)
            ib_loss = ib_info['ib_loss']
            total_loss = total_loss + ib_weight * ib_loss
            
            loss_info['ib_loss'] = ib_loss.item()
            loss_info['ib_kl'] = ib_info['kl_loss'].item()
            loss_info['ib_recon'] = ib_info['recon_loss'].item()
            loss_info['ib_beta'] = ib_info['beta'].item()
            loss_info['ib_weight'] = ib_weight.item()
        
        # ===== 2. 知识蒸馏 =====
        if original_labels is not None:
            with torch.no_grad():
                # 教师模型前向（不传 labels，节省计算）
                teacher_outputs = self.teacher_model(
                    input_ids=original_input_ids,
                    attention_mask=original_attention_mask,
                    position_ids=original_position_ids,
                    images=images,  # 教师使用原始图像
                    image_sizes=image_sizes,
                    labels=None,  # ← 不传 labels，只需要 logits 和 hidden_states
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # 2.1 Logits蒸馏（KL散度）—— 修复：按有效 token 平均
            temperature = self.temperature  # 使用初始化时的温度参数
            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits
            
            # 对齐序列长度
            min_len = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_len, :]
            teacher_logits = teacher_logits[:, :min_len, :]
            
            # 如果维度不同，投影学生logits
            if self.logits_projector is not None:
                # 需要先投影到最后一层hidden，再到vocab
                # 这里简化：直接在logits空间对齐（假设vocab相同）
                pass
            
            # ===== 关键修复：按有效 token 平均 =====
            # 有效 token mask（忽略 -100）
            valid_mask = (original_labels[:, :min_len] != -100).float()  # [B, S]
            num_valid = valid_mask.sum().clamp_min(1.0)
            
            # 计算 KL（逐 token），然后只在有效位置做平均
            s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
            t_p = F.softmax(teacher_logits.float() / temperature, dim=-1).detach()
            
            kl_per_tok = F.kl_div(s_logp, t_p, reduction='none').sum(dim=-1)  # [B, S]
            kl_mean = (kl_per_tok * valid_mask).sum() / num_valid  # 标量
            
            kl_loss = (temperature ** 2) * kl_mean
            
            kd_weight = torch.exp(self.log_kd_weight)
            total_loss = total_loss + kd_weight * kl_loss
            
            loss_info['kl_loss'] = kl_loss.item()
            loss_info['kd_weight'] = kd_weight.item()
            loss_info['kl_per_token_mean'] = kl_per_tok.mean().item()  # 健康指标
            
            # 2.2 多层特征蒸馏（余弦相似度）
            # ✨ 这里的align_hidden会自动处理维度不匹配
            if self.multi_layer_distill is not None:
                feature_loss, layer_losses = self.multi_layer_distill(
                    student_outputs.hidden_states,
                    teacher_outputs.hidden_states,
                    attention_mask=attention_mask
                )
                
                feat_weight = torch.exp(self.log_feat_weight)
                total_loss = total_loss + feat_weight * feature_loss
                
                loss_info['feature_loss'] = feature_loss.item()
                loss_info['feat_weight'] = feat_weight.item()
                loss_info.update(layer_losses)
        
        # ===== 返回 =====
        output = CausalLMOutputWithPast(
            loss=total_loss,
            logits=student_outputs.logits,
            past_key_values=student_outputs.past_key_values,
            hidden_states=student_outputs.hidden_states,
            attentions=student_outputs.attentions,
        )
        
        output.loss_info = loss_info
        
        return output