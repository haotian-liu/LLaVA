"""
改进的学生模型创建
支持更激进的压缩策略：hidden_size + heads + FFN + layers
"""

import torch
import torch.nn as nn
import copy
from transformers import AutoConfig
from llava.model.language_model.llava_llama import LlavaConfig
import logging

logger = logging.getLogger(__name__)


def create_compressed_student_config_v2(
    teacher_config, 
    compression_ratio=0.5,
    compress_hidden=False,  # 是否压缩hidden_size
    compress_heads=False    # 是否压缩attention heads
):
    """
    创建压缩后的学生模型配置 V2
    
    两种策略：
    1. 保守策略（compress_hidden=False）：只压缩层数和FFN
    2. 激进策略（compress_hidden=True）：额外压缩hidden_size和heads
    
    Args:
        teacher_config: 教师模型配置
        compression_ratio: 目标压缩率（0.3-0.7）
        compress_hidden: 是否压缩hidden_size
        compress_heads: 是否压缩attention heads
    """
    student_config = copy.deepcopy(teacher_config)
    
    # 教师参数
    teacher_layers = teacher_config.num_hidden_layers
    teacher_hidden = teacher_config.hidden_size
    teacher_intermediate = teacher_config.intermediate_size
    teacher_heads = teacher_config.num_attention_heads
    
    logger.info(f"\n{'='*60}")
    logger.info("创建学生模型配置 V2")
    logger.info(f"{'='*60}")
    logger.info(f"目标压缩率: {compression_ratio}")
    logger.info(f"压缩hidden_size: {compress_hidden}")
    logger.info(f"压缩heads: {compress_heads}")
    
    # ===== 策略1: 保守（只压缩层数和FFN）=====
    if not compress_hidden:
        logger.info("\n使用保守策略：只压缩层数和FFN")
        
        # 层数固定为16
        student_config.num_hidden_layers = 16
        
        # hidden和heads保持不变
        student_config.hidden_size = teacher_hidden
        student_config.num_attention_heads = teacher_heads
        
        # FFN根据compression_ratio调整
        if compression_ratio <= 0.3:
            ffn_ratio = 0.5
        elif compression_ratio <= 0.5:
            ffn_ratio = 0.7
        else:
            ffn_ratio = 0.85
        
        new_intermediate = int(teacher_intermediate * ffn_ratio)
        new_intermediate = (new_intermediate // 256) * 256  # 对齐
        student_config.intermediate_size = max(512, new_intermediate)
        
        logger.info(f"  Layers: {teacher_layers} -> {student_config.num_hidden_layers}")
        logger.info(f"  Hidden: {teacher_hidden} (不变)")
        logger.info(f"  Heads: {teacher_heads} (不变)")
        logger.info(f"  FFN: {teacher_intermediate} -> {student_config.intermediate_size}")
    
    # ===== 策略2: 激进（压缩所有维度）=====
    else:
        logger.info("\n使用激进策略：压缩所有维度")
        
        # 1. 层数
        if compression_ratio <= 0.3:
            target_layers = 16
        elif compression_ratio <= 0.5:
            target_layers = 24
        else:
            target_layers = 20
        
        student_config.num_hidden_layers = target_layers
        
        # 2. Hidden size压缩
        hidden_ratio = 0.75 if compression_ratio <= 0.5 else 0.875
        new_hidden = int(teacher_hidden * hidden_ratio)
        new_hidden = (new_hidden // 128) * 128  # 对齐到128
        student_config.hidden_size = new_hidden
        
        # 3. Heads压缩（保持head_dim不变）
        if compress_heads:
            head_dim = teacher_hidden // teacher_heads
            new_heads = new_hidden // head_dim
            new_heads = max(8, new_heads)  # 至少8个head
            student_config.num_attention_heads = new_heads
        else:
            # 调整heads以匹配新的hidden_size
            head_dim = teacher_hidden // teacher_heads
            new_heads = new_hidden // head_dim
            student_config.num_attention_heads = max(8, new_heads)
        
        # 4. FFN压缩
        ffn_ratio = 0.7 if compression_ratio <= 0.5 else 0.85
        new_intermediate = int(teacher_intermediate * ffn_ratio)
        new_intermediate = (new_intermediate // 256) * 256
        student_config.intermediate_size = max(512, new_intermediate)
        
        # 5. KV heads（GQA）
        if hasattr(teacher_config, 'num_key_value_heads'):
            kv_ratio = new_heads / teacher_heads
            new_kv_heads = max(1, int(teacher_config.num_key_value_heads * kv_ratio))
            student_config.num_key_value_heads = new_kv_heads
            logger.info(f"  KV Heads: {teacher_config.num_key_value_heads} -> {new_kv_heads}")
        
        logger.info(f"  Layers: {teacher_layers} -> {student_config.num_hidden_layers}")
        logger.info(f"  Hidden: {teacher_hidden} -> {student_config.hidden_size}")
        logger.info(f"  Heads: {teacher_heads} -> {student_config.num_attention_heads}")
        logger.info(f"  FFN: {teacher_intermediate} -> {student_config.intermediate_size}")
    
    # ===== 保持Embedding配置不变 =====
    if hasattr(teacher_config, 'vocab_size'):
        student_config.vocab_size = teacher_config.vocab_size
    
    if hasattr(teacher_config, 'max_position_embeddings'):
        student_config.max_position_embeddings = teacher_config.max_position_embeddings
    
    # 保持特殊token
    for attr in ['pad_token_id', 'bos_token_id', 'eos_token_id']:
        if hasattr(teacher_config, attr):
            setattr(student_config, attr, getattr(teacher_config, attr))
    
    # RoPE配置
    if hasattr(teacher_config, 'rope_theta'):
        student_config.rope_theta = teacher_config.rope_theta
    if hasattr(teacher_config, 'rope_scaling'):
        student_config.rope_scaling = teacher_config.rope_scaling
    
    # 标记
    student_config.compression_ratio = compression_ratio
    student_config.use_information_bottleneck = True
    student_config.compressed_hidden = compress_hidden
    
    # ===== 估算参数量 =====
    teacher_params = estimate_model_params(teacher_config)
    student_params = estimate_model_params(student_config)
    actual_compression = student_params / teacher_params
    
    logger.info(f"\n参数量估算:")
    logger.info(f"  教师: {teacher_params/1e9:.2f}B")
    logger.info(f"  学生: {student_params/1e9:.2f}B")
    logger.info(f"  实际压缩率: {actual_compression:.2%}")
    logger.info(f"  参数减少: {(teacher_params - student_params)/1e9:.2f}B")
    logger.info(f"{'='*60}\n")
    
    return student_config


def estimate_model_params(config):
    """估算模型参数量"""
    params = 0
    vocab_size = getattr(config, 'vocab_size', 32000)
    hidden = config.hidden_size
    layers = config.num_hidden_layers
    intermediate = config.intermediate_size
    
    # Embeddings
    params += vocab_size * hidden
    
    # Transformer layers
    per_layer = 0
    # Attention (Q,K,V,O)
    per_layer += 4 * hidden * hidden
    # MLP (gate, up, down)
    per_layer += 3 * hidden * intermediate
    # LayerNorms
    per_layer += 4 * hidden
    
    params += layers * per_layer
    
    # LM head
    params += vocab_size * hidden
    
    return params


def initialize_student_from_teacher_v2(student_model, teacher_model, student_config):
    """
    改进的权重初始化
    支持hidden_size不匹配的情况
    """
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    
    initialized = 0
    truncated = 0
    skipped = 0
    
    teacher_layers = teacher_model.config.num_hidden_layers
    student_layers = student_config.num_hidden_layers
    teacher_hidden = teacher_model.config.hidden_size
    student_hidden = student_config.hidden_size
    
    hidden_mismatch = (teacher_hidden != student_hidden)
    
    # 层映射（均匀采样）
    layer_indices = torch.linspace(0, teacher_layers-1, student_layers).long().tolist()
    layer_map = {i: layer_indices[i] for i in range(student_layers)}
    
    logger.info(f"\n权重初始化:")
    logger.info(f"  层映射: {layer_map}")
    logger.info(f"  Hidden维度: {teacher_hidden} -> {student_hidden}")
    
    for name, param in student_state.items():
        # 跳过额外模块
        if any(skip in name for skip in ['visual_ib', 'multi_layer_distill', 'logits_projector', 'log_']):
            skipped += 1
            continue
        
        # 处理transformer层
        if 'model.layers' in name or 'layers.' in name:
            import re
            layer_match = re.search(r'layers\.(\d+)\.', name)
            if layer_match:
                student_layer_idx = int(layer_match.group(1))
                
                if student_layer_idx in layer_map:
                    teacher_layer_idx = layer_map[student_layer_idx]
                    teacher_name = name.replace(
                        f'layers.{student_layer_idx}.',
                        f'layers.{teacher_layer_idx}.'
                    )
                    
                    if teacher_name in teacher_state:
                        teacher_param = teacher_state[teacher_name]
                        
                        if param.shape == teacher_param.shape:
                            # 完全匹配
                            param.data.copy_(teacher_param.data)
                            initialized += 1
                        elif hidden_mismatch:
                            # Hidden维度不匹配，需要智能截断/填充
                            if init_param_with_hidden_mismatch(param, teacher_param, name, student_hidden, teacher_hidden):
                                truncated += 1
                            else:
                                skipped += 1
                        else:
                            # FFN等其他维度不匹配
                            if init_param_with_mismatch(param, teacher_param, name):
                                truncated += 1
                            else:
                                skipped += 1
                else:
                    skipped += 1
        
        # 处理embedding、lm_head等
        elif name in teacher_state:
            teacher_param = teacher_state[name]
            
            if param.shape == teacher_param.shape:
                param.data.copy_(teacher_param.data)
                initialized += 1
            elif hidden_mismatch and 'embed' in name:
                # Embedding层的hidden维度不匹配
                if init_param_with_hidden_mismatch(param, teacher_param, name, student_hidden, teacher_hidden):
                    truncated += 1
                else:
                    skipped += 1
            elif 'lm_head' in name:
                # LM head: [vocab, hidden]
                if init_param_with_hidden_mismatch(param, teacher_param, name, student_hidden, teacher_hidden):
                    truncated += 1
                else:
                    skipped += 1
            else:
                if init_param_with_mismatch(param, teacher_param, name):
                    truncated += 1
                else:
                    skipped += 1
        else:
            skipped += 1
    
    logger.info(f"\n初始化统计:")
    logger.info(f"  ✓ 完全匹配: {initialized}")
    logger.info(f"  ✓ 截断/填充: {truncated}")
    logger.info(f"  - 跳过: {skipped}")
    logger.info(f"  总参数: {len(student_state)}")
    
    return student_model


def init_param_with_hidden_mismatch(student_param, teacher_param, name, student_hidden, teacher_hidden):
    """
    处理hidden_size不匹配的参数
    
    策略：
    - 如果学生更小：截断教师权重
    - 如果学生更大：用xavier初始化，然后复制教师权重到前面
    """
    s_shape = student_param.shape
    t_shape = teacher_param.shape
    
    try:
        if len(s_shape) == 1:
            # 一维（bias或layernorm）
            min_size = min(s_shape[0], t_shape[0])
            
            if student_hidden < teacher_hidden:
                # 截断
                student_param.data[:min_size] = teacher_param.data[:min_size]
            else:
                # 扩展：xavier + 复制
                nn.init.xavier_uniform_(student_param.data.unsqueeze(0)).squeeze(0)
                student_param.data[:min_size] = teacher_param.data[:min_size]
        
        elif len(s_shape) == 2:
            # 二维（Linear层）
            # 判断哪个维度是hidden
            if 'embed_tokens' in name or 'lm_head' in name:
                # [vocab, hidden] 或 [hidden, vocab]
                if s_shape[1] == student_hidden:
                    # [vocab, hidden]
                    min_hidden = min(s_shape[1], t_shape[1])
                    min_vocab = min(s_shape[0], t_shape[0])
                    
                    nn.init.xavier_uniform_(student_param.data)
                    student_param.data[:min_vocab, :min_hidden] = teacher_param.data[:min_vocab, :min_hidden]
                else:
                    # [hidden, vocab]
                    min_hidden = min(s_shape[0], t_shape[0])
                    min_vocab = min(s_shape[1], t_shape[1])
                    
                    nn.init.xavier_uniform_(student_param.data)
                    student_param.data[:min_hidden, :min_vocab] = teacher_param.data[:min_hidden, :min_vocab]
            else:
                # 通用Linear: [out, in] 其中in或out可能是hidden
                min_rows = min(s_shape[0], t_shape[0])
                min_cols = min(s_shape[1], t_shape[1])
                
                nn.init.xavier_uniform_(student_param.data)
                student_param.data[:min_rows, :min_cols] = teacher_param.data[:min_rows, :min_cols]
        
        return True
    except Exception as e:
        logger.warning(f"初始化失败 {name}: {e}")
        return False


def init_param_with_mismatch(student_param, teacher_param, name):
    """处理其他维度不匹配（如FFN）"""
    s_shape = student_param.shape
    t_shape = teacher_param.shape
    
    if len(s_shape) != len(t_shape):
        return False
    
    try:
        if len(s_shape) == 1:
            min_size = min(s_shape[0], t_shape[0])
            student_param.data[:min_size] = teacher_param.data[:min_size]
        elif len(s_shape) == 2:
            min_rows = min(s_shape[0], t_shape[0])
            min_cols = min(s_shape[1], t_shape[1])
            
            nn.init.xavier_uniform_(student_param.data)
            student_param.data[:min_rows, :min_cols] = teacher_param.data[:min_rows, :min_cols]
        else:
            slices = tuple(slice(0, min(s, t)) for s, t in zip(s_shape, t_shape))
            student_param.data[slices] = teacher_param.data[slices]
        
        return True
    except Exception as e:
        logger.warning(f"初始化失败 {name}: {e}")
        return False