import torch
import torch.nn as nn
import copy
from transformers import AutoConfig
from llava.model.language_model.llava_llama import LlavaConfig


def create_compressed_student_config(teacher_config, compression_ratio=0.5):
    """
    创建压缩后的学生模型配置
    新策略：
    - 固定16层（偶数层，对称性好）
    - hidden_size 和 heads 保持不变（避免embedding问题）
    - 只压缩 FFN (intermediate_size)
    - 保持所有 embedding 相关配置不变
    """
    student_config = copy.deepcopy(teacher_config)
    
    # 获取教师模型的原始参数
    teacher_layers = teacher_config.num_hidden_layers
    teacher_hidden = teacher_config.hidden_size
    teacher_intermediate = teacher_config.intermediate_size
    teacher_heads = teacher_config.num_attention_heads
    
    # ===== 新压缩策略：固定16层，只压缩FFN =====
    
    # 1. 层数固定为16（你的要求）
    target_layers = 24
    
    # 2. 根据 compression_ratio 调整 FFN 压缩程度
    if compression_ratio <= 0.3:
        ffn_ratio = 0.5    # FFN保留50%
    elif compression_ratio <= 0.5:
        ffn_ratio = 0.7    # FFN保留70%
    else:
        ffn_ratio = 0.85   # FFN保留85%
    
    student_config.num_hidden_layers = target_layers
    
    # 3. ===== 关键！hidden_size 和 heads 完全不变 =====
    student_config.hidden_size = teacher_hidden
    student_config.num_attention_heads = teacher_heads
    
    # 4. 只压缩 FFN
    new_intermediate = int(teacher_intermediate * ffn_ratio)
    new_intermediate = (new_intermediate // 256) * 256  # 对齐到256
    new_intermediate = max(512, new_intermediate)
    student_config.intermediate_size = new_intermediate
    
    # 5. GQA配置保持不变
    if hasattr(teacher_config, 'num_key_value_heads'):
        student_config.num_key_value_heads = teacher_config.num_key_value_heads
    
    # 6. ===== 保持所有 embedding 相关配置不变 =====
    if hasattr(teacher_config, 'vocab_size'):
        student_config.vocab_size = teacher_config.vocab_size
        print(f"✓ 保持 vocab_size: {teacher_config.vocab_size}")
    
    if hasattr(teacher_config, 'max_position_embeddings'):
        student_config.max_position_embeddings = teacher_config.max_position_embeddings
        print(f"✓ 保持 max_position_embeddings: {teacher_config.max_position_embeddings}")
    
    # 保持特殊token配置
    if hasattr(teacher_config, 'pad_token_id'):
        student_config.pad_token_id = teacher_config.pad_token_id
    if hasattr(teacher_config, 'bos_token_id'):
        student_config.bos_token_id = teacher_config.bos_token_id
    if hasattr(teacher_config, 'eos_token_id'):
        student_config.eos_token_id = teacher_config.eos_token_id
    
    # RoPE配置
    if hasattr(teacher_config, 'rope_theta'):
        student_config.rope_theta = teacher_config.rope_theta
    if hasattr(teacher_config, 'rope_scaling'):
        student_config.rope_scaling = teacher_config.rope_scaling
    
    # 7. 标记
    student_config.compression_ratio = compression_ratio
    student_config.use_information_bottleneck = True
    
    # 8. 打印配置
    print(f"\n{'='*60}")
    print("Student Model Configuration:")
    print(f"{'='*60}")
    print(f"Layers: {teacher_layers} -> {student_config.num_hidden_layers} "
          f"({student_config.num_hidden_layers/teacher_layers:.0%})")
    print(f"Hidden: {teacher_hidden} (保持不变)")
    print(f"FFN: {teacher_intermediate} -> {new_intermediate} "
          f"({new_intermediate/teacher_intermediate:.0%})")
    print(f"Heads: {teacher_heads} (保持不变)")
    print(f"Vocab: {teacher_config.vocab_size} (保持不变)")
    
    # 估算压缩
    teacher_params_est = estimate_model_params(teacher_config)
    student_params_est = estimate_model_params(student_config)
    actual_compression = student_params_est / teacher_params_est
    
    print(f"\n参数量: {teacher_params_est/1e9:.2f}B -> {student_params_est/1e9:.2f}B")
    print(f"实际压缩率: {actual_compression:.2%}")
    print(f"参数减少: {(teacher_params_est - student_params_est)/1e9:.2f}B")
    print(f"{'='*60}\n")
    
    return student_config


def estimate_model_params(config):
    """估算模型参数量（更精确）"""
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
    # LayerNorms (2 per layer)
    per_layer += 4 * hidden
    
    params += layers * per_layer
    
    # LM head
    params += vocab_size * hidden
    
    return params


def initialize_student_from_teacher(student_model, teacher_model, student_config):
    """
    从教师模型初始化学生模型权重
    由于hidden_size相同，大部分权重可以直接复制
    """
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    
    initialized = 0
    skipped = 0
    truncated = 0
    
    teacher_layers = teacher_model.config.num_hidden_layers
    student_layers = student_config.num_hidden_layers
    
    # 均匀选择要保留的层（0, 2, 4, 6, ..., 30 → 0-15）
    layer_indices = torch.linspace(0, teacher_layers-1, student_layers).long().tolist()
    layer_map = {i: layer_indices[i] for i in range(student_layers)}
    
    print(f"\nLayer mapping (student -> teacher):")
    print(f"  {layer_map}")
    
    for name, param in student_state.items():
        # 跳过特殊模块
        if 'visual_compressor' in name or 'visual_decoder' in name or 'feature_projector' in name or 'beta' in name:
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
                            # 完全匹配，直接复制
                            param.data.copy_(teacher_param.data)
                            initialized += 1
                        else:
                            # 维度不匹配（主要是FFN）
                            if init_param_with_mismatch(param, teacher_param, name):
                                truncated += 1
                            else:
                                skipped += 1
        
        # 处理 embedding、lm_head 等（应该完全匹配）
        elif name in teacher_state:
            teacher_param = teacher_state[name]
            
            if param.shape == teacher_param.shape:
                param.data.copy_(teacher_param.data)
                initialized += 1
            else:
                if init_param_with_mismatch(param, teacher_param, name):
                    truncated += 1
                else:
                    skipped += 1
    
    print(f"\nWeight initialization:")
    print(f"  ✓ Exact match: {initialized}")
    print(f"  ✓ Truncated/Padded: {truncated}")
    print(f"  - Skipped: {skipped}")
    print(f"  Total params: {len(student_state)}")
    
    return student_model


def init_param_with_mismatch(student_param, teacher_param, name):
    """
    处理维度不匹配的参数（主要是FFN的权重）
    """
    s_shape = student_param.shape
    t_shape = teacher_param.shape
    
    if len(s_shape) != len(t_shape):
        return False
    
    try:
        if len(s_shape) == 1:
            # 一维向量
            min_size = min(s_shape[0], t_shape[0])
            student_param.data[:min_size] = teacher_param.data[:min_size]
            
        elif len(s_shape) == 2:
            # 二维矩阵（Linear层）
            min_rows = min(s_shape[0], t_shape[0])
            min_cols = min(s_shape[1], t_shape[1])
            student_param.data[:min_rows, :min_cols] = teacher_param.data[:min_rows, :min_cols]
            
            # 如果学生更大，用xavier初始化剩余部分
            if s_shape[0] > t_shape[0] or s_shape[1] > t_shape[1]:
                nn.init.xavier_uniform_(student_param.data)
                student_param.data[:min_rows, :min_cols] = teacher_param.data[:min_rows, :min_cols]
        else:
            # 更高维
            slices = tuple(slice(0, min(s, t)) for s, t in zip(s_shape, t_shape))
            student_param.data[slices] = teacher_param.data[slices]
        
        return True
    except Exception as e:
        print(f"Failed to initialize {name}: {e}")
        return False


def analyze_compression(teacher_model, student_model):
    """分析压缩效果"""
    # 教师模型参数
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    
    # ===== 修复：只统计学生模型本身的参数，排除teacher和IB模块 =====
    student_params = 0
    for name, param in student_model.named_parameters():
        # 跳过teacher_model的参数
        if 'teacher_model' in name:
            continue
        # 跳过IB相关模块（这些是额外添加的）
        if any(skip in name for skip in ['visual_compressor', 'visual_decoder', 'log_beta', 'feature_projector']):
            continue
        student_params += param.numel()
    # ===== 修复结束 =====
    
    # 或者更简单：直接统计核心transformer的参数
    # student_params = sum(p.numel() for n, p in student_model.named_parameters() 
    #                     if 'model.layers' in n or 'model.embed_tokens' in n or 'lm_head' in n)
    
    teacher_layer_params = {}
    student_layer_params = {}
    
    for name, param in teacher_model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in teacher_layer_params:
            teacher_layer_params[layer_type] = 0
        teacher_layer_params[layer_type] += param.numel()
    
    for name, param in student_model.named_parameters():
        # 跳过teacher和IB模块
        if 'teacher_model' in name or any(skip in name for skip in ['visual_compressor', 'visual_decoder', 'log_beta', 'feature_projector']):
            continue
            
        layer_type = name.split('.')[0]
        if layer_type not in student_layer_params:
            student_layer_params[layer_type] = 0
        student_layer_params[layer_type] += param.numel()
    
    print(f"\n{'='*60}")
    print("Compression Analysis:")
    print(f"{'='*60}")
    print(f"Total parameters:")
    print(f"  Teacher: {teacher_params/1e9:.2f}B ({teacher_params/1e6:.2f}M)")
    print(f"  Student: {student_params/1e9:.2f}B ({student_params/1e6:.2f}M)")
    print(f"  Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    print(f"\nPer-module compression:")
    for module in sorted(teacher_layer_params.keys()):
        if module in student_layer_params:
            t_params = teacher_layer_params[module]
            s_params = student_layer_params[module]
            reduction = (1 - s_params/t_params) * 100
            print(f"  {module}: {t_params/1e6:.1f}M -> {s_params/1e6:.1f}M ({reduction:.1f}% reduced)")
    
    print(f"{'='*60}\n")
    
    return {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': student_params / teacher_params,
        'reduction_percentage': (1 - student_params/teacher_params) * 100
    }
