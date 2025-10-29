# cmibq/models/quantized_llava_wrapper_complete.py
"""
CM-IBQ Quantization Wrapper for LLaVA - Complete Version
完整实现激活值和权重的混合精度量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import warnings
import sys
import os
from pathlib import Path
import gc

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# 导入量化模块
from ..core.ib_quantizer import IBQuantizedLayer
from ..core.importance_estimator import HybridImportanceEstimation
from ..core.bit_allocator import BitAllocationNetwork
from ..core.differentiable_quant import DifferentiableQuantizer
from ..core.weight_quantizer import IBWeightQuantizer, WeightQuantizer, QuantizedLinear
from .lora_adapter import LoRAAdapter


class CMIBQQuantizedLLaVA(nn.Module):
    """
    完整版LLaVA量化wrapper
    同时支持激活值和权重的混合精度量化
    """
    
    def __init__(
        self,
        model_path: str,
        target_bits_act: float = 4.0,
        target_bits_weight: float = 4.0,
        use_ib: bool = True,
        use_lora: bool = True,
        lora_rank: int = 16,
        num_groups: int = 8,
        stage: int = 1,
        quantize_weights: bool = True,
        weight_quant_mode: str = 'mixed',  # 'uniform' or 'mixed'
        model_base: Optional[str] = None,
        llm_layer_interval: int = 2,  # LLM层量化间隔
        quantize_vision_embeddings: bool = False,
        quantize_lm_head: bool = False,
        skip_modules: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__()
        
        self.target_bits_act = target_bits_act
        self.target_bits_weight = target_bits_weight
        self.use_ib = use_ib
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.num_groups = num_groups
        self.stage = stage
        self.quantize_weights = quantize_weights
        self.weight_quant_mode = weight_quant_mode
        self.llm_layer_interval = llm_layer_interval
        self.quantize_vision_embeddings = quantize_vision_embeddings
        self.quantize_lm_head = quantize_lm_head
        self.skip_modules = skip_modules or []
        
        # 加载原始模型
        print(f"Loading base model from {model_path}...")
        self.tokenizer, self.base_model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=get_model_name_from_path(model_path)
        )
        
        # 提取维度
        self._extract_dimensions()
        
        # 初始化激活量化模块
        self._init_activation_quantization_modules()
        
        # 初始化权重量化模块
        if self.quantize_weights:
            self._init_weight_quantization_modules()
        
        # Stage 2: 初始化对齐模块
        if self.stage == 2:
            self._init_alignment_modules()
        
        # 设置hooks
        self._setup_quantization_hooks()
        
        # 添加LoRA（如果启用且在Stage 2）
        if self.use_lora and self.stage == 2:
            self._setup_lora_adapters()
        
        # 应用内存优化
        self._apply_memory_optimizations()
        
        # 打印量化统计
        self._print_quantization_summary()
    
    def _extract_dimensions(self):
        """提取模型维度信息"""
        # Vision维度
        try:
            vision_tower = self.base_model.get_vision_tower()
            if vision_tower:
                if hasattr(vision_tower, 'hidden_size'):
                    self.vision_dim = vision_tower.hidden_size
                elif hasattr(vision_tower, 'config') and hasattr(vision_tower.config, 'hidden_size'):
                    self.vision_dim = vision_tower.config.hidden_size
                else:
                    self.vision_dim = 1024  # CLIP默认
            else:
                self.vision_dim = 1024
        except:
            self.vision_dim = 1024
        
        # LLM维度
        try:
            if hasattr(self.base_model, 'config'):
                self.llm_dim = self.base_model.config.hidden_size
            elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'config'):
                self.llm_dim = self.base_model.model.config.hidden_size
            else:
                self.llm_dim = 4096  # 7B默认
        except:
            self.llm_dim = 4096
        
        # 投影器维度
        self.projection_input_dim = self.vision_dim
        self.projection_output_dim = self.llm_dim
        
        print(f"Dimensions - Vision: {self.vision_dim}, LLM: {self.llm_dim}")
    
    def _init_activation_quantization_modules(self):
        """初始化激活量化模块"""
        self.activation_quantizers = nn.ModuleDict()
        
        # 1. Vision Tower激活量化
        self.activation_quantizers['vision'] = nn.ModuleDict({
            'importance': HybridImportanceEstimation(
                feature_dim=self.vision_dim,
                num_groups=self.num_groups
            ),
            'bit_allocator': BitAllocationNetwork(
                feature_dim=self.vision_dim,
                num_groups=self.num_groups,
                target_bits=self.target_bits_act,
                bit_levels=[2, 4, 8]
            ),
            'quantizer': DifferentiableQuantizer(
                feature_dim=self.vision_dim,
                per_sample=True,
                bit_levels=[2, 4, 8],
                target_budget=self.target_bits_act
            )
        })
        
        # 2. 投影器激活量化（最关键）
        if self.use_ib:
            self.activation_quantizers['projector_ib'] = IBQuantizedLayer(
                input_dim=self.projection_input_dim,
                bottleneck_dim=self.projection_output_dim,
                target_bits=self.target_bits_act,
                adaptive_beta=True
            )
        
        self.activation_quantizers['projector'] = nn.ModuleDict({
            'importance': HybridImportanceEstimation(
                feature_dim=self.projection_output_dim,
                num_groups=self.num_groups
            ),
            'bit_allocator': BitAllocationNetwork(
                feature_dim=self.projection_output_dim,
                num_groups=self.num_groups,
                target_bits=self.target_bits_act,
                bit_levels=[2, 4, 8]
            ),
            'quantizer': DifferentiableQuantizer(
                feature_dim=self.projection_output_dim,
                per_sample=True,
                bit_levels=[2, 4, 8],
                target_budget=self.target_bits_act
            )
        })
        
        # 3. LLM层激活量化（选择性）
        self.activation_quantizers['llm'] = nn.ModuleDict({
            'importance': HybridImportanceEstimation(
                feature_dim=self.llm_dim,
                num_groups=self.num_groups
            ),
            'bit_allocator': BitAllocationNetwork(
                feature_dim=self.llm_dim,
                num_groups=self.num_groups,
                target_bits=self.target_bits_act,
                bit_levels=[2, 4, 8]
            ),
            'quantizer': DifferentiableQuantizer(
                feature_dim=self.llm_dim,
                per_sample=True,
                bit_levels=[2, 4, 8],
                target_budget=self.target_bits_act
            )
        })
    
    def _init_weight_quantization_modules(self):
        """初始化权重量化模块 - 完整实现"""
        print("\nInitializing weight quantization modules...")
        
        self.weight_quantizers = nn.ModuleDict()
        self.quantized_layers = {}  # 记录所有被量化的层
        
        # 1. Vision Tower权重量化
        print("  [1/3] Quantizing Vision Tower weights...")
        self._quantize_vision_tower_weights()
        
        # 2. Projector权重量化
        print("  [2/3] Quantizing Projector weights...")
        self._quantize_projector_weights()
        
        # 3. LLM层权重量化
        print("  [3/3] Quantizing LLM layer weights...")
        self._quantize_llm_weights()
        
        print(f"\n✓ Weight quantization completed: {len(self.quantized_layers)} layers quantized")
    
    def _quantize_vision_tower_weights(self):
        """量化Vision Tower的权重"""
        vision_tower = self.base_model.get_vision_tower()
        if not vision_tower:
            return
        
        quantized_count = 0
        
        # 遍历vision tower的所有层
        for name, module in vision_tower.named_modules():
            # 跳过不需要量化的模块
            if any(skip in name for skip in self.skip_modules):
                continue
            
            # 跳过embedding层（如果指定）
            if not self.quantize_vision_embeddings and 'embed' in name.lower():
                continue
            
            # 量化Linear和Conv2d层
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parent_name, attr_name = self._get_parent_and_attr(vision_tower, name)
                parent = self._get_module_by_name(vision_tower, parent_name)
                
                if isinstance(module, nn.Linear):
                    quantized_layer = self._create_quantized_linear(module)
                else:  # Conv2d
                    quantized_layer = self._create_quantized_conv2d(module)
                
                # 替换原始层
                setattr(parent, attr_name, quantized_layer)
                self.quantized_layers[f'vision.{name}'] = quantized_layer
                quantized_count += 1
        
        print(f"    Quantized {quantized_count} layers in Vision Tower")
        
        # 如果使用混合精度，创建IB量化器
        if self.weight_quant_mode == 'mixed' and quantized_count > 0:
            self._create_ib_weight_quantizer_for_vision()
    
    def _quantize_projector_weights(self):
        """量化Projector的权重"""
        model = self.base_model.get_model()
        if not hasattr(model, 'mm_projector'):
            return
        
        projector = model.mm_projector
        quantized_count = 0
        
        # 遍历projector的所有层
        for name, module in projector.named_modules():
            if isinstance(module, nn.Linear):
                parent_name, attr_name = self._get_parent_and_attr(projector, name)
                parent = self._get_module_by_name(projector, parent_name) if parent_name else projector
                
                quantized_layer = self._create_quantized_linear(module)
                
                # 替换原始层
                if parent_name:
                    setattr(parent, attr_name, quantized_layer)
                else:
                    # 如果projector本身就是Linear层
                    model.mm_projector = quantized_layer
                
                self.quantized_layers[f'projector.{name}' if name else 'projector'] = quantized_layer
                quantized_count += 1
        
        print(f"    Quantized {quantized_count} layers in Projector")
        
        # 如果使用混合精度，创建IB量化器
        if self.weight_quant_mode == 'mixed' and quantized_count > 0:
            self._create_ib_weight_quantizer_for_projector()
    
    def _quantize_llm_weights(self):
        """量化LLM层的权重 - 选择性量化以平衡精度"""
        model = self.base_model.get_model()
        if not hasattr(model, 'layers'):
            return
        
        num_layers = len(model.layers)
        # 选择要量化的层（根据间隔）
        layers_to_quantize = list(range(0, num_layers, self.llm_layer_interval))
        
        print(f"    Quantizing {len(layers_to_quantize)} out of {num_layers} LLM layers (interval={self.llm_layer_interval})")
        
        total_quantized = 0
        
        for layer_idx in layers_to_quantize:
            layer = model.layers[layer_idx]
            
            # 量化自注意力层
            if hasattr(layer, 'self_attn'):
                attn_quantized = self._quantize_attention_weights(
                    layer.self_attn, 
                    f'llm.layer_{layer_idx}.self_attn'
                )
                total_quantized += attn_quantized
            
            # 量化FFN层
            if hasattr(layer, 'mlp'):
                ffn_quantized = self._quantize_ffn_weights(
                    layer.mlp,
                    f'llm.layer_{layer_idx}.mlp'
                )
                total_quantized += ffn_quantized
        
        print(f"    Quantized {total_quantized} layers in LLM")
        
        # 量化lm_head（如果指定）
        if self.quantize_lm_head and hasattr(model, 'lm_head'):
            lm_head = model.lm_head
            if isinstance(lm_head, nn.Linear):
                quantized_layer = self._create_quantized_linear(lm_head)
                model.lm_head = quantized_layer
                self.quantized_layers['lm_head'] = quantized_layer
                print(f"    Quantized lm_head")
    
    def _quantize_attention_weights(self, attn_module, prefix):
        """量化注意力模块的权重"""
        quantized_count = 0
        
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn_module, proj_name):
                original_proj = getattr(attn_module, proj_name)
                
                if isinstance(original_proj, nn.Linear):
                    quantized_proj = self._create_quantized_linear(original_proj)
                    setattr(attn_module, proj_name, quantized_proj)
                    self.quantized_layers[f'{prefix}.{proj_name}'] = quantized_proj
                    quantized_count += 1
        
        return quantized_count
    
    def _quantize_ffn_weights(self, ffn_module, prefix):
        """量化FFN模块的权重"""
        quantized_count = 0
        
        # LLaMA风格的FFN
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(ffn_module, proj_name):
                original_proj = getattr(ffn_module, proj_name)
                
                if isinstance(original_proj, nn.Linear):
                    quantized_proj = self._create_quantized_linear(original_proj)
                    setattr(ffn_module, proj_name, quantized_proj)
                    self.quantized_layers[f'{prefix}.{proj_name}'] = quantized_proj
                    quantized_count += 1
        
        # 标准Transformer的FFN
        for proj_name in ['fc1', 'fc2']:
            if hasattr(ffn_module, proj_name):
                original_proj = getattr(ffn_module, proj_name)
                
                if isinstance(original_proj, nn.Linear):
                    quantized_proj = self._create_quantized_linear(original_proj)
                    setattr(ffn_module, proj_name, quantized_proj)
                    self.quantized_layers[f'{prefix}.{proj_name}'] = quantized_proj
                    quantized_count += 1
        
        return quantized_count
    
    def _create_quantized_linear(self, original_layer):
        """创建量化的Linear层"""
        # 获取原始层的设备
        device = original_layer.weight.device
        
        quantized_layer = QuantizedLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            num_bits=int(self.target_bits_weight),
            per_channel=True,
            symmetric=True,
            num_groups=self.num_groups,
            use_adaptive=(self.weight_quant_mode == 'mixed')
        )
        
        # 将量化层移到正确的设备
        quantized_layer = quantized_layer.to(device)
        
        # 复制权重
        quantized_layer.weight_fp.data = original_layer.weight.data.clone()
        if original_layer.bias is not None:
            quantized_layer.bias.data = original_layer.bias.data.clone()
        
        # 执行初始量化
        quantized_layer.quantize_weight()
        
        return quantized_layer
    
    def _create_quantized_conv2d(self, original_layer):
        """创建量化的Conv2d层（用于Vision Tower）"""
        # 获取原始层的设备
        device = original_layer.weight.device
        
        class QuantizedConv2d(nn.Module):
            def __init__(self, original_conv, target_bits, num_groups, device):
                super().__init__()
                self.conv = original_conv  # 保留原始conv用于结构
                
                # 创建量化的权重
                weight = original_conv.weight
                self.out_channels = weight.shape[0]
                self.in_channels = weight.shape[1]
                self.kernel_size = weight.shape[2:]
                
                # 将卷积权重展平为2D进行量化
                weight_2d = weight.view(self.out_channels, -1)
                
                self.quantized_weight = QuantizedLinear(
                    in_features=weight_2d.shape[1],
                    out_features=weight_2d.shape[0],
                    bias=original_conv.bias is not None,
                    num_bits=int(target_bits),
                    per_channel=True,
                    symmetric=True,
                    num_groups=min(num_groups, self.out_channels),
                    use_adaptive=False  # Conv2d使用统一量化
                )
                
                # 将量化层移到正确的设备
                self.quantized_weight = self.quantized_weight.to(device)
                
                self.quantized_weight.weight_fp.data = weight_2d
                if original_conv.bias is not None:
                    self.quantized_weight.bias.data = original_conv.bias.data.clone()
                
                self.quantized_weight.quantize_weight()
                
                # 保存其他参数
                self.stride = original_conv.stride
                self.padding = original_conv.padding
                self.dilation = original_conv.dilation
                self.groups = original_conv.groups
            
            def forward(self, x):
                # 获取量化的权重并重塑为卷积核形状
                weight_2d = self.quantized_weight.weight_quantized
                weight_4d = weight_2d.view(self.out_channels, self.in_channels, *self.kernel_size)
                
                return F.conv2d(
                    x, weight_4d, 
                    self.quantized_weight.bias if hasattr(self.quantized_weight, 'bias') else None,
                    self.stride, self.padding, self.dilation, self.groups
                )
        
        return QuantizedConv2d(original_layer, self.target_bits_weight, self.num_groups, device)
    
    def _create_ib_weight_quantizer_for_vision(self):
        """为Vision Tower创建基于IB的权重量化器"""
        vision_importance = HybridImportanceEstimation(
            feature_dim=self.vision_dim,
            num_groups=self.num_groups
        )
        
        vision_bit_allocator = BitAllocationNetwork(
            feature_dim=self.vision_dim,
            num_groups=self.num_groups,
            target_bits=self.target_bits_weight,
            bit_levels=[2, 4, 8]
        )
        
        self.weight_quantizers['vision_ib'] = {
            'importance': vision_importance,
            'bit_allocator': vision_bit_allocator
        }
    
    def _create_ib_weight_quantizer_for_projector(self):
        """为Projector创建基于IB的权重量化器"""
        proj_importance = HybridImportanceEstimation(
            feature_dim=self.projection_output_dim,
            num_groups=self.num_groups
        )
        
        proj_bit_allocator = BitAllocationNetwork(
            feature_dim=self.projection_output_dim,
            num_groups=self.num_groups,
            target_bits=self.target_bits_weight,
            bit_levels=[2, 4, 8]
        )
        
        self.weight_quantizers['projector_ib'] = {
            'importance': proj_importance,
            'bit_allocator': proj_bit_allocator
        }
    
    def _get_parent_and_attr(self, root_module, module_name):
        """获取模块的父模块和属性名"""
        if '.' not in module_name:
            return '', module_name
        
        parts = module_name.split('.')
        parent_name = '.'.join(parts[:-1])
        attr_name = parts[-1]
        return parent_name, attr_name
    
    def _get_module_by_name(self, root_module, module_name):
        """根据名称获取模块"""
        if not module_name:
            return root_module
        
        parts = module_name.split('.')
        module = root_module
        for part in parts:
            module = getattr(module, part)
        return module
    
    def _init_alignment_modules(self):
        """初始化对齐损失相关模块（Stage 2）"""
        print("Initializing alignment modules for Stage 2...")
        
        self.alignment_temperature = nn.Parameter(torch.tensor(0.07))
        
        self.visual_proj_head = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.ReLU(),
            nn.Linear(self.llm_dim, self.llm_dim // 2),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 2, 256)
        )
        
        self.text_proj_head = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.ReLU(),
            nn.Linear(self.llm_dim, self.llm_dim // 2),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 2, 256)
        )
    
    def _setup_quantization_hooks(self):
        """设置量化hooks"""
        self._original_methods = {}
        
        if hasattr(self.base_model, 'encode_images'):
            self._original_methods['encode_images'] = self.base_model.encode_images
            self.base_model.encode_images = self._quantized_encode_images
            print("✓ Hooked encode_images")
    
    def _quantized_encode_images(self, images):
        """量化版本的encode_images"""
        # 获取vision tower
        vision_tower = self.base_model.get_vision_tower()
        
        # 通过vision tower
        if vision_tower:
            if hasattr(self, '_original_methods') and 'encode_images' in self._original_methods:
                with torch.no_grad():
                    image_features = self._original_methods['encode_images'](images)
            else:
                image_features = vision_tower(images)
        else:
            return self._original_methods['encode_images'](images)
        
        # 处理vision输出
        if hasattr(image_features, 'last_hidden_state'):
            features = image_features.last_hidden_state
        elif isinstance(image_features, tuple):
            features = image_features[0]
        else:
            features = image_features
        
        # Vision特征量化
        if self.training and 'vision' in self.activation_quantizers:
            features = self._apply_activation_quantization(features, 'vision')
        
        # 通过投影器
        model = self.base_model.get_model()
        if hasattr(model, 'mm_projector'):
            # IB处理
            if self.use_ib and 'projector_ib' in self.activation_quantizers:
                ib_result = self.activation_quantizers['projector_ib'](features)
                features = ib_result['z']
                self._last_ib_result = ib_result
            
            # 通过投影器
            features = model.mm_projector(features)
            
            # 投影后量化
            if self.training and 'projector' in self.activation_quantizers:
                features = self._apply_activation_quantization(features, 'projector')
            
            # Stage 2: 保存用于对齐
            if self.training and self.stage == 2:
                if not hasattr(self, '_intermediate_features'):
                    self._intermediate_features = {}
                self._intermediate_features['visual_features'] = features.detach()
        
        return features
    
    def _apply_activation_quantization(self, features, module_name):
        """应用激活值量化"""
        if not self.training or module_name not in self.activation_quantizers:
            return features
        
        modules = self.activation_quantizers[module_name]
        
        # 处理维度
        original_shape = features.shape
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        # 1. 计算重要性
        importance_scores, importance_aux = modules['importance'](features)
        
        # 2. 分配比特
        allocation_result = modules['bit_allocator'](importance_scores)
        
        # 3. 执行量化
        features_quantized = modules['quantizer'](
            features,
            bit_assignment=allocation_result['bit_assignment'],
            group_indices=allocation_result['group_indices']
        )
        
        # 恢复形状
        if len(original_shape) == 2:
            features_quantized = features_quantized.squeeze(1)
        
        # 保存辅助信息
        if not hasattr(self, '_quantization_aux'):
            self._quantization_aux = {}
        self._quantization_aux[module_name] = {
            'importance_aux': importance_aux,
            'allocation_result': allocation_result
        }
        
        return features_quantized
    
    def _setup_lora_adapters(self):
        """设置LoRA适配器（Stage 2）"""
        print("Setting up LoRA adapters for Stage 2...")
        
        model = self.base_model.get_model()
        lora_params = 0
        
        if hasattr(model, 'layers'):
            num_layers = len(model.layers)
            # 选择要添加LoRA的层
            lora_layers = list(range(0, num_layers, max(1, num_layers // 8)))
            
            for i in lora_layers:
                layer = model.layers[i]
                
                # 注意力层LoRA
                if hasattr(layer, 'self_attn'):
                    for proj_name in ['q_proj', 'v_proj']:
                        if hasattr(layer.self_attn, proj_name):
                            original = getattr(layer.self_attn, proj_name)
                            lora_layer = self._add_lora_to_layer(original)
                            setattr(layer.self_attn, proj_name, lora_layer)
                            lora_params += 2 * self.lora_rank * original.in_features
                
                print(f"✓ Added LoRA to layer {i}")
        
        if lora_params > 0:
            print(f"Total LoRA parameters: {lora_params/1e6:.2f}M")
    
    def _add_lora_to_layer(self, layer):
        """为单个层添加LoRA"""
        if not isinstance(layer, nn.Linear):
            return layer
        
        class LoRALinear(nn.Module):
            def __init__(self, base_layer, rank=16, alpha=32):
                super().__init__()
                self.base_layer = base_layer
                self.rank = rank
                self.alpha = alpha
                self.scaling = alpha / rank
                
                # 冻结原始权重
                for param in base_layer.parameters():
                    param.requires_grad = False
                
                # LoRA参数
                self.lora_A = nn.Parameter(torch.randn(rank, base_layer.in_features) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
            
            def forward(self, x):
                base_out = self.base_layer(x)
                lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
                return base_out + lora_out
        
        return LoRALinear(layer, self.lora_rank)
    
    def _apply_memory_optimizations(self):
        """应用内存优化"""
        # 梯度检查点
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _print_quantization_summary(self):
        """打印量化统计摘要"""
        print("\n" + "="*80)
        print("Quantization Summary")
        print("="*80)
        
        # 激活量化统计
        print(f"Activation Quantization:")
        print(f"  Target bits: {self.target_bits_act}")
        print(f"  Number of groups: {self.num_groups}")
        print(f"  Using IB framework: {self.use_ib}")
        
        # 权重量化统计
        if self.quantize_weights:
            print(f"\nWeight Quantization:")
            print(f"  Target bits: {self.target_bits_weight}")
            print(f"  Mode: {self.weight_quant_mode}")
            print(f"  Quantized layers: {len(self.quantized_layers)}")
            
            # 按类型统计
            layer_types = {}
            for name in self.quantized_layers.keys():
                if 'vision' in name:
                    layer_types['vision'] = layer_types.get('vision', 0) + 1
                elif 'projector' in name:
                    layer_types['projector'] = layer_types.get('projector', 0) + 1
                elif 'llm' in name:
                    layer_types['llm'] = layer_types.get('llm', 0) + 1
                elif 'lm_head' in name:
                    layer_types['lm_head'] = layer_types.get('lm_head', 0) + 1
            
            for layer_type, count in layer_types.items():
                print(f"    {layer_type}: {count} layers")
            
            # 计算压缩率
            self._calculate_compression_ratio()
        
        # 总参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params/1e9:.3f}B")
        print(f"Trainable parameters: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
        print("="*80)

    def _create_ib_weight_quantizer_for_vision(self):
        """为Vision Tower创建基于IB的权重量化器"""
        vision_ib_modules = nn.ModuleDict({
            'importance': HybridImportanceEstimation(
                feature_dim=self.vision_dim,
                num_groups=self.num_groups
            ),
            'bit_allocator': BitAllocationNetwork(
                feature_dim=self.vision_dim,
                num_groups=self.num_groups,
                target_bits=self.target_bits_weight,
                bit_levels=[2, 4, 8]
            )
        })
        
        self.weight_quantizers['vision_ib'] = vision_ib_modules

    def _create_ib_weight_quantizer_for_projector(self):
        """为Projector创建基于IB的权重量化器"""
        projector_ib_modules = nn.ModuleDict({
            'importance': HybridImportanceEstimation(
                feature_dim=self.projection_output_dim,
                num_groups=self.num_groups
            ),
            'bit_allocator': BitAllocationNetwork(
                feature_dim=self.projection_output_dim,
                num_groups=self.num_groups,
                target_bits=self.target_bits_weight,
                bit_levels=[2, 4, 8]
            )
        })
        
        self.weight_quantizers['projector_ib'] = projector_ib_modules
    def _calculate_compression_ratio(self):
        """计算模型压缩率"""
        total_original_size = 0
        total_quantized_size = 0
        
        for name, module in self.quantized_layers.items():
            if isinstance(module, QuantizedLinear):
                param_count = module.in_features * module.out_features
                original_size = param_count * 32 / 8  # FP32 in bytes
                
                if module.use_adaptive and module.bit_assignment is not None:
                    avg_bits = module.bit_assignment.mean().item()
                else:
                    avg_bits = module.num_bits
                
                quantized_size = param_count * avg_bits / 8
                
                total_original_size += original_size
                total_quantized_size += quantized_size
        
        if total_original_size > 0:
            compression_ratio = total_original_size / total_quantized_size
            print(f"\n  Compression Statistics:")
            print(f"    Original size: {total_original_size/1e9:.2f} GB")
            print(f"    Quantized size: {total_quantized_size/1e9:.2f} GB")
            print(f"    Compression ratio: {compression_ratio:.2f}x")
    
    def update_weight_bit_allocation(self, sample_data=None):
        """动态更新权重比特分配（Stage 1训练）"""
        if not self.quantize_weights or self.weight_quant_mode != 'mixed':
            return
        
        print("Updating weight bit allocation...")
        
        # 更新各模块的比特分配
        for module_name in ['vision', 'projector']:
            if f'{module_name}_ib' in self.weight_quantizers:
                ib_module = self.weight_quantizers[f'{module_name}_ib']
                
                # 收集该模块的所有量化层
                module_layers = {
                    name: layer for name, layer in self.quantized_layers.items()
                    if module_name in name
                }
                
                # 为每个层计算重要性并分配比特
                for layer_name, layer in module_layers.items():
                    if layer.use_adaptive:
                        # 基于权重统计计算重要性
                        weight_magnitude = layer.weight_fp.abs().mean(dim=1)
                        weight_variance = layer.weight_fp.var(dim=1)
                        importance = weight_magnitude * torch.sqrt(weight_variance + 1e-8)
                        
                        # 归一化
                        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
                        
                        # 分配比特 - 注意这里的访问方式变了
                        if isinstance(ib_module, nn.ModuleDict):
                            allocation_result = ib_module['bit_allocator'](importance.unsqueeze(0))
                        else:  # 如果使用方法1的包装类
                            allocation_result = ib_module.bit_allocator(importance.unsqueeze(0))
                        
                        bit_assignment = allocation_result['bit_assignment'].squeeze(0)
                        
                        # 更新量化
                        layer.quantize_weight(bit_assignment)
                        
                        avg_bits = bit_assignment.mean().item()
                        print(f"  {layer_name}: {avg_bits:.2f} bits")
    

    
    def get_quantization_stats(self):
        """获取完整的量化统计"""
        stats = {}
        
        # 激活量化统计
        for module_name, aux_info in getattr(self, '_quantization_aux', {}).items():
            if 'allocation_result' in aux_info:
                bit_assignment = aux_info['allocation_result']['bit_assignment']
                stats[f'{module_name}_act_avg_bits'] = bit_assignment.mean().item()
                stats[f'{module_name}_act_bit_std'] = bit_assignment.std().item()
        
        # 权重量化统计
        if self.quantize_weights:
            weight_bits = []
            for name, module in self.quantized_layers.items():
                if isinstance(module, QuantizedLinear):
                    if module.use_adaptive and module.bit_assignment is not None:
                        avg_bits = module.bit_assignment.mean().item()
                    else:
                        avg_bits = module.num_bits
                    weight_bits.append(avg_bits)
            
            if weight_bits:
                stats['weight_avg_bits'] = sum(weight_bits) / len(weight_bits)
                stats['weight_min_bits'] = min(weight_bits)
                stats['weight_max_bits'] = max(weight_bits)
                stats['weight_quantized_layers'] = len(weight_bits)
        
        # 对齐统计（Stage 2）
        if hasattr(self, '_alignment_stats'):
            stats.update(self._alignment_stats)
        
        return stats
    
    def forward(self, **kwargs):
        """前向传播"""
        # 清空辅助信息
        self._quantization_aux = {}
        self._intermediate_features = {}
        self._alignment_stats = {}
        
        # Stage 1且使用混合精度权重量化：定期更新比特分配
        if self.training and self.stage == 1 and self.weight_quant_mode == 'mixed':
            if hasattr(self, 'update_counter'):
                self.update_counter += 1
            else:
                self.update_counter = 1
            
            # 每100步更新一次权重比特分配
            if self.update_counter % 100 == 0:
                self.update_weight_bit_allocation()
        
        # 调用base model
        outputs = self.base_model(**kwargs)
        
        # 添加各种损失
        if self.training and hasattr(outputs, 'loss'):
            total_loss = outputs.loss
            
            # Stage 1: IB损失
            if self.stage == 1:
                if hasattr(self, '_last_ib_result'):
                    total_loss = total_loss + 0.5 * self._last_ib_result['total_loss']
                
                # 比特率损失
                for module_name, aux_info in self._quantization_aux.items():
                    if 'allocation_result' in aux_info:
                        total_loss = total_loss + 0.1 * aux_info['allocation_result']['bitrate_loss']
            
            # Stage 2: 对齐损失
            elif self.stage == 2:
                if 'visual_features' in self._intermediate_features and 'text_features' in self._intermediate_features:
                    alignment_loss = self.compute_alignment_loss(
                        self._intermediate_features['visual_features'],
                        self._intermediate_features['text_features']
                    )
                    total_loss = total_loss + 0.3 * alignment_loss
                    
                    if not hasattr(outputs, 'aux_losses'):
                        outputs.aux_losses = {}
                    outputs.aux_losses['alignment_loss'] = alignment_loss
            
            outputs.loss = total_loss
        
        return outputs
    
    def compute_alignment_loss(self, visual_features, text_features):
        """计算视觉-文本对齐损失"""
        batch_size = visual_features.size(0)
        
        # 池化
        visual_pooled = visual_features.mean(dim=1)
        if text_features.dim() == 3:
            text_pooled = text_features.mean(dim=1)
        else:
            text_pooled = text_features
        
        # 投影头
        visual_proj = self.visual_proj_head(visual_pooled)
        text_proj = self.text_proj_head(text_pooled)
        
        # L2归一化
        visual_proj = F.normalize(visual_proj, p=2, dim=-1)
        text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        # 计算相似度
        temperature = torch.exp(self.alignment_temperature).clamp(min=0.01, max=0.5)
        sim_matrix = torch.matmul(visual_proj, text_proj.T) / temperature
        
        # InfoNCE损失
        labels = torch.arange(batch_size, device=sim_matrix.device)
        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.T, labels)
        
        alignment_loss = (loss_v2t + loss_t2v) / 2
        
        # 计算准确率
        with torch.no_grad():
            pred_v2t = sim_matrix.argmax(dim=1)
            pred_t2v = sim_matrix.T.argmax(dim=1)
            acc_v2t = (pred_v2t == labels).float().mean()
            acc_t2v = (pred_t2v == labels).float().mean()
            
            if not hasattr(self, '_alignment_stats'):
                self._alignment_stats = {}
            self._alignment_stats['acc_v2t'] = acc_v2t.item()
            self._alignment_stats['acc_t2v'] = acc_t2v.item()
            self._alignment_stats['temperature'] = temperature.item()
        
        return alignment_loss