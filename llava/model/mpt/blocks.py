"""GPT Blocks used for the GPT Model."""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from .attention import ATTN_CLASS_REGISTRY
from .norm import NORM_CLASS_REGISTRY

class MPTMLP(nn.Module):

    def __init__(self, d_model: int, expansion_ratio: int, device: Optional[str]=None):
        super().__init__()
        self.up_proj = nn.Linear(d_model, expansion_ratio * d_model, device=device)
        self.act = nn.GELU(approximate='none')
        self.down_proj = nn.Linear(expansion_ratio * d_model, d_model, device=device)
        self.down_proj._is_residual = True

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))

class MPTBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, expansion_ratio: int, attn_config: Dict={'attn_type': 'multihead_attention', 'attn_pdrop': 0.0, 'attn_impl': 'triton', 'qk_ln': False, 'clip_qkv': None, 'softmax_scale': None, 'prefix_lm': False, 'attn_uses_sequence_id': False, 'alibi': False, 'alibi_bias_max': 8}, resid_pdrop: float=0.0, norm_type: str='low_precision_layernorm', device: Optional[str]=None, **kwargs):
        del kwargs
        super().__init__()
        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]
        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(attn_impl=attn_config['attn_impl'], clip_qkv=attn_config['clip_qkv'], qk_ln=attn_config['qk_ln'], softmax_scale=attn_config['softmax_scale'], attn_pdrop=attn_config['attn_pdrop'], d_model=d_model, n_heads=n_heads, device=device)
        self.norm_2 = norm_class(d_model, device=device)
        self.ffn = MPTMLP(d_model=d_model, expansion_ratio=expansion_ratio, device=device)
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor]]=None, attn_bias: Optional[torch.Tensor]=None, attention_mask: Optional[torch.ByteTensor]=None, is_causal: bool=True) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        a = self.norm_1(x)
        (b, _, past_key_value) = self.attn(a, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=is_causal)
        x = x + self.resid_attn_dropout(b)
        m = self.norm_2(x)
        n = self.ffn(m)
        x = x + self.resid_ffn_dropout(n)
        return (x, past_key_value)