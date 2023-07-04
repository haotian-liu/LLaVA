from typing import Optional, Tuple

import torch
import torch.backends.cuda
import transformers
from einops import rearrange
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

torch.backends.cuda.enable_flash_sdp(True)


def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, nh, t, hd]

    kv_seq_len = key_states.shape[-2]
    offset = 0

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"
    assert past_key_value is None, "past_key_value is not supported"

    output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
    )  # [bsz, nh, t, hd]
    return self.o_proj(rearrange(output, 'b h t d -> b t (h d)')), None, None


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward

