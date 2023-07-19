"""Converts Huggingface Causal LM to Prefix LM.

Conversion does lightweight surgery on a HuggingFace
Causal LM to convert it to a Prefix LM.

Prefix LMs accepts a `bidirectional_mask` input in `forward`
and treat the input prompt as the prefix in `generate`.
"""
import math
import warnings
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers.models.bloom.modeling_bloom import BaseModelOutputWithPastAndCrossAttentions, BloomForCausalLM, BloomModel, CausalLMOutputWithCrossAttentions, CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import _expand_mask as _expand_mask_bloom
from transformers.models.bloom.modeling_bloom import _make_causal_mask as _make_causal_mask_bloom
from transformers.models.bloom.modeling_bloom import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.opt.modeling_opt import _expand_mask as _expand_mask_opt
from transformers.models.opt.modeling_opt import _make_causal_mask as _make_causal_mask_opt
logger = logging.get_logger(__name__)
_SUPPORTED_GPT_MODELS = (GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM)
CAUSAL_GPT_TYPES = Union[GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM]

def _convert_gpt_causal_lm_to_prefix_lm(model: CAUSAL_GPT_TYPES) -> CAUSAL_GPT_TYPES:
    """Converts a GPT-style Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`

    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, '_prefix_lm_converted'):
        return model
    assert isinstance(model, _SUPPORTED_GPT_MODELS)
    assert model.config.add_cross_attention == False, 'Only supports GPT-style decoder-only models'

    def _get_attn_modules(model: CAUSAL_GPT_TYPES) -> List[torch.nn.Module]:
        """Helper that gets a list of the model's attention modules.

        Each module has a `bias` buffer used for causal masking. The Prefix LM
        conversion adds logic to dynamically manipulate these biases to support
        Prefix LM attention masking.
        """
        attn_modules = []
        if isinstance(model, GPTNeoXForCausalLM):
            blocks = model.gpt_neox.layers
        else:
            blocks = model.transformer.h
        for block in blocks:
            if isinstance(model, GPTNeoForCausalLM):
                if block.attn.attention_type != 'global':
                    continue
                attn_module = block.attn.attention
            elif isinstance(model, GPTNeoXForCausalLM):
                attn_module = block.attention
            else:
                attn_module = block.attn
            attn_modules.append(attn_module)
        return attn_modules
    setattr(model, '_original_forward', getattr(model, 'forward'))
    setattr(model, '_original_generate', getattr(model, 'generate'))

    def forward(self: CAUSAL_GPT_TYPES, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, bidirectional_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        """Wraps original forward to enable PrefixLM attention."""

        def call_og_forward():
            if isinstance(self, GPTNeoXForCausalLM):
                return self._original_forward(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            else:
                return self._original_forward(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if bidirectional_mask is None:
            return call_og_forward()
        assert isinstance(bidirectional_mask, torch.Tensor)
        attn_modules = _get_attn_modules(model)
        (b, s) = bidirectional_mask.shape
        max_length = attn_modules[0].bias.shape[-1]
        if s > max_length:
            raise ValueError(f'bidirectional_mask sequence length (={s}) exceeds the ' + f'max length allowed by the model ({max_length}).')
        assert s <= max_length
        if s < max_length:
            pad = torch.zeros((int(b), int(max_length - s)), dtype=bidirectional_mask.dtype, device=bidirectional_mask.device)
            bidirectional_mask = torch.cat([bidirectional_mask, pad], dim=1)
        bidirectional = bidirectional_mask.unsqueeze(1).unsqueeze(1)
        for attn_module in attn_modules:
            attn_module.bias.data = torch.logical_or(attn_module.bias.data, bidirectional)
        output = call_og_forward()
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(attn_module.bias.data[0, 0])[None, None]
        return output

    def generate(self: CAUSAL_GPT_TYPES, *args: tuple, **kwargs: Dict[str, Any]):
        """Wraps original generate to enable PrefixLM attention."""
        attn_modules = _get_attn_modules(model)
        for attn_module in attn_modules:
            attn_module.bias.data[:] = 1
        output = self._original_generate(*args, **kwargs)
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(attn_module.bias.data[0, 0])[None, None]
        return output
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'generate', MethodType(generate, model))
    setattr(model, '_prefix_lm_converted', True)
    return model

def _convert_bloom_causal_lm_to_prefix_lm(model: BloomForCausalLM) -> BloomForCausalLM:
    """Converts a BLOOM Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `BloomForCausalLM`

    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, '_prefix_lm_converted'):
        return model
    assert isinstance(model, BloomForCausalLM)
    assert model.config.add_cross_attention == False, 'Only supports BLOOM decoder-only models'

    def _prepare_attn_mask(self: BloomModel, attention_mask: torch.Tensor, bidirectional_mask: Optional[torch.Tensor], input_shape: Tuple[int, int], past_key_values_length: int) -> torch.BoolTensor:
        combined_attention_mask = None
        device = attention_mask.device
        (_, src_length) = input_shape
        if src_length > 1:
            combined_attention_mask = _make_causal_mask_bloom(input_shape, device=device, past_key_values_length=past_key_values_length)
            if bidirectional_mask is not None:
                assert attention_mask.shape == bidirectional_mask.shape
                expanded_bidirectional_mask = _expand_mask_bloom(bidirectional_mask, tgt_length=src_length)
                combined_attention_mask = torch.logical_and(combined_attention_mask, expanded_bidirectional_mask)
        expanded_attn_mask = _expand_mask_bloom(attention_mask, tgt_length=src_length)
        combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        return combined_attention_mask

    def _build_alibi_tensor(self: BloomModel, batch_size: int, query_length: int, key_length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        num_heads = self.config.n_head
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = torch.tensor(2 ** (-2 ** (-(math.log2(closest_power_of_2) - 3))), device=device, dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
        slopes = torch.pow(base, powers)
        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(2 ** (-2 ** (-(math.log2(2 * closest_power_of_2) - 3))), device=device, dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        qa = torch.arange(query_length, device=device, dtype=torch.int32).view(-1, 1)
        ka = torch.arange(key_length, device=device, dtype=torch.int32).view(1, -1)
        diffs = qa - ka + key_length - query_length
        diffs = -diffs.abs()
        alibi = slopes.view(1, num_heads, 1, 1) * diffs.view(1, 1, query_length, key_length)
        alibi = alibi.expand(batch_size, -1, -1, -1).reshape(-1, query_length, key_length)
        return alibi.to(dtype)
    KeyValueT = Tuple[torch.Tensor, torch.Tensor]

    def forward(self: BloomModel, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[KeyValueT, ...]]=None, attention_mask: Optional[torch.Tensor]=None, bidirectional_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn('`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. ' + 'You can safely ignore passing `position_ids`.', FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(f'Got unexpected arguments: {deprecated_arguments}')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            (batch_size, seq_length) = input_ids.shape
        elif inputs_embeds is not None:
            (batch_size, seq_length, _) = inputs_embeds.shape
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            tmp = past_key_values[0][0]
            past_key_values_length = tmp.shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)
        alibi = self._build_alibi_tensor(batch_size=batch_size, query_length=seq_length, key_length=seq_length_with_past, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask = self._prepare_attn_mask(attention_mask, bidirectional_mask, input_shape=(batch_size, seq_length), past_key_values_length=past_key_values_length)
        for (i, (block, layer_past)) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                hst = (hidden_states,)
                all_hidden_states = all_hidden_states + hst
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)
                    return custom_forward
                outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(block), hidden_states, alibi, causal_mask, head_mask[i])
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=causal_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions, alibi=alibi)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                oa = (outputs[2 if use_cache else 1],)
                all_self_attentions = all_self_attentions + oa
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            hst = (hidden_states,)
            all_hidden_states = all_hidden_states + hst
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions)
    setattr(model.transformer, '_prepare_attn_mask', MethodType(_prepare_attn_mask, model.transformer))
    setattr(model.transformer, '_build_alibi_tensor', MethodType(_build_alibi_tensor, model.transformer))
    setattr(model.transformer, 'forward', MethodType(forward, model.transformer))
    KeyValueT = Tuple[torch.Tensor, torch.Tensor]

    def forward(self: BloomForCausalLM, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[KeyValueT, ...]]=None, attention_mask: Optional[torch.Tensor]=None, bidirectional_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        """Replacement forward method for BloomCausalLM."""
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn('`position_ids` have no functionality in BLOOM and will be removed ' + 'in v5.0.0. You can safely ignore passing `position_ids`.', FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(f'Got unexpected arguments: {deprecated_arguments}')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, bidirectional_mask=bidirectional_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            (batch_size, seq_length, vocab_size) = shift_logits.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def prepare_inputs_for_generation(self: BloomForCausalLM, input_ids: torch.LongTensor, past: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> dict:
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            bidirectional_mask = None
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_bloom_cache(past)
        else:
            bidirectional_mask = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'past_key_values': past, 'use_cache': True, 'attention_mask': attention_mask, 'bidirectional_mask': bidirectional_mask}
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'prepare_inputs_for_generation', MethodType(prepare_inputs_for_generation, model))
    setattr(model, '_prefix_lm_converted', True)
    return model

def _convert_opt_causal_lm_to_prefix_lm(model: OPTForCausalLM) -> OPTForCausalLM:
    """Converts an OPT Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `OPTForCausalLM`

    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, '_prefix_lm_converted'):
        return model
    assert isinstance(model, OPTForCausalLM)
    assert model.config.add_cross_attention == False, 'Only supports OPT decoder-only models'
    setattr(model, '_original_forward', getattr(model, 'forward'))
    setattr(model, '_original_generate', getattr(model, 'generate'))
    model.model.decoder.bidirectional_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            if self.bidirectional_mask == 'g':
                (bsz, src_length) = input_shape
                combined_attention_mask = torch.zeros((bsz, 1, src_length, src_length + past_key_values_length), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            else:
                combined_attention_mask = _make_causal_mask_opt(input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length).to(inputs_embeds.device)
                if self.bidirectional_mask is not None:
                    assert attention_mask.shape == self.bidirectional_mask.shape
                    expanded_bidirectional_mask = _expand_mask_opt(self.bidirectional_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
                    combined_attention_mask = torch.maximum(expanded_bidirectional_mask, combined_attention_mask)
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask_opt(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        return combined_attention_mask
    setattr(model.model.decoder, '_prepare_decoder_attention_mask', MethodType(_prepare_decoder_attention_mask, model.model.decoder))

    def forward(self: OPTForCausalLM, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, bidirectional_mask: Optional[torch.ByteTensor]=None, head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):

        def call_og_forward():
            return self._original_forward(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if bidirectional_mask is None:
            return call_og_forward()
        self.model.decoder.bidirectional_mask = bidirectional_mask
        try:
            outputs = call_og_forward()
        except:
            self.model.decoder.bidirectional_mask = None
            raise
        self.model.decoder.bidirectional_mask = None
        return outputs

    def generate(self: OPTForCausalLM, *args: tuple, **kwargs: Dict[str, Any]):
        """Wraps original generate to enable PrefixLM-style attention."""
        self.model.decoder.bidirectional_mask = 'g'
        try:
            output = self._original_generate(*args, **kwargs)
        except:
            self.model.decoder.bidirectional_mask = None
            raise
        self.model.decoder.bidirectional_mask = None
        return output
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'generate', MethodType(generate, model))
    setattr(model, '_prefix_lm_converted', True)
    return model
_SUPPORTED_HF_MODELS = _SUPPORTED_GPT_MODELS + (BloomForCausalLM, OPTForCausalLM)
CAUSAL_LM_TYPES = Union[GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM, BloomForCausalLM, OPTForCausalLM]

def convert_hf_causal_lm_to_prefix_lm(model: CAUSAL_LM_TYPES) -> CAUSAL_LM_TYPES:
    """Converts a HuggingFace Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`
        - `BloomForCausalLM`
        - `OPTForCausalLM`

    Conversion to a Prefix LM is done by modifying the `forward` method, and possibly also the
    `generate` method and/or select underlying methods depending on the model class.

    These changes preserve the model API, but add a new input to `forward`: "bidirectional_mask".

    Notes on training:
        To actually train the converted model as a Prefix LM, training batches will need to indicate
        the prefix/target structure by including `bidirectional_mask` as part of the batch inputs.

        **This is not a standard input and requires custom layers either within or after your dataloader.**

        In addition to adding `bidirectional_mask` to the batch, this custom code should modify `labels`
        such that `batch['labels'][batch['bidirectional_mask'] == 1] == -100`.
        That is, the prefix portion of the sequence should not generate any loss. Loss should only be
        generated by the target portion of the sequence.

    Notes on `GPTNeoForCausalLM`:
        To simplify the implementation, "global" and "local" attention layers are handled differently.
        For "global" layers, we handle conversion as described above. For "local" layers, which use a
        causal attention mask within a restricted local window, we do not alter the masking.

    Notes on `forward` method conversion:
        After conversion, the `forward` method will handle a new input, `bidirectional_mask`,
        which should be a [batch_size, seq_length] byte tensor, where 1 indicates token positions
        belonging to the prefix (prefix tokens can attend to one another bidirectionally), and
        0 indicates token positions belonging to the target.

        The new `forward` method will incorporate `bidirectional_mask` (if supplied) into the existing
        causal mask, call the original `forward` method, and (if the causal mask is a buffer) reset
        the causal masks before returning the result.

    Notes on `generate` method conversion:
        After conversion, the `generate` method will have the same signature but will internally
        convert all causal masks to be purely bidirectional, call the original `generate` method, and
        (where appropriate) reset the causal masks before returning the result.

        This works thanks to the logic of the HuggingFace `generate` API, which first encodes the token
        "prompt" passed to `generate` (which is treated as the prefix) and then sequentially generates
        each new token. Encodings are cached as generation happens, so all prefix tokens can attend to one
        another (as expected in a Prefix LM) and generated tokens can only attend to prefix tokens and
        previously-generated tokens (also as expected in a Prefix LM).

    To preserve the API, the original methods are renamed to `_original_forward` and
    `_original_generate`, and replaced with new `forward` and `generate` methods that wrap
    them, respectively. Although implementation details vary by model class.
    """
    if isinstance(model, _SUPPORTED_GPT_MODELS):
        return _convert_gpt_causal_lm_to_prefix_lm(model)
    elif isinstance(model, BloomForCausalLM):
        return _convert_bloom_causal_lm_to_prefix_lm(model)
    elif isinstance(model, OPTForCausalLM):
        return _convert_opt_causal_lm_to_prefix_lm(model)
    else:
        raise TypeError(f'Cannot convert model to Prefix LM. ' + f'Model does not belong to set of supported HF models:' + f'\n{_SUPPORTED_HF_MODELS}')

def add_bidirectional_mask_if_missing(batch: Dict[str, Any]):
    """Attempts to add bidirectional_mask to batch if missing.

    Raises:
        KeyError if bidirectional_mask is missing and can't be inferred
    """
    if 'bidirectional_mask' not in batch:
        if batch.get('mode', None) == 'icl_task':
            batch['bidirectional_mask'] = batch['attention_mask'].clone()
            for (i, continuation_indices) in enumerate(batch['continuation_indices']):
                batch['bidirectional_mask'][i, continuation_indices] = 0
        elif 'labels' in batch and 'attention_mask' in batch:
            batch['bidirectional_mask'] = torch.logical_and(torch.eq(batch['attention_mask'], 1), torch.eq(batch['labels'], -100)).type_as(batch['attention_mask'])
        else:
            raise KeyError('No bidirectional_mask in batch and not sure how to construct one.')