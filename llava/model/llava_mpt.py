#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import math

from transformers import AutoConfig, AutoModelForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .mpt.modeling_mpt import MPTConfig, MPTForCausalLM, MPTModel


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class LlavaMPTConfig(MPTConfig):
    model_type = "llava_mpt"


class LlavaMPTModel(MPTModel):
    config_class = LlavaMPTConfig

    def __init__(self, config: MPTConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(LlavaMPTModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]
            # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.d_model)

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.d_model)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items() if 'mm_projector' in k})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, images=None):

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        inputs_embeds = self.wte(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(images, output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                    select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                    image_features = select_hidden_state[:, 1:]
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlavaMPTModel, self).forward(input_ids=None, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, tok_emb=inputs_embeds)


class LlavaMPTForCausalLM(MPTForCausalLM):
    config_class = LlavaMPTConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MPTForCausalLM, self).__init__(config)

        if not config.tie_word_embeddings:
            raise ValueError('MPTForCausalLM only supports tied word embeddings')
        self.transformer = LlavaMPTModel(config)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaMPTModel):
            module.gradient_checkpointing = value

    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, images=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, images=images)
        logits = F.linear(outputs.last_hidden_state, self.transformer.wte.weight)
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale
        loss = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            raise NotImplementedError('inputs_embeds is not implemented for MPT yet')
        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError('MPT does not support generation with right padding.')
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get('use_cache') == False:
                raise NotImplementedError('MPT with prefix_lm=True does not support use_cache=False.')
        else:
            prefix_mask = None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'prefix_mask': prefix_mask, 'sequence_id': sequence_id, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache', True), "images": kwargs.get("images", None)}

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_tower[0].config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['transformer.wte.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

AutoConfig.register("llava_mpt", LlavaMPTConfig)
AutoModelForCausalLM.register(LlavaMPTConfig, LlavaMPTForCausalLM)
