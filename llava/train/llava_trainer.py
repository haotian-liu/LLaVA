import os
import torch
import torch.nn as nn

from transformers import Trainer
from typing import Dict, Optional, Sequence
from transformers.trainer import logger



def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class LLaVATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get('args')
        model = kwargs.get('model')

        self.no_grad_params = []
        if not training_args.force_fsdp and \
            (training_args.fsdp is not None and len(training_args.fsdp) > 0) and \
                any(not p.requires_grad for p in model.parameters()):
            logger.info(f'[Experimental] Using FSDP while some parameters do not require grad.  Setting their LR to zero.')
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    self.no_grad_params.append(n)
                    p.requires_grad = True
            logger.info('Params that do not require grad: {}'.format(self.no_grad_params))
        
        self.no_grad_params = set(self.no_grad_params)

        super(LLaVATrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        super(LLaVATrainer, self)._save(output_dir, state_dict)

    def create_optimizer(self):
        # patch for FSDP

        if self.args.force_fsdp:
            # When forced, use the original implementation
            assert len(self.no_grad_params) == 0
            return super(LLaVATrainer, self).create_optimizer()

        if self.args.fsdp is None or len(self.args.fsdp) == 0:
            # When not using FSDP, use the original implementation
            assert len(self.no_grad_params) == 0
            return super(LLaVATrainer, self).create_optimizer()

        if len(self.no_grad_params) == 0:
            # When using full-model finetuning, use the original implementation
            return super(LLaVATrainer, self).create_optimizer()

        from transformers.trainer import is_sagemaker_mp_enabled
        from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
        from transformers.trainer import ShardedDDPOption

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        logger.info('[Experimental] Make pretraining work with FSDP by setting learning rate of backbone to zero.')

        assert len(self.no_grad_params) > 0

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad) and n not in self.no_grad_params
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad) and n not in self.no_grad_params
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if n in self.no_grad_params
                    ],
                    "lr": 0.,
                },
            ]

            for p in opt_model.parameters():
                p.requires_grad = True

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                from transformers.trainer import OSS
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            from transformers.trainer import smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


