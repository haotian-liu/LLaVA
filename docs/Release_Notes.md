# Release Notes

We document release notes here for each update.

### 7/19/2023

The first major update since our initial release!

- Added **LLaMA-2** support
- **Full LoRA support**. To make model training more accessible, we release a set of model weights based on LoRA, which supports training on academic resources (e.g. 4x A6000s, or 8x 3090s, **without the need of CPU offloading**)
- A more versatile design for training large multimodal models, including swapping different language models, vision encoders, and more coming soon
- Support higher resolution input using CLIP-ViT-L-336px as the vision encoder for a more detailed visual understanding
- Ablate and clean up some design choices to make the training simpler and smoother
- Full DeepSpeed support
- Improved model checkpoint saving during pretraining stage to save disk space
- Improved WebUI interface
- Improved support for inference with multiple-GPUs
- Support inference with 4-bit and 8-bit quantization
- Support interactive CLI inference

We train all models in this release using LLaVA-LCS-558K for pretraining and LLaVA-Instruct-80K for instruction tuning, to maintain an efficient and affordable training budget. **The full training (including both pretraining and finetuning) can be completed within 6 hours on 8x 3090s.**

*We hope this release further benefits the community and makes large multimodal models more accessible.*

#### Detailed Changes (7/19/2023)

- Tokenization. We remove the dependency of the additional tokens (`<IM_START>`, `<IM_END>`, `<IM_PATCH>`), so that during the pretraining stage, the tokenizer does not change at all and we only update the linear projector weights.
- Prompt.
    - Pretraining. We simplified the pretraining prompts by removing additional instructions like `Describe the image details`, which we find to allow the zero-shot inference and can slightly improve the training speed.
    - We keep the train/test prompt consistent, which we find to slightly improve the model's performance during the inference.

