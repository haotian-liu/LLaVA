# Model Zoo

We'll add more model checkpoints into model zoo very soon. Stay tuned!

If you are interested in including any other details in Model Zoo, please open a issue :)

The model weights below are *merged* weights. You do not need to apply delta. The usage of LLaVA checkpoints should comply with the base LLM's model license: [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).

| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data | Finetuning schedule | LLaVA-Bench-Conv | LLaVA-Bench-Detail | LLaVA-Bench-Complex | LLaVA-Bench-Overall | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|------------------|--------------------|---------------------|---------------------|---------------------|
| Vicuna-13B-v1.3 | CLIP-L-336px | LCS-558K | 1e | LLaVA-Instruct-80K | proj-1e, lora-1e | 64.3 | 55.9 | 81.7 | 70.1 | [LoRA](https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3) [LoRA-Merged](https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3) |
| LLaMA-2-13B-Chat | CLIP-L | LCS-558K | 1e | LLaVA-Instruct-80K | full_ft-1e | 56.7 | 58.6 | 80.0 | 67.9 | [preview](https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview) |
| LLaMA-2-7B-Chat | CLIP-L | LCS-558K | 1e | LLaVA-Instruct-80K | lora-1e | 51.2 | 58.9 | 71.6 | 62.8 | [preview](https://huggingface.co/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview) |



## Legacy Models (merged weights)

The model weights below are *merged* weights. You do not need to apply delta. The usage of LLaVA checkpoints should comply with the base LLM's model license: [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).

| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data | Finetuning schedule | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|------------------|
| MPT-7B-Chat | CLIP-L | LCS-558K | 1e | LLaVA-Instruct-80K | full_ft-1e | [preview](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview) |


## Legacy Models (delta weights)

The model weights below are *delta* weights. The usage of LLaVA checkpoints should comply with the base LLM's model license: [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).

You can add our delta to the original LLaMA weights to obtain the LLaVA weights.

Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get LLaVA weights by applying our delta. It will automatically download delta weights from our Hugging Face account. In the script below, we use the delta weights of [`liuhaotian/LLaVA-7b-delta-v0`](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0) as an example. It can be adapted for other delta weights by changing the `--delta` argument (and base/target accordingly).

```bash
python3 -m llava.model.apply_delta \
    --base /path/to/llama-7b \
    --target /output/path/to/LLaVA-7B-v0 \
    --delta liuhaotian/LLaVA-7b-delta-v0
```

| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data | Finetuning schedule | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|------------------|
| Vicuna-13B-v1.1 | CLIP-L | CC-595K | 1e | LLaVA-Instruct-158K | full_ft-3e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1) |
| Vicuna-7B-v1.1 | CLIP-L | LCS-558K | 1e | LLaVA-Instruct-80K | full_ft-1e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1) |
| Vicuna-13B-v0 | CLIP-L | CC-595K | 1e | LLaVA-Instruct-158K | full_ft-3e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0) |
| Vicuna-13B-v0 | CLIP-L | CC-595K | 1e | ScienceQA | full_ft-12e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0-science_qa) |
| Vicuna-7B-v0 | CLIP-L | CC-595K | 1e | LLaVA-Instruct-158K | full_ft-3e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0) |
