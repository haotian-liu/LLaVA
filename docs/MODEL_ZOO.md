# Model Zoo

We'll add more model checkpoints into model zoo very soon. Stay tuned!

If you are interested in including any other details in Model Zoo, please open a issue :)

| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data | Finetuning schedule | LLaVA-Bench-Conv | LLaVA-Bench-Detail | LLaVA-Bench-Complex | LLaVA-Bench-Overall | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|------------------|--------------------|---------------------|---------------------|---------------------|
| Vicuna-13B-v1.3 | CLIP-L-336px | LCS-558K | 1e | LLaVA-Instruct-80K | proj-1e, lora-1e | 64.3 | 55.9 | 81.7 | 70.1 | [LoRA](https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3) [LoRA-Merged](https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3) |
| LLaMA-2-13B-Chat | CLIP-L | LCS-558K | 1e | LLaVA-Instruct-80K | full_ft-1e | 56.7 | 58.6 | 80.0 | 67.9 | [preview](https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview) |
| LLaMA-2-7B-Chat | CLIP-L | LCS-558K | 1e | LLaVA-Instruct-80K | lora-1e | 51.2 | 58.9 | 71.6 | 62.8 | [preview](https://huggingface.co/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview) |



## Legacy Models (merged weights)

| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data | Finetuning schedule | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|------------------|
| MPT-7B-Chat | CLIP-L | LCS-558K | 1e | LLaVA-Instruct-80K | full_ft-1e | [preview](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview) |


## Legacy Models (delta weights)

| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data | Finetuning schedule | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|------------------|
| Vicuna-13B-v1.1 | CLIP-L | CC-595K | 1e | LLaVA-Instruct-158K | full_ft-3e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1) |
| Vicuna-7B-v1.1 | CLIP-L | LCS-558K | 1e | LLaVA-Instruct-80K | full_ft-1e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1) |
| Vicuna-13B-v0 | CLIP-L | CC-595K | 1e | LLaVA-Instruct-158K | full_ft-3e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0) |
| Vicuna-13B-v0 | CLIP-L | CC-595K | 1e | ScienceQA | full_ft-12e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0-science_qa) |
| Vicuna-7B-v0 | CLIP-L | CC-595K | 1e | LLaVA-Instruct-158K | full_ft-3e | [delta-weights](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0) |
