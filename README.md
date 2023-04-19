# 🌋 LLaVA: Large Language and Vision Assistant

*Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*

[[Project Page](https://llava-vl.github.io/)] [[Paper](https://arxiv.org/abs/2304.08485)] [[Demo](https://llava.hliu.cc/)]  [[Data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)] [[Model](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0)]

**Visual Instruction Tuning** <br>
[Haotian Liu*](https://hliu.cc), [Chunyuan Li*](https://chunyuan.li/), [Qingyang Wu](https://scholar.google.ca/citations?user=HDiw-TsAAAAJ&hl=en/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (*Equal Contribution)



## Release

- 🔥 We released **LLaVA: Large Language and Vision Assistant**. We propose visual instruction tuning, towards building large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2304.08485) and [demo](https://llava.hliu.cc/).

<a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.


## Contents
- [Data Donwnload](#data-donwnload)
- [Install](#install)
- [LLaVA Weights](#llava-weights)
- [Serving](#serving)
- [Evaluation](#evaluation)
- [Fine-tuning](#fine-tuning)

## Data Donwnload

| Data file name | Size |
| --- | ---: |
| [conversation_58k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/conversation_58k.json) | 126 MB |
| [detail_23k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/detail_23k.json) | 20.5 MB |
| [complex_reasoning_77k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/complex_reasoning_77k.json) | 79.6 MB

To download our langauge-image multimodal instruction-folllowing dataset [`LLaVA-Instruct-150K`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K), please run the following script:
```bash
sh download_data.sh
```

## Install

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# For training
pip install ninja
pip install flash-attn
```

**NOTE**:
In this research preview, we used a modified version of huggingface/transformers library to support multimodal models and the LLaMA tokenizer.  Make sure that you are using the correct transformers library from https://github.com/haotian-liu/transformers_llava.

## LLaVA Weights
We release [LLaVA](https://llava-vl.github.io/) weights as delta weights to comply with the LLaMA model license.
You can add our delta to the original LLaMA weights to obtain the LLaVA weights. Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get LLaVA weights by applying [our delta](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0). It will automatically download delta weights from our Hugging Face account.

### LLaVA-13B
This conversion command needs around 60 GB of CPU RAM.
```bash
python3 -m llava.model.apply_delta \
    --base /path/to/llama-13b \
    --target /output/path/to/LLaVA-13B-v0 \
    --delta liuhaotian/LLaVA-13b-delta-v0
```

### LLaVA-7B
Coming soon.

## Serving

### Web UI

#### Launch a controller
```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a model worker
```Shell
python -m llava.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/LLaVA-13B-v0 --multi-modal
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Send a test message
```Shell
python -m llava.serve.test_message --model-name LLaVA-13B-v0 --controller http://localhost:10000
```

#### Launch a gradio web server.
```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000
```
#### You can open your browser and chat with a model now.

## Evaluation

### GPT-assisted Evaluation

Our GPT-assisted evaluation pipeline for multimodal modeling is provided for a comprehensive understanding of the capabilities of vision-language models.  Please see our paper for more details.

1. Generate LLaVA responses

```Shell
python model_vqa.py \
    --model-name ./checkpoints/LLaVA-13B-v0 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014_val \
    --answers-file \
    /path/to/answer-file.jsonl
```

2. Evaluate the generated responses

```Shell
OPENAI_API_KEY="sk-***********************************" python eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    /path/to/answer-file-1.jsonl \
    /path/to/answer-file-2.jsonl \
    --rule table/rule.json \
    --output /path/to/review.json
```

3. Summarize the evaluation results

```Shell
python summarize_gpt_review.py
```

### ScienceQA

Please see ScienceQA [repo](https://github.com/lupantech/ScienceQA) for setting up the dataset.  You may either use the official ScienceQA evaluation script or [our script](eval_science_qa.py) to evaluate the model.

## Fine-tuning
### Data

The current version of LLaVA is fine-tuned from a Vicuna-13B model.  We use approximately 600K filtered CC3M in feature alignment pretraining and 150K GPT-generated multimodal instruction-following data in finetuning. For detailed description of the data generation pipeline, please refer see our [paper](#).

We are working on a more capable model that is pretrained with the data at a larger scale.  Stay tuned!

We release all three types of multimodal instruction-following data.  The use of these data is subject to OpenAI [TOS](#).

### Code and Hyperparameters
We fine-tune the model using the code from [FastChat](https://github.com/lm-sys/FastChat). We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-13B | 128 | 2e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-13B | 32 | 2e-5 | 3 | 2048 | 0 |

### Fine-tuning with Local GPUs
LLaVA is trained on 8 A100 GPUs with 80GB memory with the following code. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly to keep the global batch size the same.

1. Pretraining

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    fastchat/train/train_mem.py \
    --model_name_or_path ./checkpoints/llama-vicuna-13b \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

2. Extract projector features
```Shell
python scripts/extract_mm_projector.py \
  --model_name_or_path ./checkpoints/llava-13b-pretrain \
  --output ./checkpoints/mm_projector/llava-13b-pretrain.bin
```

3. Finetuning

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    fastchat/train/train_mem.py \
    --model_name_or_path /path/to/llama-vicuna-13b \
    --data_path /path/to/llava_instruct_150k.json \
    --image_folder /Data/haotian/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/llava-13b-pretrain.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!


If you find LLaVA useful for your your research and applications, please cite using this BibTeX:
```bibtex
@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={arXiv:2304.08485},
      year={2023},
}
```

## Related Projects

For future project ideas, checkout the [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).
