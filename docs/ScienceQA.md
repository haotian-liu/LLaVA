### ScienceQA

#### Prepare Data
1. Please see ScienceQA [repo](https://github.com/lupantech/ScienceQA) for setting up the dataset.
2. Generate ScienceQA dataset for LLaVA conversation-style format.

```Shell
python scripts/convert_sqa_to_llava \
    convert_to_llava \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --split {train,val,minival,test,minitest}
```

#### Training
**NOTE**: Due to that ScienceQA experiments were done earlier, the current checkpoints are trained *without* `<im_start>` and `<im_end>` tokens. Here we provide our training scripts for the current checkpoints.

<details>
<summary>1. Pretraining</summary>

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/llama-vicuna-13b \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain-no_im_start_end_token \
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
</details>

<details>
<summary>2. Finetuning</summary>

You may download our pretrained `llava-13b-v0-pretrain-no_im_start_end_token.bin` [here](https://huggingface.co/liuhaotian/LLaVA-13b-pretrain-projector-v0/blob/main/LLaVA-13b-pretrain-projector-v0-CC3M-595K-original_caption-no_im_token.bin).

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /path/to/llama-vicuna-13b \
    --data_path /path/to/scienceqa/llava_train_QCM-LEPA.json \
    --image_folder /path/to/scienceqa/images/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-13b-pretrain-no_im_start_end_token/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain-no_im_start_end_token-finetune_scienceqa \
    --num_train_epochs 12 \
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
</details>

#### Evaluation

1. Download our pretrained LLaVA-13B (delta) weights for ScienceQA dataset [here](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0-science_qa).  Convert the delta weights to actual weights.

```Shell
python -m llava.model.apply_delta \
    --base /path/to/llama-13b \
    --target /path/to/LLaVA-13b-v0-science_qa \
    --delta liuhaotian/LLaVA-13b-delta-v0-science_qa
```

2. [Option 1] Multiple-GPU inference
You may evaluate this with multiple GPUs, and concatenate the generated jsonl files.  Please refer to our script for [batch evaluation](https://github.com/haotian-liu/LLaVA/blob/main/scripts/sqa_eval_batch.sh) and [results gathering](https://github.com/haotian-liu/LLaVA/blob/main/scripts/sqa_eval_gather.sh).

3. [Option 2] Single-GPU inference

(a) Generate LLaVA responses on ScienceQA dataset

```Shell
python -m llava.eval.model_vqa_science \
    --model-path /path/to/LLaVA-13b-v0-science_qa \
    --question-file /path/to/ScienceQA/data/scienceqa/llava_test.json \
    --image-folder /path/to/ScienceQA/data/scienceqa/images/test \
    --answers-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --answer-prompter \
    --conv-mode llava_v0
```

(b) Evaluate the generated responses

```Shell
python eval_science_qa.py \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --result-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --output-file vqa/results/ScienceQA/test_llava-13b_output.json \
    --output-result vqa/results/ScienceQA/test_llava-13b_result.json \
```

For reference, we attach our prediction file [`test_llava-13b_result.json`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/table/results/test_sqa_llava_13b_v0.json) for comparison when reproducing our results, as well as for further analysis in detail.
