### ScienceQA

#### Prepare Data
1. Please see ScienceQA [repo](https://github.com/lupantech/ScienceQA) for setting up the dataset.
2. Generate ScienceQA dataset for LLaVA conversation-style format.

```Shell
python scripts/convert_sqa_to_llava.py \
    convert_to_llava \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --prompt-format "QCM-LEA" \
    --split {train,val,minival,test,minitest}
```

#### Training

1. Pretraining

You can download our pretrained projector weights from our [Model Zoo](), or train your own projector weights using [`pretrain.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/pretrain.sh).

2. Finetuning

See [`finetune_sqa.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/finetune_sqa.sh).

#### Evaluation

1. Multiple-GPU inference
You may evaluate this with multiple GPUs, and concatenate the generated jsonl files.  Please refer to our script for [batch evaluation](https://github.com/haotian-liu/LLaVA/blob/main/scripts/sqa_eval_batch.sh) and [results gathering](https://github.com/haotian-liu/LLaVA/blob/main/scripts/sqa_eval_gather.sh).

2. Single-GPU inference

(a) Generate LLaVA responses on ScienceQA dataset

```Shell
python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
    --question-file /path/to/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /path/to/ScienceQA/data/scienceqa/images/test \
    --answers-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --conv-mode llava_v1
```

(b) Evaluate the generated responses

```Shell
python eval_science_qa.py \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --result-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --output-file vqa/results/ScienceQA/test_llava-13b_output.json \
    --output-result vqa/results/ScienceQA/test_llava-13b_result.json \
```

For reference, we attach our prediction file [`test_sqa_llava_lcs_558k_sqa_12e_vicuna_v1_3_13b.json`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/table/results/test_sqa_llava_lcs_558k_sqa_12e_vicuna_v1_3_13b.json) and [`test_sqa_llava_13b_v0.json`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/table/results/test_sqa_llava_13b_v0.json) for comparison when reproducing our results, as well as for further analysis in detail.
