#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-mistral-7b-pretrain \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-pretrain.jsonl \
    --model-base mistralai/Mistral-7B-v0.1 \
    --single-pred-prompt \
    --temperature 0

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-pretrain.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-pretrain_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-mistral-7b-pretrain_result.json
