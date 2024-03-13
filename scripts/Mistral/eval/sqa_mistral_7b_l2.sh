#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-mistral-7b-pretrain \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-l2.jsonl \
    --model-base mistralai/Mistral-7B-v0.1 \
    --single-pred-prompt \
    --conv_mode mistral_instruct \
    --temperature 0

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-l2.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-l2-output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-mistral-7b-l2-result.json
