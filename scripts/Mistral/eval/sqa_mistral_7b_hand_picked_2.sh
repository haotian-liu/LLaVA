#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-mistral-7b-lora-hand-picked-2 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-hand-picked-2.jsonl \
    --model-base liuhaotian/llava-v1.6-mistral-7b  \
    --single-pred-prompt \
    --conv-mode mistral_instruct \
    --temperature 0

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-hand-picked-2.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-mistral-7b-hand-picked-output-2.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-mistral-7b-hand-picked-result-2.json
