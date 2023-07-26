#!/bin/bash

CHUNKS=8
for IDX in {0..7}; do
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_science \
        --model-path ./checkpoints/LLaVA-13b-v0-science_qa \
        --question-file ~/haotian/datasets/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json \
        --image-folder ~/haotian/datasets/ScienceQA/data/scienceqa/images/test \
        --answers-file ./test_llava-13b-chunk$CHUNKS_$IDX.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --answer-prompter \
        --conv-mode llava_v0 &
done
