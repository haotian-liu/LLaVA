#!/bin/bash

CHUNKS=8
output_file="test_llava-13b.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for idx in $(seq 0 $((CHUNKS-1))); do
  cat "./test_llava-13b-chunk${idx}.jsonl" >> "$output_file"
done

python llava/eval/eval_science_qa.py \
    --base-dir ~/haotian/datasets/ScienceQA/data/scienceqa \
    --result-file ./test_llava-13b.jsonl \
    --output-file ./test_llava-13b_output.json \
    --output-result ./test_llava-13b_result.json
