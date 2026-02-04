#!/bin/bash
./.venv/bin/python - <<'PY'
import os
import torch
from peft import PeftModel

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

base = "liuhaotian/llava-v1.5-7b"
adapter = "/scratch/s4723708/llava/llava_QCM-LEA"
out = "/scratch/s4723708/llava/llava_QCM-LEA-merged"

os.makedirs(out, exist_ok=True)

# Load full LLaVA base (vision + projector + LLM)
model_name = get_model_name_from_path(base)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    base, model_base=None, model_name=model_name
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(model, adapter)

# Merge LoRA into base weights
model = model.merge_and_unload()

# Save merged model (this becomes a full loadable model directory)
tokenizer.save_pretrained(out)
model.save_pretrained(out)
# also save image_processor config if present
try:
    image_processor.save_pretrained(out)
except Exception:
    pass

print("Merged model saved to:", out)
PY




python -m llava.eval.model_vqa_science \
    --model-path /scratch/$USER/llava/llava_QCM-A\
    --question-file /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/llava_test_QCM-A.json \
    --image-folder /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/images/test \
    --answers-file /scratch/s4723708/llava/answers/llava_QCM-A_pred.json \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

./.venv/bin/python -m llava.eval.model_vqa_science \
    --model-base liuhaotian/llava-v1.5-7b \
    --model-path /scratch/s4723708/llava/llava_QCM-A \
    --question-file /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/llava_test_QCM-A.json \
    --image-folder /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/images/test \
    --answers-file /scratch/s4723708/llava/llava_QCM-A/preds.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

./.venv/bin/python -m llava.eval.model_vqa_science \
    --model-base liuhaotian/llava-v1.5-7b \
    --model-path /scratch/s4723708/llava/llava_QCM-LEA-merged \
    --question-file /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/images/test \
    --answers-file /scratch/s4723708/llava/llava_QCM-LEA/preds.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

./.venv/bin/python -m llava.eval.model_vqa_science \
  --model-path /scratch/s4723708/llava/llava_QCM-LEA-merged \
  --question-file /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
  --image-folder /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/images/test \
  --answers-file /scratch/s4723708/llava/llava_QCM-LEA/preds.jsonl \
  --temperature 0 --conv-mode vicuna_v1

./.venv/bin/python -m llava.eval.model_vqa_science \
    --model-path /scratch/s4723708/llava/runs/QCM-A/latest/merged \
    --question-file /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/llava_test_QCM-A.json \
    --image-folder /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa/images/test \
    --answers-file /scratch/s4723708/llava/answers/QCM-A/latest_pred.jsonl \
    --output-format "letter" \
    --temperature 0 \
    --conv-mode llava_v1


python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json

./.venv/bin/python llava/eval/eval_science_qa.py \
    --base-dir /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa \
    --result-file /scratch/s4723708/llava/llava_QCM-A/preds.jsonl  \
    --output-file /scratch/s4723708/llava/llava_QCM-A/output.jsonl  \
    --output-result /scratch/s4723708/llava/llava_QCM-A/result.json


./.venv/bin/python llava/eval/eval_science_qa.py \
    --base-dir /scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa \
    --result-file /scratch/s4723708/llava/answers/QCM-EA/latest_pred.jsonl \
    --output-file /scratch/s4723708/llava/answers/QCM-EA/latest_output.jsonl \
    --output-result /scratch/s4723708/llava/answers/QCM-EA/latest_result.json



./.venv/bin/python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download

repo = "liuhaotian/llava-v1.5-7b"
src = hf_hub_download(repo_id=repo, filename="mm_projector.bin")
dst = "/scratch/s4723708/llava/llava_QCM-LEA/mm_projector.bin"

os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copyfile(src, dst)
print("Copied:", src, "->", dst)
PY