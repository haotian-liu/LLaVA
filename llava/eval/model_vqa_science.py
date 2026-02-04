import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import re

def extract_option_letters(text):
    # Matches (A), (B), (C), ...
    letters = re.findall(r"\(([A-Z])\)", text)
    # Deduplicate but preserve order
    seen = set()
    return [l for l in letters if not (l in seen or seen.add(l))]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def add_output_format_instruction(qs: str, fmt: str) -> str:
    opts = extract_option_letters(qs)
    opts_or = " or ".join(opts)

    if fmt == "letter":
        return qs + f"\nRespond with one letter: {opts_or}."

    elif fmt == "cot_reason_first":
        return qs + (
            "\nOutput format (exactly two lines):\n"
            "Reasoning: 1-3 sentences.\n"
            f"Final answer: one letter from {opts_or}.\n"
            "Do not write anything else."
        )

    elif fmt == "cot_answer_first":
        return qs + (
            "\nOutput format (exactly two lines):\n"
            f"Final answer: one letter from {opts_or}.\n"
            "Reasoning: 1-3 sentences.\n"
            "Do not write anything else."
        )

    else:
        return qs




def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        device_map=None,
        device="cuda"
    )

    model = model.to("cuda")
    device = torch.device("cuda")

    # Ensure vision tower + image processor are loaded (some configs delay-load)
    if image_processor is None:
        vt = model.get_vision_tower()
        if isinstance(vt, list):
            vt = vt[0]
        if hasattr(vt, "load_model"):
            vt.load_model()
        image_processor = vt.image_processor

    # move vision tower
    vt = model.get_vision_tower()
    if isinstance(vt, list):
        vt = vt[0]
    if hasattr(vt, "to"):
        vt.to(device)

    # move projector
    model.get_model().mm_projector.to(device)

    # ---------------------------------------------------------------------------


    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().to(device)
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        qs = add_output_format_instruction(qs, args.output_format)
        cur_prompt = add_output_format_instruction(cur_prompt, args.output_format)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        max_new = 32 if args.output_format == "letter" else 256
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=256, #max_new,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--output-format", type=str, default="letter", choices=["letter", "cot_reason_first", "cot_answer_first"], help="Force model output format.")
    args = parser.parse_args()

    eval_model(args)
