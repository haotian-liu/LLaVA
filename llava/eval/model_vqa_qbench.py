import argparse
import torch
from tqdm import tqdm
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, True)



    
    with open(args.questions_file) as f:
        llvqa_data = json.load(f)  
        
    for i, llddata in enumerate(tqdm(llvqa_data)):
        filename = llddata["img_path"]
        if args.lang == "en":
            message = llddata["question"] + "\nChoose between one of the options as follows:\n"
        elif args.lang == "zh":
            message = llddata["question"] + "\在下列选项中选择一个:\n"
        else:
            raise NotImplementedError("Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/VQAssessment/Q-Bench/) to convert  Q-Bench into more languages.")
        for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
            message += f"{choice} {ans}\n"
        qs = message
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = load_image(args.image_folder + filename)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                num_beams=1,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        llddata["response"] = outputs
        with open(args.answers_file, "a") as wf:
            json.dump(llddata, wf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-v1.5")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/qbench/images_llvisionqa")
    parser.add_argument("--questions-file", type=str, default="./playground/data/qbench/llvisionqa_dev.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
