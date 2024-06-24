import os
import tkinter as tk
from tkinter import ttk  # Import ttk module
from tkinter import filedialog
from PIL import Image
import torch
import warnings
import threading
from transformers import AutoTokenizer, MistralForCausalLM, set_seed, AutoModelForCausalLM
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import gc

warnings.filterwarnings("ignore")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

def check_auth(username, password):
    return password == "fv"


def process_single_image(instruction, images, should_translate):

    global tokenizer, model, t_tokenizer, t_model

    outputTexts = []
    count = 1
    # Load the pretrained model (Image + Text)
    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        load_4bit=True
    )

    print(f"Model ready on {device}")


    # Load the pretrained model (Translation Text)
    t_tokenizer = AutoTokenizer.from_pretrained("mayflowergmbh/occiglot-7b-de-en-instruct-hf", use_fast=True)
    t_model = AutoModelForCausalLM.from_pretrained("mayflowergmbh/occiglot-7b-de-en-instruct-hf", load_in_4bit=True)
    # t_model.to(device, dtype=torch.float4)

    print(f"Translation model ready on {device}")

    for image in images:

        print(f"\n\n\nProcessing image {count}\n\n\n")

        prompt = f"Instruction: {instruction} Answer:"

        image = Image.open(image)

        # Process the image
        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

        # Prepare the input_ids
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        qs = image_token_se + "\n" + prompt
        input_ids = tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

        # Generate the output
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                do_sample=True if 0 > 0 else False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=500,
                use_cache=True
            )

            # Update the output in the output_textbox
            outputText = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(f"Output: {outputText}")

        if should_translate:
            # del model
            # torch.cuda.empty_cache()
            outputText = translate_text(outputText)

        outputText = f"Image: {os.path.basename(image.filename)}\n\n{outputText}"
        
        outputTexts.append(outputText)
        count += 1


    outputTexts = "\n\n-----\n\n".join(outputTexts)

    unload_models()

    return outputTexts

def translate_text(text):

    global t_tokenizer, t_model

    messages = [
   {"role": "system", 'content': 'You are a helpful translator. Translate the following to German.'},
   {"role": "user", "content": f"{text}"},
    ]

    # Prepare the input_ids
    tokenized_chat = t_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False, return_tensors='pt',)
    outputs = t_model.generate(tokenized_chat.to('cuda'), max_new_tokens=500,)

    # Update the output in the output_textbox
    outputText = tokenizer.decode(outputs[0][len(tokenized_chat[0]):])
    outputText = outputText.replace("</s>", "").strip()
    outputText = outputText.replace("<|im_end|>", "").strip()
    print(f"Translated: {outputText}")
    # del model
    # torch.cuda.empty_cache() 
    return outputText

def unload_models():
    global model, t_model
    del model
    del t_model
    torch.cuda.empty_cache()
    gc.collect()
    print("\n\n\n[+] Finished")