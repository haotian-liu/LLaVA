import os
import tkinter as tk
from tkinter import ttk  # Import ttk module
from tkinter import filedialog
from PIL import Image
import torch
import warnings
import threading

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from deep_translator import GoogleTranslator

warnings.filterwarnings("ignore")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Load the pretrained model
model_path = "liuhaotian/llava-v1.6-mistral-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_4bit=True
)

print(f"Model ready on {device}")

def check_auth(username, password):
    return password == "fv"

def process_single_image(instruction, images, translation_language):


    outputTexts = []

    for image in images:
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
                max_new_tokens=512,
                use_cache=True
            )

            # Update the output in the output_textbox
            outputText = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if translation_language != "en":
                outputText = GoogleTranslator(source="auto", target=translation_language).translate(outputText)

            # Also append the filename
            outputText = f"Image: {os.path.basename(image.filename)}\n\n{outputText}"
            
            outputTexts.append(outputText)
    outputTexts = "\n\n-----\n\n".join(outputTexts)

    return outputTexts