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


instruction = ""

# Define the query
prompt = f"Instruction: {instruction} Answer:"

# Specify the directory containing the images
image_directory = ""

def choose_directory():
    global image_directory
    image_directory = filedialog.askdirectory()

def start_processing():
    global instruction
    instruction = instruction_entry.get()
    global prompt
    prompt = f"Instruction: {instruction} Answer:"

    if image_directory == "":
        print("Please choose a directory first.")
        return
    global maxImg
    maxImg = len([name for name in os.listdir(image_directory) if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg")])

    # Create a separate thread for image processing
    def process_images_thread():
        outputs = ""
        curImg = 0
        for filename in os.listdir(image_directory):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                curImg += 1
                output_textbox.insert(tk.END, f"Processing image {curImg}/{maxImg}...\n")
                print(f"\n---\n\n{filename}\n\n")
                # Load the image
                image_path = os.path.join(image_directory, filename)
                image = Image.open(image_path).convert("RGB")

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
                    output_textbox.insert(tk.END, f"{filename}\n------------\n")
                    outputText = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    if translate_output.get():  # Check the state of the checkbox
                        outputText = GoogleTranslator(source='auto', target=selected_language.get()).translate(text=outputText)  # Translate the text
                    output_textbox.insert(tk.END, f"{outputText}\n\n")
        if curImg == 0:
            output_textbox.insert(tk.END, "No output.\n")
        output_textbox.see(tk.END)


    # Start the image processing thread
    thread = threading.Thread(target=process_images_thread)
    thread.start()

# Create the GUI
root = tk.Tk()

root.title("Fluffyvision")
root.resizable(False, False)
icoFile = os.path.join(os.path.dirname(__file__), "icon.ico")
root.iconbitmap(icoFile)
root.geometry("1000x800")

# Create a BooleanVar for the checkbox
translate_output = tk.BooleanVar()

# Create a list of language codes
languages = ["de", "en", "fr", "es", "it", "nl", "ru"]

# Selected language
selected_language = tk.StringVar()

choose_directory_button = tk.Button(root, text="Choose Directory", command=choose_directory)
choose_directory_button.pack(pady=5)

instruction_label = tk.Label(root, text="Enter Instruction [English]:")
instruction_label.pack(pady=10)

instruction_entry = tk.Entry(root)
instruction_entry.config(width=100)
instruction_entry.pack(pady=5)

# Create a Checkbutton for translation
translate_checkbutton = ttk.Checkbutton(root, text="Translate output?", variable=translate_output)
translate_checkbutton.pack(pady=5)

# Create a Combobox for language selection
language_combobox = ttk.Combobox(root, values=languages, textvariable=selected_language)
language_combobox.pack(pady=5)

start_button = tk.Button(root, text="Start", command=start_processing)
start_button.pack(pady=5)

output_text = tk.StringVar()
scrollbar = tk.Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

output_textbox = tk.Text(root, wrap=tk.WORD, yscrollcommand=scrollbar.set)
output_textbox.pack(pady=10)

scrollbar.config(command=output_textbox.yview)

root.mainloop()