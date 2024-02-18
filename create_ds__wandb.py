
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import uuid

def save_dataset(dataset_name, output_folder, class_name, subset_name, val_samples=None):
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, split=subset_name)

    # Filter for images with the specified class in 'question_type'
    filtered_dataset = [item for item in dataset if item['question_type'] == class_name]

    # Determine the split for training and validation
    if val_samples is not None and subset_name == 'train':
        train_dataset = filtered_dataset[val_samples:]
        val_dataset = filtered_dataset[:val_samples]
    else:
        train_dataset = filtered_dataset
        val_dataset = []

    # Process and save the datasets
    for subset, data in [('train', train_dataset), ('validation', val_dataset)]:
        if data:
            process_and_save(data, output_folder, subset)

def process_and_save(dataset, output_folder, subset_name):
    # Define image subfolder within output folder
    image_subfolder = os.path.join(output_folder, 'images')
    if not os.path.exists(image_subfolder):
        os.makedirs(image_subfolder)

    # Initialize list to hold all JSON data
    json_data_list = []

    # Process and save images and labels
    for item in dataset:
        # Load image if it's a URL or a file path
        if isinstance(item['image'], str):
            response = requests.get(item['image'])
            image = Image.open(BytesIO(response.content))
        else:
            image = item['image']  # Assuming it's a PIL.Image object

        # Create a unique ID for each image
        unique_id = str(uuid.uuid4())

        # Define image path
        image_path = os.path.join(image_subfolder, f"{unique_id}.jpg")

        # Save image
        image.save(image_path)

        # Remove duplicates and format answers
        answers = item['answers']
        unique_answers = list(set(answers))
        formatted_answers = ", ".join(unique_answers)

        # Structure for LLaVA JSON
        json_data = {
            "id": unique_id,
            "image": f"{unique_id}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": item['question']
                },
                {
                    "from": "gpt",
                    "value": formatted_answers
                }
            ]
        }

        # Append to list
        json_data_list.append(json_data)

    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, 'dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)

# Usage example
output_folder = 'dataset'
class_name = 'other'
val_samples = 300
save_dataset('Multimodal-Fatima/OK-VQA_train', output_folder, class_name, 'train', val_samples)
save_dataset('Multimodal-Fatima/OK-VQA_test', output_folder, class_name, 'test')

