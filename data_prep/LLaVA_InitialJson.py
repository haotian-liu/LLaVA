import os
import json

# Function to extract conversation data from .gui file
def extract_conversations_from_gui(gui_file):
    with open(gui_file, 'r') as file:
        gui_text = file.read().strip()
        return [{'from': 'human', 'value': '<image>\nWrite a code for the given UI'}, {'from': 'gpt', 'value': gui_text}]


# Function to convert data to JSON format
def convert_data_to_json(input_folder, output_folder):
    data = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.gui'):
            sample_id = filename.split('.')[0]
            image_path = "./Sketch2Code_og/" + os.path.relpath(os.path.join(input_folder, f"{sample_id}.png"), output_folder)
            gui_path = os.path.join(input_folder, filename)
            conversations = extract_conversations_from_gui(gui_path)
            sample = {
                'id': sample_id,
                'image': image_path,
                'conversations': conversations
            }
            data.append(sample)
    
    output_path = os.path.join(output_folder, 'samples.json')
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert data to JSON format")
    parser.add_argument("input_folder", help="Input folder containing .gui files and corresponding images")
    parser.add_argument("output_folder", help="Output folder to store the JSON file")
    
    args = parser.parse_args()
    
    convert_data_to_json(args.input_folder, args.output_folder)
