import os
import json
import random
import shutil
import argparse

def split_dataset(input_folder, output_folder, test_split=10, validation_split=10, seed=42):
    # Create output folders for train, test, and validation sets
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    validation_folder = os.path.join(output_folder, 'validation')
    split_json_folder = os.path.join(output_folder, 'split_json_files')

    for folder in [train_folder, test_folder, validation_folder, split_json_folder]:
        os.makedirs(folder, exist_ok=True)

    # Load the JSON file containing the data
    json_file_path = os.path.join(input_folder, 'samples.json')
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Calculate the number of samples for test and validation sets
    num_samples = len(data)
    num_test_samples = num_samples * test_split // 100
    num_validation_samples = num_samples * validation_split // 100

    # Set random seed for reproducibility
    random.seed(seed)

    # Randomly select indices for test set
    test_indices = random.sample(range(num_samples), num_test_samples)

    # Remove test indices from the list of all indices
    remaining_indices = [i for i in range(num_samples) if i not in test_indices]

    # Randomly select indices for validation set from remaining indices
    validation_indices = random.sample(remaining_indices, num_validation_samples)

    # The remaining indices are for the training set
    train_indices = [i for i in remaining_indices if i not in validation_indices]

    # Copy files to corresponding folders and update JSON files
    for idx, sample in enumerate(data):
        source_image = sample['image']
        source_gui = os.path.join(input_folder, 'data', f"{sample['id']}.gui")
        destination_folder = None
        if idx in test_indices:
            destination_folder = test_folder
        elif idx in validation_indices:
            destination_folder = validation_folder
        else:
            destination_folder = train_folder

        # Copy files to destination folder
        image_filename = os.path.basename(source_image)
        gui_filename = f"{sample['id']}.gui"
        destination_image = os.path.join(destination_folder, image_filename)
        destination_gui = os.path.join(destination_folder, gui_filename)
        shutil.copy(source_image, destination_image)
        shutil.copy(source_gui, destination_gui)

        # Update JSON data with relative paths
        relative_image_path = os.path.relpath(destination_image, output_folder)
        relative_gui_path = os.path.relpath(destination_gui, output_folder)
        sample['image'] = "./" + relative_image_path
        sample['gui'] = "./" + relative_gui_path

    # Create updated JSON files for each split
    splits = {'train': train_indices, 'test': test_indices, 'validation': validation_indices}
    for split, indices in splits.items():
        split_data = [data[i] for i in indices]
        split_json_path = os.path.join(split_json_folder, f"{split}_json.json")
        with open(split_json_path, 'w') as json_file:
            json.dump(split_data, json_file, indent=2)

    print("Dataset splitting and JSON file creation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset and generate JSON files")
    parser.add_argument("input_folder", help="Path to the input folder containing the dataset")
    parser.add_argument("output_folder", help="Path to the output folder to save the split dataset")
    parser.add_argument("--test_split", type=int, default=10, help="Percentage of data to use for test (default: 10)")
    parser.add_argument("--validation_split", type=int, default=10, help="Percentage of data to use for validation (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    split_dataset(args.input_folder, args.output_folder, args.test_split, args.validation_split, args.seed)
