import subprocess
import os
import shutil
import tarfile
import tempfile
import zipfile
import json


from cog import BaseModel, Input, Path
from llava.utils import disable_torch_init
from file_utils import is_url, download_file, download_weights, REPLICATE_WEIGHTS_URL, DEFAULT_WEIGHTS

# we don't use the huggingface hub cache, but we need to set this to a local folder
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/models"

def check_zip_contents(zip_path):
    # Check if the ZIP file contains 'data.json' and a folder named 'images' in root
    error_msgs = []
    train_data_has_right_structure = True
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all contents of the zip file
            zip_contents = zip_ref.namelist()

            if "data.json" not in zip_contents:
                wrong_locations = [item for item in zip_contents if "data.json" in item]
                data_json_msg = f"{zip_path} does not contain a file named data.json in root. This file might be in the wrong location: {', '.join(wrong_locations)}"
                error_msgs.append(data_json_msg)
                train_data_has_right_structure = False

            if "images/" not in zip_contents:
                images_folder_msg = f"{zip_path} does not contain a folder named images in root."
                error_msgs.append(images_folder_msg)
                train_data_has_right_structure = False

            if "data.json" in zip_contents:
                # Read and load the content of 'data.json'
                with zip_ref.open("data.json", 'r') as data_json_file:
                    data_json_content = json.load(data_json_file)
                for datapoint in data_json_content:
                    img_filename = "images/" + datapoint.get('image')
                    if not img_filename in zip_contents:
                        missing_file_str = f"data.json refers to image {img_filename}, but this file is missing in {zip_path}"
                        error_msgs.append(missing_file_str)
                        train_data_has_right_structure = False
    except zipfile.BadZipFile:
        badzip_msg = f"File '{zip_path}' is not a valid ZIP file or is corrupted."
        error_msgs.append(badzip_msg)
        print(badzip_msg)
        train_data_has_right_structure = False
    return train_data_has_right_structure, error_msgs

def run_training(
        image_folder: Path,
        data_path: Path,
        output_dir: Path,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-4,
        model_max_length: int = 2048
    ):
    # Command and arguments as a list
    command = [
        'python',
        '-m',
        'deepspeed.launcher.runner',
        'llava/train/train_mem.py',
        '--model_name_or_path', 'liuhaotian/llava-v1.5-13b',
        '--data_path', data_path,
        '--image_folder', image_folder,
        '--vision_tower', 'openai/clip-vit-large-patch14-336',
        '--output_dir', output_dir,
        '--lora_enable', 'True',
        '--lora_r', '128',
        '--lora_alpha', '256',
        '--mm_projector_lr', '2e-5',
        '--save_steps', '500',
        '--deepspeed', './scripts/zero3.json',
        '--version', 'v1',
        '--mm_projector_type', 'mlp2x_gelu',
        '--mm_vision_select_layer', '-2',
        '--mm_use_im_start_end', 'False',
        '--mm_use_im_patch_token', 'False',
        '--image_aspect_ratio', 'pad',
        '--group_by_modality_length', 'True',
        '--bf16', 'True',
        '--num_train_epochs', str(num_train_epochs),
        '--per_device_train_batch_size', '16',
        '--per_device_eval_batch_size', '4',
        '--gradient_accumulation_steps', '1',
        '--evaluation_strategy', 'no',
        '--save_strategy', 'steps',
        '--save_total_limit', '1',
        '--learning_rate', str(learning_rate),
        '--weight_decay', '0.',
        '--warmup_ratio', '0.03',
        '--lr_scheduler_type', 'cosine',
        '--logging_steps', '1',
        '--tf32', 'True',
        '--model_max_length', str(model_max_length),
        '--gradient_checkpointing', 'True',
        '--dataloader_num_workers', '4',
        '--lazy_preprocess', 'True',
        '--report_to', 'none'
    ]

    # Execute the command
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    subprocess.run(command, env=env, check=True)




class TrainingOutput(BaseModel):
    # this must be a key named `weights`, otherwise image creation will silently fail
    # source: https://github.com/replicate/api/blob/6b73b27e0da6afbea0531bb4162e9b4f5a74d744/pkg/server/internal.go#L282 
    weights: Path

def train(
    train_data: str = Input(description="https url or path name of a zipfile containing training data. Training data should have a json file data.json and an images/ folder. data.json should link the images from images/ to conversations."),
    num_train_epochs: int = Input(description="The number of training epochs", ge=1, le=1000, default=1),
    learning_rate: float = Input(description="The learning rate during training", ge=1e-10, default=2e-4),
    model_max_length: int = Input(description="The maximum length (in number of tokens) for the inputs to the model.", ge=1, default=2048),
    ) -> TrainingOutput:
    
    # check the structure of the train_data zipfile
    train_data_has_right_structure, errors = check_zip_contents(train_data)
    if not train_data_has_right_structure:
        raise ValueError(f"There was a problem with the training data in {train_data}:\n\n" + "\n".join(errors))
    
    # download base models
    for weight in DEFAULT_WEIGHTS:
        download_weights(weight["src"], weight["dest"], weight["files"])
    disable_torch_init()
    
    # Path to the weights file
    weights_file = Path("my_weights.tar")

    # Remove old output tar if it exists
    if weights_file.exists():
        weights_file.unlink()

    # Create a temporary directory to unzip train_data
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)

        # Download train_data if it is a URL
        if is_url(train_data):
            local_train_data_path = tmp_dir / "train_data_archive.zip"
            download_file(train_data, local_train_data_path)
        else:
            local_train_data_path = Path(train_data)

        # Unzip train_data into tmp_dir
        # todo think about safety
        shutil.unpack_archive(str(local_train_data_path), tmp_dir)

        # Define paths to data_path, image_folder, and output_dir within tmp_dir
        data_path = tmp_dir / "data.json"
        image_folder = tmp_dir / "images"
        output_dir = tmp_dir / "output"

        # Make sure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run the training command
        run_training(
            image_folder,
            data_path, 
            output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            model_max_length=model_max_length
        )

        # Tar the checkpoints and put into weights_file without compression
        with tarfile.open(str(weights_file), "w") as tar:
            tar.add(output_dir, arcname="")

    # Return the path to the weights file
    return TrainingOutput(weights=weights_file)

# todo: deal with recursive lora

