import subprocess
import os
from cog import BaseModel, Input, Path
import shutil
import tarfile
import tempfile
from typing import NamedTuple
import requests
from urllib.parse import urlparse

def run_training(image_folder: Path, data_path: Path, output_dir: Path):
    # Command and arguments as a list
    command = [
        'deepspeed',
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
        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '16',
        '--per_device_eval_batch_size', '4',
        '--gradient_accumulation_steps', '1',
        '--evaluation_strategy', 'no',
        '--save_strategy', 'steps',
        '--save_total_limit', '1',
        '--learning_rate', '2e-4',
        '--weight_decay', '0.',
        '--warmup_ratio', '0.03',
        '--lr_scheduler_type', 'cosine',
        '--logging_steps', '1',
        '--tf32', 'True',
        '--model_max_length', '2048',
        '--gradient_checkpointing', 'True',
        '--dataloader_num_workers', '4',
        '--lazy_preprocess', 'True',
        '--report_to', 'none'
    ]

    # Execute the command
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    subprocess.run(command, env=env, check=True)


def download_file(url: str, local_path: Path) -> None:
    # Stream download to handle large files
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

class TrainingOutput(BaseModel):
    output_weights: Path

# todo: download_weights
def train(train_data: str = Input(description="https url or path name of a file containing training data. Training data should have a json file data.json and an images/ folder. data.json should link the images from images/ to conversations.")) -> TrainingOutput:
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
            local_train_data_path = tmp_dir / "train_data_archive"
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
        run_training(image_folder, data_path, output_dir)

        # Tar the checkpoints and put into weights_file without compression
        with tarfile.open(str(weights_file), "w") as tar:
            tar.add(output_dir, arcname="")

    # Return the path to the weights file
    return TrainingOutput(output_weights=weights_file)

# todo: deal with recursive lora

