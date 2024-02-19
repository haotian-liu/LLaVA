import os
import time
import subprocess
from urllib.parse import urlparse

import requests
from cog import Path

# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"

# files to download from the weights mirrors
DEFAULT_WEIGHTS = [
    {
        "dest": "liuhaotian/llava-v1.5-13b",
        # git commit hash from huggingface
        "src": "llava-v1.5-13b/006818fc465ebda4c003c0998674d9141d8d95f8",
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00003.bin",
            "pytorch_model-00002-of-00003.bin",
            "pytorch_model-00003-of-00003.bin",
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
    },
    {
        "dest": "openai/clip-vit-large-patch14-336",
        "src": "clip-vit-large-patch14-336/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin"
        ],
    }
]


def is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def download_file(url: str, local_path: Path) -> None:
    # Stream download to handle large files
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

def download_weights(baseurl: str, basedest: str, files: list[str]):
    """Download model weights from Replicate and save to file.
    Weights and download locations are specified in DEFAULT_WEIGHTS
    """
    basedest = Path(basedest)
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(REPLICATE_WEIGHTS_URL, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                download_json(url, dest)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)