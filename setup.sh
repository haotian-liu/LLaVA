export CUDA_HOME=/usr/local/cuda-11.8
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"

#conda create -n llava python=3.10 -y
#conda activate llava
python -m pip install --upgrade pip

python -m pip install -e .
python -m pip install -e ".[train]"
python -m pip install torch==2.1.2 torchvision==0.16.2 triton==2.1.0 accelerate==0.26.1 deepspeed==0.13.1 pynvml==11.5.0 --upgrade
python -m pip install "sglang[all]"
python -m pip install flash-attn==2.5.2 --no-build-isolation 
