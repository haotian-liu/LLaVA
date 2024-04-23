# Run Llava on AscendNPU



## Installation
1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
```

4. Install Ascend Extension for PyTorch

You can follow this [guide](https://www.hiascend.com/document/detail/en/ModelZoo/pytorchframework/ptes/ptes_00001.html) to download and install the Ascend NPU Firmware, Ascend NPU Driver, and CANN. Afterwards, you need to install additional Python packages.
```shell
pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu  #For X86
pip3 install torch==2.1.0  #For Aarch64
pip3 install accelerate==0.28.0 decorator==5.1.1 scipy==1.13.0 attrs==23.2.0 openpyxl
```
After installing the above Python packages,
You can follow this [README](https://github.com/Ascend/pytorch/blob/master/README.md) to install the torch_npu environment.
Then you can use Llava on Ascend NPU.




## Pretrain/Finetune Llava on AscendNPU
If you want to Pretrain/Finetune Llava on AscendNPU, you only need to make modifications to two lines in the Pretrain/Finetune shell script.

As shown below:
```shell
# Firstly, add environment variables to the system via the 'source' command.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Disable TF32 mode
--tf32 False
```
Here is [finetune shell](scripts/v1_5/finetune_npu.sh) example on AscendNPU


## Inference/Evaluate Llava on AscendNPU
If you want to perform inference/evaluation, a small modification to your shell script is all that's needed.


As shown below, you only need to add a 'source' command in your shell script,and the usage for inference remains the same.
```shell
# textvqa.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh #Add this
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl

# inference.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh #Add this
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \

```
*NOTE:Ascend NPU doesn't support all quantization methods. If you encounter issues during inference, you can remove the quantization.*



