# Run LLaVA on macOS

*NOTE: LLaVA on macOS is not fully supported. Currently we only support 16-bit inference. More functionalities on macOS is to be added soon, stay tuned.*

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
python -mpip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install torch==2.1.0 torchvision==0.16.0
pip uninstall bitsandbytes
```

## Run demo

Specify `--device mps` when launching model worker or CLI.

See instructions [here](https://github.com/haotian-liu/LLaVA#demo).

Note that quantization (4-bit, 8-bit) is *NOT* supported on macOS. Stay tuned for the 4-bit support on macOS!
