from llava.train.llama_npu_monkey_patch import (
    replace_with_torch_npu_flash_attention,
    replace_with_torch_npu_rmsnorm
)

replace_with_torch_npu_flash_attention()
replace_with_torch_npu_rmsnorm()

from llava.train.train import train
import torch_npu
from torch_npu.contrib import transfer_to_npu

if __name__ == "__main__":
    train()
