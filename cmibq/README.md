# CM-IBQ: Cross-Modal Information Bottleneck Quantization for LLaVA

## 项目结构

```
cmibq/
├── cmibq/
│   ├── __init__.py
│   ├── core/                      # 核心量化模块
│   │   ├── __init__.py
│   │   ├── ib_quantizer.py       # 信息瓶颈量化器
│   │   ├── importance_estimator.py # 重要性估计
│   │   ├── bit_allocator.py      # 比特分配网络
│   │   └── differentiable_quant.py # 可微量化
│   ├── models/                    # 模型包装器
│   │   ├── __init__.py
│   │   ├── quantized_llava_wrapper.py # LLaVA量化包装
│   │   └── lora_adapter.py       # LoRA适配器
│   └── training/                  # 训练工具
│       ├── __init__.py
│       ├── distributed_trainer.py # 分布式训练器
│       └── data_utils.py          # 数据加载
├── train_cmibq.py                 # 主训练脚本
├── setup.sh                       # 环境设置脚本
├── requirements.txt               # 依赖列表
└── README.md                      # 本文件
```

## 快速开始

### 1. 环境安装

```bash
# 创建conda环境
conda create -n cmibq python=3.10 -y
conda activate cmibq

# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 安装LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
cd ..

# 安装本项目
pip install -e .
```

### 2. 准备数据

```bash
# 下载LLaVA-Instruct-150K数据集
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json -O data/llava_instruct_150k.json

# 下载COCO图像
# 方法1: 从官网下载
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/coco/

# 方法2: 使用LLaVA提供的脚本
# 参考: https://github.com/haotian-liu/LLaVA#visual-instruction-tuning
```

### 3. 运行训练

#### 方式1: 快速测试（单卡）

```bash
./train_quick_test.sh
```

#### 方式2: 完整训练流程（Stage 1 + Stage 2）

```bash
# 修改train_full_pipeline.sh中的数据路径
# 然后运行:
./train_full_pipeline.sh
```

#### 方式3: 分阶段训练

```bash
# Stage 1: Bottleneck Shaping
./train_stage1_multi_gpu.sh

# Stage 2: Alignment + LoRA (需要先完成Stage 1)
./train_stage2_multi_gpu.sh
```

## 训练参数说明

### 核心参数

- `--stage`: 训练阶段
  - `1`: Bottleneck Shaping (训练量化模块)
  - `2`: Task-Aware Optimization with Alignment (对齐损失 + LoRA)

- `--target_bits_act`: 激活量化目标比特数 (建议: 4.0)
- `--target_bits_weight`: 权重量化目标比特数 (建议: 4.0)
- `--num_groups`: 量化分组数 (建议: 8)
- `--use_ib`: 是否使用信息瓶颈框架
- `--use_lora`: 是否使用LoRA (仅Stage 2)
- `--lora_rank`: LoRA秩 (建议: 16)

### 训练策略

**Stage 1 (Bottleneck Shaping)**
- 目标: 训练量化模块，学习最优的比特分配
- 学习率: 2e-5
- Epoch: 3
- 损失: 任务损失 + IB损失 + 比特率损失

**Stage 2 (Alignment)**
- 目标: 通过对齐损失微调，同时用LoRA补偿精度
- 学习率: 1e-5 (更小)
- Epoch: 2
- 损失: 任务损失 + 对齐损失 + 少量比特率损失

### 硬件建议

| 模型大小 | 最小GPU | 推荐GPU | Batch Size | 梯度累积 |
|---------|---------|---------|-----------|----------|
| 7B      | 1×A100  | 4×A100  | 2-4       | 4-8      |
| 13B     | 2×A100  | 8×A100  | 1-2       | 8-16     |
| 34B     | 8×A100  | 16×A100 | 1         | 16-32    |

## 高级用法

### 使用DeepSpeed ZeRO

```bash
# 适合超大模型（13B+）
./train_stage1_deepspeed.sh
```

### 多机训练

```bash
# 主节点
export NODE_RANK=0
export MASTER_ADDR="主节点IP"
./train_stage1_multi_node.sh

# 从节点
export NODE_RANK=1
export MASTER_ADDR="主节点IP"
./train_stage1_multi_node.sh
```

### 使用W&B监控

```bash
# 设置API密钥
export WANDB_API_KEY="你的密钥"

# 运行
./train_with_wandb.sh
```

## 监控指标

### Stage 1关键指标
- `train/loss`: 总训练损失
- `quantization/vision_avg_bits`: Vision模块平均比特数
- `quantization/projector_avg_bits`: 投影器平均比特数
- `train/grad_norm`: 梯度范数

### Stage 2关键指标
- `train/alignment_loss`: 对齐损失
- `alignment/acc_v2t`: 视觉→文本检索准确率
- `alignment/acc_t2v`: 文本→视觉检索准确率
- `alignment/temperature`: 对比学习温度

## 常见问题

### Q1: OOM (内存不足)

**解决方案:**
- 减小 `--batch_size`
- 增大 `--gradient_accumulation_steps`
- 启用 `--gradient_checkpointing`
- 使用DeepSpeed ZeRO-3

### Q2: 训练不稳定

**解决方案:**
- 检查学习率是否过大
- 启用 `--max_grad_norm` 进行梯度裁剪
- 确保Stage 1充分训练后再进行Stage 2

### Q3: 对齐损失不下降

**解决方案:**
- 检查batch size是否足够大 (对比学习需要足够负样本)
- 调整对齐损失权重 (代码中默认0.3)
- 确保视觉和文本特征维度匹配

## 引用

如果这个项目对你有帮助，请引用:

```bibtex
@article{cmibq2024,
  title={Cross-Modal Information Bottleneck Quantization for Large Vision-Language Models},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 许可证

MIT License

---

## requirements.txt

```txt
# PyTorch生态
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.36.0
accelerate>=0.25.0
tokenizers>=0.15.0

# 训练加速
deepspeed>=0.12.0
xformers>=0.0.23

# 数据处理
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0

# LLaVA依赖
einops>=0.7.0
einops-exts>=0.0.4
timm>=0.9.0

# 日志和监控
wandb>=0.16.0
tensorboard>=2.15.0
tqdm>=4.66.0

# 工具
pydantic>=2.5.0
pyyaml>=6.0
```

---

## setup.sh (一键安装脚本)

```bash
#!/bin/bash

echo "=========================================="
echo "CM-IBQ Environment Setup"
echo "=========================================="

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# 创建环境
echo "Creating conda environment..."
conda create -n cmibq python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate cmibq

# 检测CUDA版本
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "Warning: nvidia-smi not found, installing CPU version"
    CUDA_VERSION="cpu"
fi

# 安装PyTorch
echo "Installing PyTorch..."
if [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.1.0 torchvision==0.16.0
fi

# 安装其他依赖
echo "Installing dependencies..."
pip install -r requirements.txt

# 安装LLaVA
echo "Installing LLaVA..."
if [ ! -d "LLaVA" ]; then
    git clone https://github.com/haotian-liu/LLaVA.git
fi
cd LLaVA && pip install -e . && cd ..

# 安装本项目
echo "Installing CM-IBQ..."
pip install -e .

# 创建必要目录
mkdir -p data/coco checkpoints logs

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "Activate environment: conda activate cmibq"
echo "=========================================="
```
