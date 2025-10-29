"""
数据加载和预处理工具 - 改进版
修复了与LLaVA集成的关键问题
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import copy
from transformers import PreTrainedTokenizer
import re


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


@dataclass
class LLaVADataCollator:
    """
    LLaVA数据整理器 - 改进版
    """
    tokenizer: any
    image_processor: any
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理batch数据
        """
        # 提取各个字段
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        
        # Padding - 使用tokenizer的pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100  # ignore_index for CrossEntropyLoss
        )
        
        # Attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        
        # 处理图像 - 关键修复
        if any('image' in instance for instance in instances):
            images = []
            for instance in instances:
                if 'image' in instance and instance['image'] is not None:
                    images.append(instance['image'])
            
            if images:
                # 使用LLaVA的处理方式
                if hasattr(self.image_processor, 'preprocess'):
                    # 这是LLaVA的标准处理方式
                    image_tensor = self.image_processor.preprocess(
                        images, 
                        return_tensors='pt'
                    )['pixel_values']
                else:
                    # 备用处理方式
                    from torchvision import transforms
                    preprocess = transforms.Compose([
                        transforms.Resize((336, 336)),  # LLaVA v1.5默认尺寸
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
                    ])
                    image_tensor = torch.stack([preprocess(img) for img in images])
                
                batch['images'] = image_tensor
        
        return batch


class LLaVAInstructDataset(Dataset):
    """
    LLaVA Instruct数据集 - 改进版
    完全兼容LLaVA的对话格式
    """
    
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        max_length: int = 2048,
        is_training: bool = True,
        use_mm_proj: bool = True,  # 是否使用多模态投影
        image_aspect_ratio: str = 'pad'  # 'pad' or 'square'
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.is_training = is_training
        self.image_folder = image_folder
        self.use_mm_proj = use_mm_proj
        self.image_aspect_ratio = image_aspect_ratio
        
        # 加载数据
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # 处理不同格式的数据
        if isinstance(data, dict):
            # 可能是 {'data': [...]} 格式
            if 'data' in data:
                self.data = data['data']
            else:
                self.data = [data]
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        print(f"Loaded {len(self.data)} examples")
        
        # 设置特殊token
        self._setup_special_tokens()
    
    def _setup_special_tokens(self):
        """设置LLaVA特殊token"""
        # 添加特殊token到tokenizer（如果还没有）
        special_tokens_dict = {}
        
        if DEFAULT_IMAGE_TOKEN not in self.tokenizer.get_vocab():
            special_tokens_dict['additional_special_tokens'] = [DEFAULT_IMAGE_TOKEN]
        
        if DEFAULT_IM_START_TOKEN not in self.tokenizer.get_vocab():
            if 'additional_special_tokens' not in special_tokens_dict:
                special_tokens_dict['additional_special_tokens'] = []
            special_tokens_dict['additional_special_tokens'].extend([
                DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            ])
        
        if special_tokens_dict:
            self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # 设置图像token ID
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    
    def __len__(self):
        return len(self.data)
    
    def _tokenize_conversations(self, conversations: List[Dict], has_image: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        转换对话为token - LLaVA格式
        """
        # LLaVA v1.5使用的对话模板
        if self.tokenizer.name_or_path and 'vicuna' in self.tokenizer.name_or_path.lower():
            # Vicuna格式
            system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            roles = ("USER", "ASSISTANT")
            sep = " "
            sep2 = "</s>"
        else:
            # 默认格式
            system_message = ""
            roles = ("USER", "ASSISTANT") 
            sep = "\n"
            sep2 = "</s>"
        
        # 构建完整对话
        messages = []
        if system_message:
            messages.append(system_message)
        
        for i, conv in enumerate(conversations):
            role = conv.get('from', 'human')
            content = conv.get('value', '')
            
            # 处理第一个消息中的图像token
            if i == 0 and has_image and DEFAULT_IMAGE_TOKEN not in content:
                # 在第一个用户消息前添加图像token
                content = DEFAULT_IMAGE_TOKEN + '\n' + content
            
            if role == 'human':
                messages.append(f"{roles[0]}: {content}")
            elif role == 'gpt' or role == 'assistant':
                messages.append(f"{roles[1]}: {content}{sep2}")
        
        # 合并所有消息
        full_prompt = sep.join(messages)
        
        # Tokenize
        encoding = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'][0]
        
        # 创建labels
        if self.is_training:
            labels = input_ids.clone()
            
            # Mask人类输入部分（只计算助手回复的loss）
            for i, conv in enumerate(conversations):
                if conv.get('from') == 'human':
                    # 找到这部分在input_ids中的位置并mask
                    # 这里简化处理，实际可能需要更精确的定位
                    pass  # TODO: 实现精确的masking
        else:
            labels = input_ids.clone()
        
        return input_ids, labels
    
    def __getitem__(self, idx):
        """
        获取单个样本
        """
        item = self.data[idx]
        
        # 获取对话
        conversations = item.get('conversations', [])
        if not conversations:
            # 如果没有conversations字段，尝试其他格式
            if 'question' in item and 'answer' in item:
                conversations = [
                    {'from': 'human', 'value': item['question']},
                    {'from': 'gpt', 'value': item['answer']}
                ]
        
        # 检查是否有图像
        image_file = item.get('image', None)
        has_image = image_file is not None
        
        # Tokenize对话
        input_ids, labels = self._tokenize_conversations(conversations, has_image)
        
        result = {
            'input_ids': input_ids,
            'labels': labels,
            'has_image': has_image
        }
        
        # 加载图像
        if has_image:
            image_path = os.path.join(self.image_folder, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
                
                # 根据设置调整图像比例
                if self.image_aspect_ratio == 'square':
                    # 裁剪为正方形
                    w, h = image.size
                    min_dim = min(w, h)
                    left = (w - min_dim) // 2
                    top = (h - min_dim) // 2
                    image = image.crop((left, top, left + min_dim, top + min_dim))
                
                result['image'] = image
                
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                # 创建空白图像作为替代
                result['image'] = Image.new('RGB', (336, 336), color='gray')
        else:
            # 纯文本对话，可能需要占位图像
            if self.use_mm_proj:
                result['image'] = None
        
        return result


def make_supervised_data_module(
    tokenizer,
    image_processor, 
    data_args: Dict
) -> Dict:
    """
    创建训练和验证数据集
    
    Args:
        tokenizer: 分词器
        image_processor: 图像处理器
        data_args: 数据参数
    """
    # 训练数据集
    train_dataset = None
    if data_args.get('train_data_path'):
        train_dataset = LLaVAInstructDataset(
            data_path=data_args['train_data_path'],
            image_folder=data_args['image_folder'],
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=data_args.get('max_length', 2048),
            is_training=True,
            use_mm_proj=data_args.get('use_mm_proj', True),
            image_aspect_ratio=data_args.get('image_aspect_ratio', 'pad')
        )
    
    # 验证数据集
    eval_dataset = None
    if data_args.get('eval_data_path'):
        eval_dataset = LLaVAInstructDataset(
            data_path=data_args['eval_data_path'],
            image_folder=data_args['image_folder'],
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=data_args.get('max_length', 2048),
            is_training=False,
            use_mm_proj=data_args.get('use_mm_proj', True),
            image_aspect_ratio=data_args.get('image_aspect_ratio', 'pad')
        )
    
    # 数据整理器
    data_collator = LLaVADataCollator(
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    is_train: bool = True,
    collate_fn=None,
    **kwargs
) -> DataLoader:
    """
    创建数据加载器的辅助函数
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_train,
        **kwargs
    )
