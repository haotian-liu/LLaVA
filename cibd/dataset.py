import json
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import numpy as np
from typing import Dict, List, Optional


class LLaVADataset(Dataset):
    """
    LLaVA-150K数据集类
    用于CIBD蒸馏训练
    
    关键修复：
    1. 智能图片加载（自动尝试 train2017/val2017）
    2. 图片加载失败时自动跳过样本
    3. 强制 RGB 模式，防止 RGBA/4通道问题
    """
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        max_length: int = 2048,
        is_training: bool = True,
        use_augmentation: bool = False
    ):
        self.data_path = data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.is_training = is_training
        self.use_augmentation = use_augmentation
        
        # 加载数据
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples")
        
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
            
    def __len__(self):
        return len(self.data)
    
    def load_image_smart(self, image_path: str) -> Optional[Image.Image]:
        """
        智能加载图像，强制 RGB 模式
        """
        if not image_path:
            return None
        
        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(self.image_folder, image_path),
            os.path.join(self.image_folder, 'train2017', os.path.basename(image_path)),
            os.path.join(self.image_folder, 'val2017', os.path.basename(image_path)),
            os.path.join(self.image_folder, 'coco', 'train2017', os.path.basename(image_path)),
            os.path.join(self.image_folder, 'coco', 'val2017', os.path.basename(image_path)),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # ===== 关键：强制 RGB 转换，防止 RGBA =====
                    im = Image.open(path)
                    
                    # 如果不是 RGB，先转 RGBA 再转 RGB（处理各种模式）
                    if im.mode != 'RGB':
                        if im.mode in ['RGBA', 'LA', 'P']:
                            im = im.convert('RGBA').convert('RGB')
                        else:
                            im = im.convert('RGB')
                    # ===== 结束 =====
                    
                    # 数据增强
                    if self.is_training and self.use_augmentation:
                        if random.random() > 0.5:
                            im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    return im
                except Exception as e:
                    continue
        
        return None
    
    def process_conversation(self, conversations: List[Dict]) -> Dict[str, torch.Tensor]:
        """处理对话数据"""
        full_text = ""
        labels_list = []
        
        for conv in conversations:
            role = conv['from']
            value = conv['value']
            
            if role == 'human':
                if '<image>' in value:
                    value = value.replace('<image>', ' <image> ')
                prompt = f"USER: {value} ASSISTANT: "
                full_text += prompt
                labels_list.extend([-100] * len(self.tokenizer.encode(prompt, add_special_tokens=False)))
                
            elif role == 'gpt':
                response = f"{value}</s>"
                full_text += response
                response_ids = self.tokenizer.encode(response, add_special_tokens=False)
                labels_list.extend(response_ids)
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        labels = torch.full_like(input_ids, -100)
        
        if len(labels_list) <= self.max_length:
            labels[:len(labels_list)] = torch.tensor(labels_list)
        else:
            labels = torch.tensor(labels_list[:self.max_length])
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                item = self.data[idx]
                
                # 处理图像
                image_path = item.get('image', None)
                if image_path:
                    image = self.load_image_smart(image_path)
                    
                    if image is None:
                        if attempt == 0:
                            print(f"Warning: Could not load image {image_path}, trying next sample...")
                        idx = (idx + 1) % len(self.data)
                        continue
                    
                    # 使用 image_processor 处理
                    processed = self.image_processor(image, return_tensors="pt")
                    image_tensor = processed['pixel_values'][0]
                    
                    # ===== 双重保险：再次检查通道数 =====
                    if image_tensor.shape[0] == 4:
                        print(f"Warning: Got 4 channels after processing, trimming to 3")
                        image_tensor = image_tensor[:3, :, :]
                    # ===== 结束 =====
                else:
                    # 无图样本
                    image_tensor = None
                
                # 处理对话
                conversation_data = self.process_conversation(item['conversations'])
                
                return {
                    'input_ids': conversation_data['input_ids'],
                    'attention_mask': conversation_data['attention_mask'],
                    'labels': conversation_data['labels'],
                    'images': image_tensor,  # None 或 [3, H, W]
                    'id': item.get('id', f'sample_{idx}')
                }
                
            except Exception as e:
                print(f"Error processing sample {idx} (attempt {attempt+1}/{max_attempts}): {e}")
                idx = (idx + 1) % len(self.data)
                continue
        
        raise RuntimeError(f"Failed to load valid sample after {max_attempts} attempts")


class LLaVACollator:
    """
    数据整理器 - 修复对齐问题
    
    关键修复：
    1. 保持图像与样本的一一对应（使用 list）
    2. 不丢弃无图样本的图像位
    3. 兜底裁剪 alpha 通道
    """
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __call__(self, batch: List[Dict]) -> Dict:
        """整理批次数据"""
        # 收集文本字段
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        # ===== 关键修复：保持图像与样本对齐 =====
        images_list = []
        for item in batch:
            img = item['images']  # None 或 [3, H, W]
            
            if img is not None:
                # 兜底：再次检查并裁剪 alpha 通道
                if torch.is_tensor(img) and img.dim() == 3 and img.size(0) == 4:
                    img = img[:3]
                
                # 验证形状
                if img.shape[0] != 3:
                    print(f"Warning: Image has {img.shape[0]} channels, expected 3")
                    img = None  # 标记为无效
            
            images_list.append(img)
        # ===== 修复结束 =====
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': images_list,  # ← 返回 list，保持对齐
            'ids': [item['id'] for item in batch]
        }


class LLaVAPretrainingDataset(Dataset):
    """用于第一阶段预训练的数据集（可选）"""
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        max_length: int = 512
    ):
        self.data_path = data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def load_image_smart(self, image_path: str) -> Optional[Image.Image]:
        """智能加载图像"""
        if not image_path:
            return None
        
        possible_paths = [
            os.path.join(self.image_folder, image_path),
            os.path.join(self.image_folder, 'train2017', os.path.basename(image_path)),
            os.path.join(self.image_folder, 'val2017', os.path.basename(image_path)),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    im = Image.open(path)
                    if im.mode != 'RGB':
                        if im.mode in ['RGBA', 'LA', 'P']:
                            im = im.convert('RGBA').convert('RGB')
                        else:
                            im = im.convert('RGB')
                    return im
                except:
                    continue
        return None
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        image = self.load_image_smart(item.get('image', ''))
        if image is None:
            return self.__getitem__((idx + 1) % len(self.data))
        
        processed = self.image_processor(image, return_tensors="pt")
        image_tensor = processed['pixel_values'][0]
        
        # 兜底裁剪
        if image_tensor.shape[0] == 4:
            image_tensor = image_tensor[:3]
        
        caption = item.get('caption', '')
        text = f"USER: <image>\nWhat is in this image? ASSISTANT: {caption}</s>"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = encoding.input_ids[0].clone()
        assistant_idx = (labels == self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)[0]).nonzero()
        if len(assistant_idx) > 0:
            labels[:assistant_idx[0][0]+2] = -100
            
        return {
            'input_ids': encoding.input_ids[0],
            'attention_mask': encoding.attention_mask[0],
            'labels': labels,
            'images': image_tensor
        }