#!/usr/bin/env python
"""
LLaVA数据预处理工具
将各种常见数据集格式转换为LLaVA训练所需的格式
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import random
from tqdm import tqdm
from PIL import Image
import argparse


class LLaVADataPreprocessor:
    """
    将各种数据集转换为LLaVA格式的预处理器
    
    LLaVA格式要求：
    [
        {
            "id": "unique_sample_id",
            "image": "image_filename.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion here?"},
                {"from": "gpt", "value": "Answer here."}
            ]
        }
    ]
    """
    
    def __init__(self, output_dir: str = "./processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_coco_captions(
        self, 
        captions_file: str, 
        images_dir: str,
        output_file: str = "coco_captions_llava.json",
        max_samples: Optional[int] = None
    ) -> str:
        """
        将COCO Captions数据集转换为LLaVA格式
        
        Args:
            captions_file: COCO captions JSON文件路径
            images_dir: COCO图像目录
            output_file: 输出文件名
            max_samples: 最大样本数（用于测试）
        """
        print(f"Converting COCO Captions from {captions_file}...")
        
        with open(captions_file, 'r') as f:
            coco_data = json.load(f)
        
        # 构建image_id到文件名的映射
        image_map = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # 构建image_id到captions的映射
        caption_map = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in caption_map:
                caption_map[img_id] = []
            caption_map[img_id].append(ann['caption'])
        
        # 转换为LLaVA格式
        llava_data = []
        for img_id, captions in tqdm(list(caption_map.items())[:max_samples], desc="Processing COCO"):
            if img_id not in image_map:
                continue
                
            # 为每个caption创建一个对话
            for idx, caption in enumerate(captions[:3]):  # 最多使用3个captions
                conversation = {
                    "id": f"coco_{img_id}_{idx}",
                    "image": image_map[img_id],
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nDescribe this image in detail."
                        },
                        {
                            "from": "gpt",
                            "value": caption
                        }
                    ]
                }
                llava_data.append(conversation)
        
        # 保存结果
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(llava_data, f, indent=2)
        
        print(f"✓ Converted {len(llava_data)} COCO samples to {output_path}")
        return str(output_path)
    
    def convert_vqav2(
        self,
        questions_file: str,
        annotations_file: str,
        images_dir: str,
        output_file: str = "vqav2_llava.json",
        max_samples: Optional[int] = None
    ) -> str:
        """
        将VQAv2数据集转换为LLaVA格式
        
        Args:
            questions_file: VQAv2 questions JSON文件
            annotations_file: VQAv2 annotations JSON文件
            images_dir: 图像目录
            output_file: 输出文件名
        """
        print(f"Converting VQAv2 dataset...")
        
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        # 构建question_id到annotation的映射
        ann_map = {ann['question_id']: ann for ann in annotations_data['annotations']}
        
        llava_data = []
        for q in tqdm(questions_data['questions'][:max_samples], desc="Processing VQAv2"):
            if q['question_id'] not in ann_map:
                continue
            
            ann = ann_map[q['question_id']]
            
            # 构造图像文件名（VQAv2的命名规则）
            # COCO_split_imageID.jpg
            image_id = str(q['image_id']).zfill(12)
            split = 'train2014' if 'train' in questions_file else 'val2014'
            image_file = f"COCO_{split}_{image_id}.jpg"
            
            # 选择最常见的答案
            best_answer = ann['multiple_choice_answer']
            
            conversation = {
                "id": f"vqa_{q['question_id']}",
                "image": image_file,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{q['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": best_answer
                    }
                ]
            }
            llava_data.append(conversation)
        
        # 保存结果
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(llava_data, f, indent=2)
        
        print(f"✓ Converted {len(llava_data)} VQAv2 samples to {output_path}")
        return str(output_path)
    
    def convert_custom_qa(
        self,
        qa_data: List[Dict],
        output_file: str = "custom_qa_llava.json"
    ) -> str:
        """
        将自定义问答数据转换为LLaVA格式
        
        Args:
            qa_data: 自定义QA数据，格式为:
                [{"image": "xxx.jpg", "question": "...", "answer": "..."}, ...]
            output_file: 输出文件名
        """
        llava_data = []
        
        for idx, item in enumerate(qa_data):
            conversation = {
                "id": f"custom_{idx}",
                "image": item['image'],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{item['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": item['answer']
                    }
                ]
            }
            
            # 如果有多轮对话
            if 'follow_up' in item:
                for follow_up in item['follow_up']:
                    conversation['conversations'].extend([
                        {
                            "from": "human",
                            "value": follow_up['question']
                        },
                        {
                            "from": "gpt",
                            "value": follow_up['answer']
                        }
                    ])
            
            llava_data.append(conversation)
        
        # 保存结果
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(llava_data, f, indent=2)
        
        print(f"✓ Converted {len(llava_data)} custom samples to {output_path}")
        return str(output_path)
    
    def convert_instruction_tuning(
        self,
        instructions: List[str],
        image_files: List[str],
        output_file: str = "instruction_llava.json"
    ) -> str:
        """
        创建指令微调数据
        
        Args:
            instructions: 指令模板列表
            image_files: 图像文件列表
        """
        llava_data = []
        
        # 预定义的指令模板
        default_instructions = [
            "What is shown in this image?",
            "Describe this image in detail.",
            "What are the main objects in this image?",
            "Explain what's happening in this picture.",
            "What can you tell me about this image?",
            "Analyze the content of this image.",
            "Provide a comprehensive description of this image.",
            "What is the most interesting aspect of this image?"
        ]
        
        if not instructions:
            instructions = default_instructions
        
        for idx, img_file in enumerate(image_files):
            # 随机选择一个指令
            instruction = random.choice(instructions)
            
            conversation = {
                "id": f"instruct_{idx}",
                "image": img_file,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{instruction}"
                    },
                    {
                        "from": "gpt",
                        "value": "I'll need to analyze the actual image to provide a description. This is a placeholder response."
                    }
                ]
            }
            llava_data.append(conversation)
        
        # 保存结果
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(llava_data, f, indent=2)
        
        print(f"✓ Created {len(llava_data)} instruction samples to {output_path}")
        return str(output_path)
    
    def merge_datasets(
        self,
        json_files: List[str],
        output_file: str = "merged_llava.json",
        shuffle: bool = True
    ) -> str:
        """
        合并多个LLaVA格式的JSON文件
        
        Args:
            json_files: 要合并的JSON文件列表
            output_file: 输出文件名
            shuffle: 是否打乱数据
        """
        merged_data = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                merged_data.extend(data)
                print(f"  Added {len(data)} samples from {json_file}")
        
        if shuffle:
            random.shuffle(merged_data)
        
        # 重新分配ID以避免重复
        for idx, item in enumerate(merged_data):
            item['id'] = f"merged_{idx}"
        
        # 保存结果
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"✓ Merged {len(merged_data)} total samples to {output_path}")
        return str(output_path)
    
    def validate_dataset(self, json_file: str, images_dir: str) -> Dict:
        """
        验证LLaVA格式数据集的完整性
        
        Args:
            json_file: LLaVA格式的JSON文件
            images_dir: 图像目录
        
        Returns:
            验证报告
        """
        print(f"Validating dataset: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        report = {
            'total_samples': len(data),
            'missing_images': [],
            'invalid_conversations': [],
            'missing_image_token': [],
            'statistics': {
                'avg_turns': 0,
                'min_turns': float('inf'),
                'max_turns': 0
            }
        }
        
        total_turns = 0
        images_dir = Path(images_dir)
        
        for item in tqdm(data, desc="Validating"):
            # 检查图像文件是否存在
            img_path = images_dir / item['image']
            if not img_path.exists():
                report['missing_images'].append(item['image'])
            
            # 检查对话格式
            if 'conversations' not in item or len(item['conversations']) == 0:
                report['invalid_conversations'].append(item['id'])
                continue
            
            # 检查是否包含<image>标记
            first_human = item['conversations'][0]
            if first_human['from'] == 'human' and '<image>' not in first_human['value']:
                report['missing_image_token'].append(item['id'])
            
            # 统计对话轮数
            num_turns = len(item['conversations']) // 2
            total_turns += num_turns
            report['statistics']['min_turns'] = min(report['statistics']['min_turns'], num_turns)
            report['statistics']['max_turns'] = max(report['statistics']['max_turns'], num_turns)
        
        report['statistics']['avg_turns'] = total_turns / len(data) if data else 0
        
        # 打印报告
        print("\n" + "="*50)
        print("Validation Report:")
        print("="*50)
        print(f"Total samples: {report['total_samples']}")
        print(f"Missing images: {len(report['missing_images'])}")
        print(f"Invalid conversations: {len(report['invalid_conversations'])}")
        print(f"Missing <image> token: {len(report['missing_image_token'])}")
        print(f"Average conversation turns: {report['statistics']['avg_turns']:.2f}")
        print(f"Min/Max turns: {report['statistics']['min_turns']}/{report['statistics']['max_turns']}")
        
        if report['missing_images']:
            print(f"\nFirst 5 missing images: {report['missing_images'][:5]}")
        
        return report
    
    def create_train_val_split(
        self,
        json_file: str,
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> tuple:
        """
        将数据集分割为训练集和验证集
        
        Args:
            json_file: 输入的JSON文件
            train_ratio: 训练集比例
            seed: 随机种子
        
        Returns:
            (train_file, val_file) 路径元组
        """
        random.seed(seed)
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        random.shuffle(data)
        
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # 保存训练集
        train_file = json_file.replace('.json', '_train.json')
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # 保存验证集
        val_file = json_file.replace('.json', '_val.json')
        with open(val_file, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"✓ Split dataset: {len(train_data)} train, {len(val_data)} val")
        print(f"  Train: {train_file}")
        print(f"  Val: {val_file}")
        
        return train_file, val_file


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(description='LLaVA Data Preprocessor')
    parser.add_argument('--task', type=str, required=True,
                       choices=['coco', 'vqa', 'merge', 'validate', 'split'],
                       help='预处理任务类型')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                       help='输入文件路径')
    parser.add_argument('--images_dir', type=str,
                       help='图像目录路径')
    parser.add_argument('--output', type=str, default='./processed_data',
                       help='输出目录')
    parser.add_argument('--max_samples', type=int,
                       help='最大样本数（用于测试）')
    
    args = parser.parse_args()
    
    preprocessor = LLaVADataPreprocessor(output_dir=args.output)
    
    if args.task == 'coco':
        if len(args.input) != 1:
            raise ValueError("COCO task requires 1 input file (captions)")
        preprocessor.convert_coco_captions(
            args.input[0], 
            args.images_dir,
            max_samples=args.max_samples
        )
    
    elif args.task == 'vqa':
        if len(args.input) != 2:
            raise ValueError("VQA task requires 2 input files (questions, annotations)")
        preprocessor.convert_vqav2(
            args.input[0],
            args.input[1],
            args.images_dir,
            max_samples=args.max_samples
        )
    
    elif args.task == 'merge':
        preprocessor.merge_datasets(args.input)
    
    elif args.task == 'validate':
        if len(args.input) != 1:
            raise ValueError("Validate task requires 1 input file")
        preprocessor.validate_dataset(args.input[0], args.images_dir)
    
    elif args.task == 'split':
        if len(args.input) != 1:
            raise ValueError("Split task requires 1 input file")
        preprocessor.create_train_val_split(args.input[0])


if __name__ == '__main__':
    main()
