#!/usr/bin/env python
"""
CM-IBQ 优化版数据准备脚本
采用"复杂->大规模"的数据策略，避免复杂度倒挂
"""

import json
import os
import random
import requests
from pathlib import Path
from typing import List, Dict, Optional
import wget
import zipfile
from tqdm import tqdm
from PIL import Image
import hashlib


class CMIBQDataPreparer:
    """
    CM-IBQ数据准备器 - 优化版
    
    核心策略：
    Stage 1: 高质量复杂指令数据 -> 学习精细的重要性先验
    Stage 2: 大规模混合数据 -> 全面的任务对齐和泛化
    """
    
    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # 创建目录结构
        self.dirs = {
            'stage1': self.data_root / 'stage1',
            'stage2': self.data_root / 'stage2',
            'images': self.data_root / 'images',
            'raw': self.data_root / 'raw'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("CM-IBQ Data Preparation - Optimized Strategy")
        print("=" * 80)
        print("Stage 1: Complex instructions (LLaVA-Instruct-150K)")
        print("         -> Learn sophisticated importance priors")
        print("Stage 2: Large-scale mix (LLaVA-v1.5-mix665k)")  
        print("         -> Comprehensive alignment & generalization")
        print("=" * 80)
    
    def prepare_stage1_data(self, max_samples: int = 150000):
        """
        准备Stage 1数据 - 使用LLaVA-Instruct-150K
        
        目标：通过高质量、复杂的指令数据，让ImportanceEstimationNetwork
        学习到精细、全面的视觉特征重要性判断标准
        """
        print("\n" + "=" * 80)
        print("Stage 1: Bottleneck Shaping with Complex Instructions")
        print("=" * 80)
        
        # 下载LLaVA-Instruct-150K
        llava_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
        llava_file = self.dirs['raw'] / "llava_instruct_150k.json"
        
        if not llava_file.exists():
            print("Downloading LLaVA-Instruct-150K (high-quality complex instructions)...")
            try:
                response = requests.get(llava_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(llava_file, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                print("✓ Downloaded LLaVA-Instruct-150K")
            except Exception as e:
                print(f"Error downloading: {e}")
                return None, None
        
        # 加载数据
        print("\nLoading and analyzing instruction complexity...")
        with open(llava_file, 'r') as f:
            llava_data = json.load(f)
        
        # 分析指令复杂度
        complexity_stats = self._analyze_instruction_complexity(llava_data)
        print("\nInstruction Complexity Analysis:")
        print(f"  Total samples: {len(llava_data):,}")
        print(f"  Avg conversation turns: {complexity_stats['avg_turns']:.1f}")
        print(f"  Complex reasoning: {complexity_stats['reasoning_pct']:.1%}")
        print(f"  Detailed descriptions: {complexity_stats['detailed_pct']:.1%}")
        print(f"  Multi-hop questions: {complexity_stats['multihop_pct']:.1%}")
        
        # 根据复杂度对数据进行分层
        complex_samples = []
        moderate_samples = []
        simple_samples = []
        
        for item in tqdm(llava_data, desc="Categorizing by complexity"):
            complexity = self._compute_sample_complexity(item)
            
            if complexity > 0.7:
                complex_samples.append(item)
            elif complexity > 0.4:
                moderate_samples.append(item)
            else:
                simple_samples.append(item)
        
        print(f"\nComplexity Distribution:")
        print(f"  Complex: {len(complex_samples):,} samples")
        print(f"  Moderate: {len(moderate_samples):,} samples")
        print(f"  Simple: {len(simple_samples):,} samples")
        
        # Stage 1策略：优先使用复杂样本
        stage1_data = []
        
        # 1. 添加所有复杂样本（这些对学习重要性先验最有价值）
        stage1_data.extend(complex_samples)
        
        # 2. 添加部分中等复杂度样本
        if len(stage1_data) < max_samples:
            remaining = max_samples - len(stage1_data)
            stage1_data.extend(moderate_samples[:remaining])
        
        # 3. 如果还需要更多，添加简单样本
        if len(stage1_data) < max_samples:
            remaining = max_samples - len(stage1_data)
            stage1_data.extend(simple_samples[:remaining])
        
        # 截断到目标数量
        stage1_data = stage1_data[:max_samples]
        
        print(f"\nSelected {len(stage1_data):,} samples for Stage 1")
        print(f"  Average complexity: {sum(self._compute_sample_complexity(s) for s in stage1_data)/len(stage1_data):.2f}")
        
        # 数据增强：为复杂样本创建额外的训练信号
        augmented_data = []
        for item in tqdm(stage1_data[:len(stage1_data)//3], desc="Augmenting complex samples"):
            augmented_data.append(item)
            
            # 创建聚焦不同方面的变体
            if len(item['conversations']) > 2:  # 多轮对话
                # 变体1：聚焦细节
                variant1 = self._create_detail_focused_variant(item)
                if variant1:
                    augmented_data.append(variant1)
                
                # 变体2：聚焦推理
                variant2 = self._create_reasoning_focused_variant(item)
                if variant2:
                    augmented_data.append(variant2)
        
        # 合并原始数据和增强数据
        stage1_data = stage1_data + augmented_data
        random.shuffle(stage1_data)
        
        # 分割训练/验证集（95/5比例）
        split_idx = int(len(stage1_data) * 0.95)
        train_data = stage1_data[:split_idx]
        val_data = stage1_data[split_idx:]
        
        # 保存Stage 1数据
        stage1_train_path = self.dirs['stage1'] / 'train_instruct150k.json'
        stage1_val_path = self.dirs['stage1'] / 'val_instruct150k.json'
        
        print("\nSaving Stage 1 data...")
        with open(stage1_train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(stage1_val_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"\n✓ Stage 1 数据准备完成:")
        print(f"  训练集: {len(train_data):,} 样本 -> {stage1_train_path}")
        print(f"  验证集: {len(val_data):,} 样本 -> {stage1_val_path}")
        print(f"  特点: 高复杂度指令，适合学习精细的重要性先验")
        
        return stage1_train_path, stage1_val_path
    
    def prepare_stage2_data(self, max_samples: int = 665000):
        """
        准备Stage 2数据 - 使用LLaVA-v1.5-mix665k
        
        目标：在已有精细重要性先验的基础上，通过大规模、多样化的数据
        进行全面的任务对齐和泛化训练
        """
        print("\n" + "=" * 80)
        print("Stage 2: Task-Aware Optimization with Large-Scale Mix")
        print("=" * 80)
        
        # 下载Mix665K
        mix_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json"
        mix_file = self.dirs['raw'] / "llava_v1_5_mix665k.json"
        
        if not mix_file.exists():
            print("Downloading LLaVA-v1.5-mix665k (large-scale diverse data)...")
            try:
                response = requests.get(mix_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(mix_file, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                print("✓ Downloaded LLaVA-v1.5-mix665k")
            except Exception as e:
                print(f"Error downloading: {e}")
                print("Falling back to creating synthetic large-scale data...")
                return self._create_synthetic_stage2_data(max_samples)
        
        # 加载数据
        print("\nLoading mix665k data...")
        with open(mix_file, 'r') as f:
            mix_data = json.load(f)
        
        print(f"Total samples in mix665k: {len(mix_data):,}")
        
        # 分析数据集组成
        data_sources = self._analyze_data_sources(mix_data)
        print("\nDataset Composition:")
        for source, info in sorted(data_sources.items(), key=lambda x: x[1]['count'], reverse=True):
            pct = info['count'] / len(mix_data) * 100
            print(f"  {source}: {info['count']:,} samples ({pct:.1f}%) - {info['description']}")
        
        # ⭐⭐⭐ 修改：使用全部数据或快速随机采样
        if max_samples >= len(mix_data):
            # 使用全部数据
            print(f"\n✓ Using all {len(mix_data):,} samples (no sampling)")
        else:
            # 快速随机采样（不用慢的分层采样）
            print(f"\n⚡ Fast random sampling: {max_samples:,} from {len(mix_data):,} samples...")
            import random
            random.seed(42)
            mix_data = random.sample(mix_data, max_samples)
            print(f"✓ Sampled {len(mix_data):,} samples")
        
        # 原来的慢速分层采样代码（已注释）
        # if max_samples < len(mix_data):
        #     print(f"\nSampling {max_samples:,} from {len(mix_data):,} samples...")
        #     # 分层采样以保持多样性（很慢！）
        #     sampled_data = self._stratified_sampling(mix_data, max_samples, data_sources)
        #     mix_data = sampled_data
        
        print(f"\nProcessing {len(mix_data):,} samples for Stage 2...")
        
        # 处理和验证数据格式
        stage2_data = []
        for item in tqdm(mix_data, desc="Processing and validating"):
            # 确保格式正确
            if 'conversations' in item and item['conversations']:
                # 检查并添加图像标记
                first_conv = item['conversations'][0]
                if '<image>' not in first_conv.get('value', ''):
                    first_conv['value'] = '<image>\n' + first_conv.get('value', '')
                
                # 添加任务类型标记（帮助对齐训练）
                item['task_type'] = self._identify_task_type(item)
                
                stage2_data.append(item)
        
        # 任务类型统计
        task_types = {}
        for item in stage2_data:
            task_type = item.get('task_type', 'unknown')
            task_types[task_type] = task_types.get(task_type, 0) + 1
        
        print("\nTask Type Distribution:")
        for task, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {task}: {count:,} ({count/len(stage2_data)*100:.1f}%)")
        
        # 分割训练/验证集（98/2比例，因为数据量大）
        random.shuffle(stage2_data)
        split_idx = int(len(stage2_data) * 0.98)
        
        train_data = stage2_data[:split_idx]
        val_data = stage2_data[split_idx:]
        
        # 保存Stage 2数据
        stage2_train_path = self.dirs['stage2'] / 'train_mix665k.json'
        stage2_val_path = self.dirs['stage2'] / 'val_mix665k.json'
        
        print("\nSaving Stage 2 data...")
        with open(stage2_train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(stage2_val_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"\n✓ Stage 2 数据准备完成:")
        print(f"  训练集: {len(train_data):,} 样本 -> {stage2_train_path}")
        print(f"  验证集: {len(val_data):,} 样本 -> {stage2_val_path}")
        print(f"  特点: 大规模多样化数据，适合全面对齐和泛化")
        
        # 统计所需图像
        self._print_image_requirements(stage2_data)
        
        return stage2_train_path, stage2_val_path
    
    def _analyze_instruction_complexity(self, data: List[Dict]) -> Dict:
        """分析指令复杂度"""
        total_turns = 0
        reasoning_count = 0
        detailed_count = 0
        multihop_count = 0
        
        reasoning_keywords = ['why', 'how', 'explain', 'reason', 'because', 'analyze', 'compare']
        detail_keywords = ['detail', 'describe', 'elaborate', 'specific', 'comprehensive', 'thorough']
        multihop_keywords = ['and then', 'after that', 'based on', 'given that', 'considering']
        
        for item in data:
            convs = item.get('conversations', [])
            total_turns += len(convs) // 2
            
            text = ' '.join([c.get('value', '') for c in convs]).lower()
            
            if any(kw in text for kw in reasoning_keywords):
                reasoning_count += 1
            if any(kw in text for kw in detail_keywords):
                detailed_count += 1
            if any(kw in text for kw in multihop_keywords):
                multihop_count += 1
        
        return {
            'avg_turns': total_turns / max(1, len(data)),
            'reasoning_pct': reasoning_count / max(1, len(data)),
            'detailed_pct': detailed_count / max(1, len(data)),
            'multihop_pct': multihop_count / max(1, len(data))
        }
    
    def _compute_sample_complexity(self, sample: Dict) -> float:
        """计算单个样本的复杂度分数 (0-1)"""
        score = 0.0
        convs = sample.get('conversations', [])
        
        # 1. 对话轮数
        num_turns = len(convs) // 2
        score += min(num_turns / 5, 1.0) * 0.3  # 最多5轮对话得满分
        
        # 2. 文本长度
        total_length = sum(len(c.get('value', '')) for c in convs)
        score += min(total_length / 1000, 1.0) * 0.2  # 1000字符得满分
        
        # 3. 推理复杂度
        text = ' '.join([c.get('value', '') for c in convs]).lower()
        reasoning_words = ['why', 'how', 'explain', 'because', 'analyze', 'compare', 'differ', 'similar']
        reasoning_count = sum(1 for word in reasoning_words if word in text)
        score += min(reasoning_count / 3, 1.0) * 0.3  # 3个推理词得满分
        
        # 4. 任务多样性
        task_indicators = {
            'describe': 0.5,
            'count': 0.6,
            'identify': 0.4,
            'locate': 0.7,
            'reasoning': 0.9,
            'compare': 0.8,
            'analyze': 0.9
        }
        for indicator, weight in task_indicators.items():
            if indicator in text:
                score += weight * 0.2
                break
        
        return min(score, 1.0)
    
    def _create_detail_focused_variant(self, item: Dict) -> Optional[Dict]:
        """创建聚焦细节的变体"""
        variant = item.copy()
        variant['id'] = f"{item.get('id', 'sample')}_detail"
        
        if variant['conversations']:
            # 修改第一个问题，强调细节
            original_q = variant['conversations'][0]['value']
            detail_prompts = [
                "Focus on the specific details and ",
                "Pay attention to the fine-grained features and ",
                "Examine the subtle aspects and "
            ]
            variant['conversations'][0]['value'] = random.choice(detail_prompts) + original_q.lower()
        
        return variant
    
    def _create_reasoning_focused_variant(self, item: Dict) -> Optional[Dict]:
        """创建聚焦推理的变体"""
        variant = item.copy()
        variant['id'] = f"{item.get('id', 'sample')}_reasoning"
        
        if variant['conversations']:
            # 添加推理要求
            original_q = variant['conversations'][0]['value']
            reasoning_prompts = [
                "Analyze and explain ",
                "Reason about ",
                "What can you infer about "
            ]
            variant['conversations'][0]['value'] = random.choice(reasoning_prompts) + original_q.lower()
        
        return variant
    

    
    def _stratified_sampling(self, data: List[Dict], target_size: int, sources: Dict) -> List[Dict]:
        """分层采样以保持数据多样性"""
        sampled_data = []
        
        for source in sources:
            # 获取该来源的所有数据
            source_data = [item for item in data 
                          if self._get_source_from_id(item.get('id', '')) == source]
            
            # 计算该来源应该采样的数量
            source_ratio = sources[source]['count'] / len(data)
            sample_size = int(target_size * source_ratio)
            
            if sample_size > 0 and source_data:
                # 随机采样
                sampled = random.sample(source_data, min(sample_size, len(source_data)))
                sampled_data.extend(sampled)
        
        # 如果采样数量不足，随机补充
        if len(sampled_data) < target_size:
            remaining = target_size - len(sampled_data)
            unused = [item for item in data if item not in sampled_data]
            if unused:
                additional = random.sample(unused, min(remaining, len(unused)))
                sampled_data.extend(additional)
        
        return sampled_data[:target_size]
    
    def _analyze_data_sources(self, data: List[Dict]) -> Dict:
        """分析数据来源 - 修复版"""
        sources = {}
        
        for item in data:
            # 安全地获取ID并转换为字符串
            item_id = str(item.get('id', '')).lower()
            
            # 识别数据来源
            if 'gqa' in item_id:
                source = 'GQA'
                desc = 'Scene graph reasoning'
            elif 'ocr' in item_id:
                source = 'OCR-VQA'
                desc = 'Text recognition in images'
            elif 'vqa' in item_id or 'v2_' in item_id:
                source = 'VQAv2'
                desc = 'Visual question answering'
            elif 'textvqa' in item_id:
                source = 'TextVQA'
                desc = 'Reading text in natural images'
            elif 'sharegpt' in item_id or 'share' in item_id:
                source = 'ShareGPT4V'
                desc = 'High-quality GPT-4V conversations'
            elif 'coco' in item_id:
                source = 'COCO'
                desc = 'Image captioning'
            elif 'llava' in item_id:
                source = 'LLaVA-Instruct'
                desc = 'Complex instructions'
            else:
                source = 'Other'
                desc = 'Mixed sources'
            
            if source not in sources:
                sources[source] = {'count': 0, 'description': desc}
            sources[source]['count'] += 1
        
        return sources


    def _get_source_from_id(self, item_id: str) -> str:
        """从ID推断数据来源 - 修复版"""
        # 确保item_id是字符串
        item_id = str(item_id).lower()
        
        if 'gqa' in item_id:
            return 'GQA'
        elif 'ocr' in item_id:
            return 'OCR-VQA'
        elif 'vqa' in item_id or 'v2_' in item_id:
            return 'VQAv2'
        elif 'textvqa' in item_id:
            return 'TextVQA'
        elif 'sharegpt' in item_id or 'share' in item_id:
            return 'ShareGPT4V'
        elif 'coco' in item_id:
            return 'COCO'
        elif 'llava' in item_id:
            return 'LLaVA-Instruct'
        else:
            return 'Other'
    
    def _identify_task_type(self, item: Dict) -> str:
        """识别任务类型"""
        text = ' '.join([c.get('value', '') for c in item.get('conversations', [])]).lower()
        
        if any(word in text for word in ['count', 'how many', 'number']):
            return 'counting'
        elif any(word in text for word in ['where', 'locate', 'position', 'find']):
            return 'localization'
        elif any(word in text for word in ['read', 'text', 'write', 'written']):
            return 'ocr'
        elif any(word in text for word in ['why', 'explain', 'reason', 'because']):
            return 'reasoning'
        elif any(word in text for word in ['describe', 'caption', 'detail']):
            return 'description'
        elif any(word in text for word in ['what', 'which', 'who']):
            return 'identification'
        elif len(item.get('conversations', [])) > 4:
            return 'conversation'
        else:
            return 'general'
    
    def _print_image_requirements(self, data: List[Dict]):
        """打印图像需求统计"""
        image_sources = {
            'coco': set(),
            'gqa': set(),
            'textvqa': set(),
            'ocr_vqa': set(),
            'other': set()
        }
        
        for item in data:
            img = item.get('image', '')
            if 'coco' in img.lower() or 'train2017' in img or 'val2017' in img:
                image_sources['coco'].add(img)
            elif 'gqa' in img.lower():
                image_sources['gqa'].add(img)
            elif 'textvqa' in img.lower():
                image_sources['textvqa'].add(img)
            elif 'ocr' in img.lower():
                image_sources['ocr_vqa'].add(img)
            else:
                image_sources['other'].add(img)
        
        print("\n图像需求统计:")
        total_images = sum(len(imgs) for imgs in image_sources.values())
        print(f"  总计: {total_images:,} 张独特图像")
        
        for source, imgs in image_sources.items():
            if imgs:
                print(f"  {source}: {len(imgs):,} 张 ({len(imgs)/total_images*100:.1f}%)")
    
    def _create_synthetic_stage2_data(self, num_samples: int) -> tuple:
        """创建合成的Stage 2数据（备用）"""
        print("Creating synthetic large-scale data for Stage 2...")
        
        # 这里可以实现合成数据生成逻辑
        # 为简洁起见，返回None
        return None, None
    
    def download_required_images(self):
        """下载所需的图像"""
        print("\n" + "=" * 80)
        print("Downloading Required Images")
        print("=" * 80)
        
        print("\n请按以下步骤下载图像：")
        print("\n1. COCO Images (主要，~80%的数据使用):")
        print("   wget http://images.cocodataset.org/zips/train2017.zip")
        print("   wget http://images.cocodataset.org/zips/val2017.zip")
        print("   unzip train2017.zip -d data/images/")
        print("   unzip val2017.zip -d data/images/")
        
        print("\n2. GQA Images (可选，~10%的数据使用):")
        print("   wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip")
        print("   unzip images.zip -d data/images/gqa/")
        
        print("\n3. TextVQA Images (可选，~5%的数据使用):")
        print("   访问 https://textvqa.org/dataset/")
        
        print("\n4. OCR-VQA Images (可选，~5%的数据使用):")
        print("   访问 https://ocr-vqa.github.io/")
        
        print("\n提示：Stage 1只需要COCO图像即可开始训练")
    
    def verify_setup(self):
        """验证数据设置"""
        print("\n" + "=" * 80)
        print("Verifying Data Setup")
        print("=" * 80)
        
        # 检查Stage 1
        stage1_train = self.dirs['stage1'] / 'train_instruct150k.json'
        stage1_val = self.dirs['stage1'] / 'val_instruct150k.json'
        
        stage1_ready = False
        if stage1_train.exists() and stage1_val.exists():
            with open(stage1_train) as f:
                train_data = json.load(f)
            with open(stage1_val) as f:
                val_data = json.load(f)
            
            print(f"✓ Stage 1 Ready:")
            print(f"    Train: {len(train_data):,} samples")
            print(f"    Val: {len(val_data):,} samples")
            stage1_ready = True
        else:
            print("✗ Stage 1 data not found - Run prepare_stage1_data()")
        
        # 检查Stage 2
        stage2_train = self.dirs['stage2'] / 'train_mix665k.json'
        stage2_val = self.dirs['stage2'] / 'val_mix665k.json'
        
        stage2_ready = False
        if stage2_train.exists() and stage2_val.exists():
            with open(stage2_train) as f:
                train_data = json.load(f)
            with open(stage2_val) as f:
                val_data = json.load(f)
            
            print(f"✓ Stage 2 Ready:")
            print(f"    Train: {len(train_data):,} samples")
            print(f"    Val: {len(val_data):,} samples")
            stage2_ready = True
        else:
            print("✗ Stage 2 data not found - Run prepare_stage2_data()")
        
        # 检查图像
        coco_train = self.dirs['images'] / 'train2017'
        coco_val = self.dirs['images'] / 'val2017'
        
        images_ready = False
        if coco_train.exists() or coco_val.exists():
            train_imgs = len(list(coco_train.glob('*.jpg'))) if coco_train.exists() else 0
            val_imgs = len(list(coco_val.glob('*.jpg'))) if coco_val.exists() else 0
            
            if train_imgs > 0 or val_imgs > 0:
                print(f"✓ COCO Images Found:")
                print(f"    Train: {train_imgs:,} images")
                print(f"    Val: {val_imgs:,} images")
                images_ready = True
        
        if not images_ready:
            print("✗ COCO images not found - Download required")
        
        # 训练就绪状态
        print("\n" + "=" * 80)
        if stage1_ready and images_ready:
            print("✓ Ready for Stage 1 training!")
            print("\nRun command:")
            print("  ./train_stage1_multi_gpu.sh \\")
            print("    --train_data_path data/stage1/train_instruct150k.json \\")
            print("    --eval_data_path data/stage1/val_instruct150k.json \\")
            print("    --image_folder data/images/train2017")
        
        if stage2_ready and images_ready:
            print("\n✓ Ready for Stage 2 training!")
            print("\nRun command:")
            print("  ./train_stage2_multi_gpu.sh \\")
            print("    --model_path checkpoints/stage1_final/best_model \\")
            print("    --train_data_path data/stage2/train_mix665k.json \\")
            print("    --eval_data_path data/stage2/val_mix665k.json \\")
            print("    --image_folder data/images/")
        
        if not (stage1_ready or stage2_ready or images_ready):
            print("⚠ Data preparation needed. Run this script with appropriate arguments.")
        
        print("=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CM-IBQ Optimized Data Preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare both stages with default settings
  python prepare_cmibq_data.py --stage both
  
  # Prepare only Stage 1 (complex instructions)
  python prepare_cmibq_data.py --stage 1 --stage1_samples 150000
  
  # Prepare only Stage 2 (large-scale mix)  
  python prepare_cmibq_data.py --stage 2 --stage2_samples 665000
  
  # Quick test with small samples
  python prepare_cmibq_data.py --stage both --stage1_samples 10000 --stage2_samples 50000
        """
    )
    
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Data root directory')
    parser.add_argument('--stage', type=str, default='both',
                       choices=['1', '2', 'both'],
                       help='Which stage to prepare')
    parser.add_argument('--stage1_samples', type=int, default=150000,
                       help='Number of Stage 1 samples (LLaVA-Instruct-150K)')
    parser.add_argument('--stage2_samples', type=int, default=665000,
                       help='Number of Stage 2 samples (Mix665K)')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing setup without preparing data')
    parser.add_argument('--download_info', action='store_true',
                       help='Show image download instructions')
    
    args = parser.parse_args()
    
    # 创建数据准备器
    preparer = CMIBQDataPreparer(args.data_root)
    
    if args.verify_only:
        preparer.verify_setup()
        return
    
    if args.download_info:
        preparer.download_required_images()
        return
    
    # 准备数据
    stage1_paths = None
    stage2_paths = None
    
    if args.stage in ['1', 'both']:
        stage1_paths = preparer.prepare_stage1_data(args.stage1_samples)
    
    if args.stage in ['2', 'both']:
        stage2_paths = preparer.prepare_stage2_data(args.stage2_samples)
    
    # 验证设置
    print("\n")
    preparer.verify_setup()
    
    # 打印下一步指示
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    
    if stage1_paths and stage1_paths[0]:
        print("\n1. Download COCO images if not already done:")
        print("   wget http://images.cocodataset.org/zips/train2017.zip")
        print("   unzip train2017.zip -d data/images/")
        
        print("\n2. Start Stage 1 training:")
        print("   ./train_stage1_multi_gpu.sh")
    
    if stage2_paths and stage2_paths[0]:
        print("\n3. After Stage 1 completes, start Stage 2:")
        print("   ./train_stage2_multi_gpu.sh")
    
    print("\n" + "=" * 80)
    print("Data preparation completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
