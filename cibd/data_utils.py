import json
import os
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple


def prepare_llava_data(
    llava_json_path: str,
    train_ratio: float = 0.95,
    val_ratio: float = 0.05,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    准备LLaVA数据，分割训练集和验证集
    
    Args:
        llava_json_path: LLaVA JSON文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
    
    Returns:
        train_data, val_data
    """
    with open(llava_json_path, 'r') as f:
        data = json.load(f)
    
    # 分割数据
    train_data, val_data = train_test_split(
        data,
        test_size=val_ratio,
        random_state=random_seed
    )
    
    print(f"Data split: {len(train_data)} training, {len(val_data)} validation")
    
    return train_data, val_data


def create_data_splits(
    llava_json_path: str,
    output_dir: str
):
    """
    创建数据分割并保存
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_data, val_data = prepare_llava_data(llava_json_path)
    
    # 保存分割后的数据
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
        
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
        
    print(f"Saved splits to {output_dir}")
    

def analyze_dataset(data_path: str):
    """分析数据集统计信息"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 统计信息
    total_samples = len(data)
    has_image = sum(1 for item in data if 'image' in item)
    
    # 对话轮数统计
    conversation_lengths = [len(item['conversations']) for item in data]
    avg_turns = sum(conversation_lengths) / len(conversation_lengths)
    
    # 文本长度统计
    text_lengths = []
    for item in data:
        text = ' '.join([conv['value'] for conv in item['conversations']])
        text_lengths.append(len(text.split()))
    avg_text_length = sum(text_lengths) / len(text_lengths)
    
    print(f"Dataset Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Samples with images: {has_image}")
    print(f"  Average conversation turns: {avg_turns:.2f}")
    print(f"  Average text length (words): {avg_text_length:.2f}")
    print(f"  Min text length: {min(text_lengths)}")
    print(f"  Max text length: {max(text_lengths)}")
    
    return {
        'total_samples': total_samples,
        'has_image': has_image,
        'avg_turns': avg_turns,
        'avg_text_length': avg_text_length
    }