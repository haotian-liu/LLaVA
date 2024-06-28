import json

# 读取JSON文件
# with open('/home/users/ntu/chih0001/scratch/VLM/LLaVA/train/lora.json') as f:
with open('/home/users/ntu/chih0001/scratch/VLM/LLaVA/train/lora_distraction.json') as f:
    data = json.load(f)

# 计算条目数量
num_entries = len(data)

print("JSON文件中的条目数量为:", num_entries)
