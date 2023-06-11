import os
import argparse
import torch
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Extract MMProjector weights')
    parser.add_argument('--model_name_or_path', type=str, help='model folder')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    keys_to_match = ['mm_projector', 'embed_tokens', 'transformer.wte']
    ckpt_to_key = defaultdict(list)
    try:
        model_indices = json.load(open(os.path.join(args.model_name_or_path, 'pytorch_model.bin.index.json')))
        for k, v in model_indices['weight_map'].items():
            if any(key_match in k for key_match in keys_to_match):
                ckpt_to_key[v].append(k)
    except FileNotFoundError:
        # Smaller models or model checkpoints saved by DeepSpeed.
        v = 'pytorch_model.bin'
        for k in torch.load(os.path.join(args.model_name_or_path, v), map_location='cpu').keys():
            if any(key_match in k for key_match in keys_to_match):
                ckpt_to_key[v].append(k)

    loaded_weights = {}

    for ckpt_name, weight_keys in ckpt_to_key.items():
        ckpt = torch.load(os.path.join(args.model_name_or_path, ckpt_name), map_location='cpu')
        for k in weight_keys:
            loaded_weights[k] = ckpt[k]

    torch.save(loaded_weights, args.output)
