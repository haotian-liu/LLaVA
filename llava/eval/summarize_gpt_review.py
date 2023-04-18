import json
import os
from collections import defaultdict

import numpy as np


if __name__ == '__main__':
    base_dir = "vqa/reviews/coco2014_val80"
    review_files = [x for x in os.listdir(base_dir) if x.endswith('.jsonl') and x.startswith('gpt4_text')]

    for review_file in sorted(review_files):
        config = review_file.replace('gpt4_text_', '').replace('.jsonl', '')
        scores = defaultdict(list)
        print(f'GPT-4 vs. {config}')
        with open(os.path.join(base_dir, review_file)) as f:
            for review_str in f:
                review = json.loads(review_str)
                scores[review['category']].append(review['tuple'])
                scores['all'].append(review['tuple'])
        for k, v in scores.items():
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            print(k, stats, round(stats[1]/stats[0]*100, 1))
        print('=================================')

