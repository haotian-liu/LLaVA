import os
import json

for file in os.listdir('answers'):
    if not file.endswith('.jsonl'):
        continue
    if os.path.isfile(os.path.join('results', file.replace('.jsonl', '.json'))):
        continue

    cur_result = {}

    for line in open(os.path.join('answers', file)):
        data = json.loads(line)
        qid = data['question_id']
        cur_result[f'v1_{qid}'] = data['text']

    with open(os.path.join('results', file.replace('.jsonl', '.json')), 'w') as f:
        json.dump(cur_result, f, indent=2)
