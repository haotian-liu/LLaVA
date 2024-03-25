import os
import argparse
import json

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, required=True)
    parser.add_argument('--result-file', type=str, required=True)
    parser.add_argument('--result-upload-file', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(os.path.dirname(args.result_upload_file), exist_ok=True)

    results = []
    error_line = 0
    for line_idx, line in enumerate(open(args.result_file)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1
    results = {x['question_id']: x['text'] for x in results}
    test_split = [json.loads(line) for line in open(args.annotation_file)]
    split_ids = set([x['question_id'] for x in test_split])

    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []

    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        assert x['question_id'] in results
        all_answers.append({
            'image': x['image'],
            'answer': answer_processor(results[x['question_id']])
        })

    with open(args.result_upload_file, 'w') as f:
        json.dump(all_answers, f)
