import argparse
import json
import os
import re
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def convert_caps(results):
    return [{"image_id": int(result['question_id']), "caption": result['text']} for result in results]


def get_pred_idx(prediction, choices, options):
    return options.index(prediction) if prediction in options[:len(choices)] else random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    with open(os.path.join(base_dir, "pid_splits.json")) as f:
        split_indices = set(json.load(f)[args.split])
    with open(os.path.join(base_dir, "problems.json")) as f:
        problems = json.load(f)
    with open(args.result_file) as f:
        predictions = {pred['question_id']: json.loads(line) for line in f}

    split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'correct': [], 'incorrect': []}
    sqa_results = {'acc': None, 'correct': None, 'count': None, 'results': {}, 'outputs': {}}

    pattern = re.compile(r'The answer is ([A-Z]).')

    for prob_id, prob in split_problems.items():
        pred = predictions.get(prob_id, {'text': 'FAILED', 'prompt': 'Unknown'})
        pred_text = pred['text']

        if pred_text in args.options:
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in args.options and pred_text[1:3] == ". ":
            answer = pred_text[0]
        else:
            res = pattern.findall(pred_text)
            answer = res[0] if len(res) == 1 else "FAILED"

        pred_idx = get_pred_idx(answer, prob['choices'], args.options)

        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': args.options[prob['answer']],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        sqa_results['results'][prob_id] = pred_idx
        sqa_results['outputs'][prob_id] = pred_text

        (results['correct'] if pred_idx == prob['answer'] else results['incorrect']).append(analysis)

    correct = len(results['correct'])
    total = correct + len(results['incorrect'])

    multimodal_correct = sum(1 for x in results['correct'] if x['is_multimodal'])
    multimodal_total = multimodal_correct + sum(1 for x in results['incorrect'] if x['is_multimodal'])

    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    with open(args.output_file, 'w') as f
