import argparse
import json
import os
import re
import random
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--gpt4-result', type=str)
    parser.add_argument('--requery-result', type=str)
    parser.add_argument('--our-result', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def convert_caps(results):
    fakecaps = []
    for result in results:
        image_id = result['question_id']
        caption = result['text']
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))
    our_predictions = [json.loads(line) for line in open(args.our_result)]
    our_predictions = {pred['question_id']: pred for pred in our_predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    requery_predictions = [json.loads(line) for line in open(args.requery_result)]
    requery_predictions = {pred['question_id']: pred for pred in requery_predictions}

    gpt4_predictions = json.load(open(args.gpt4_result))['outputs']

    results = defaultdict(lambda: 0)

    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}

    for prob_id, prob in split_problems.items():
        if prob_id not in our_predictions:
            assert False
        if prob_id not in gpt4_predictions:
            assert False
        our_pred = our_predictions[prob_id]['text']
        gpt4_pred = gpt4_predictions[prob_id]
        if prob_id not in requery_predictions:
            results['missing_requery'] += 1
            requery_pred = "MISSING"
        else:
            requery_pred = requery_predictions[prob_id]['text']

        pattern = re.compile(r'The answer is ([A-Z]).')
        our_res = pattern.findall(our_pred)
        if len(our_res) == 1:
            our_answer = our_res[0]  # 'A', 'B', ...
        else:
            our_answer = "FAILED"

        requery_res = pattern.findall(requery_pred)
        if len(requery_res) == 1:
            requery_answer = requery_res[0]  # 'A', 'B', ...
        else:
            requery_answer = "FAILED"

        gpt4_res = pattern.findall(gpt4_pred)
        if len(gpt4_res) == 1:
            gpt4_answer = gpt4_res[0]  # 'A', 'B', ...
        else:
            gpt4_answer = "FAILED"

        our_pred_idx = get_pred_idx(our_answer, prob['choices'], args.options)
        gpt4_pred_idx = get_pred_idx(gpt4_answer, prob['choices'], args.options)
        requery_pred_idx = get_pred_idx(requery_answer, prob['choices'], args.options)

        results['total'] += 1

        if gpt4_answer == 'FAILED':
            results['gpt4_failed'] += 1
            if gpt4_pred_idx == prob['answer']:
                results['gpt4_correct'] += 1
            if our_pred_idx == prob['answer']:
                results['gpt4_ourvisual_correct'] += 1
        elif gpt4_pred_idx == prob['answer']:
            results['gpt4_correct'] += 1
            results['gpt4_ourvisual_correct'] += 1

        if our_pred_idx == prob['answer']:
            results['our_correct'] += 1

        if requery_answer == 'FAILED':
            sqa_results['results'][prob_id] = our_pred_idx
            if our_pred_idx == prob['answer']:
                results['requery_correct'] += 1
        else:
            sqa_results['results'][prob_id] = requery_pred_idx
            if requery_pred_idx == prob['answer']:
                results['requery_correct'] += 1
            else:
                print(f"""
Question ({args.options[prob['answer']]}): {our_predictions[prob_id]['prompt']}
Our ({our_answer}): {our_pred}
GPT-4 ({gpt4_answer}): {gpt4_pred}
Requery ({requery_answer}): {requery_pred}
print("=====================================")
""")

        if gpt4_pred_idx == prob['answer'] or our_pred_idx == prob['answer']:
            results['correct_upperbound'] += 1

    total = results['total']
    print(f'Total: {total}, Our-Correct: {results["our_correct"]}, Accuracy: {results["our_correct"] / total * 100:.2f}%')
    print(f'Total: {total}, GPT-4-Correct: {results["gpt4_correct"]}, Accuracy: {results["gpt4_correct"] / total * 100:.2f}%')
    print(f'Total: {total}, GPT-4 NO-ANS (RANDOM): {results["gpt4_failed"]}, Percentage: {results["gpt4_failed"] / total * 100:.2f}%')
    print(f'Total: {total}, GPT-4-OursVisual-Correct: {results["gpt4_ourvisual_correct"]}, Accuracy: {results["gpt4_ourvisual_correct"] / total * 100:.2f}%')
    print(f'Total: {total}, Requery-Correct: {results["requery_correct"]}, Accuracy: {results["requery_correct"] / total * 100:.2f}%')
    print(f'Total: {total}, Correct upper: {results["correct_upperbound"]}, Accuracy: {results["correct_upperbound"] / total * 100:.2f}%')

    sqa_results['acc'] = results["requery_correct"] / total * 100
    sqa_results['correct'] = results["requery_correct"]
    sqa_results['count'] = total

    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)

