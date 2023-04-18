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
    parser.add_argument('--our-result', type=str)
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

    gpt4_predictions = json.load(open(args.gpt4_result))['outputs']

    results = defaultdict(lambda: 0)

    for prob_id, prob in split_problems.items():
        if prob_id not in our_predictions:
            continue
        if prob_id not in gpt4_predictions:
            continue
        our_pred = our_predictions[prob_id]['text']
        gpt4_pred = gpt4_predictions[prob_id]

        pattern = re.compile(r'The answer is ([A-Z]).')
        our_res = pattern.findall(our_pred)
        if len(our_res) == 1:
            our_answer = our_res[0]  # 'A', 'B', ...
        else:
            our_answer = "FAILED"
        gpt4_res = pattern.findall(gpt4_pred)
        if len(gpt4_res) == 1:
            gpt4_answer = gpt4_res[0]  # 'A', 'B', ...
        else:
            gpt4_answer = "FAILED"

        our_pred_idx = get_pred_idx(our_answer, prob['choices'], args.options)
        gpt4_pred_idx = get_pred_idx(gpt4_answer, prob['choices'], args.options)

        if gpt4_answer == 'FAILED':
            results['gpt4_failed'] += 1
            # continue
            gpt4_pred_idx = our_pred_idx
            # if our_pred_idx != prob['answer']:
            #     print(our_predictions[prob_id]['prompt'])
            #     print('-----------------')
            #     print(f'LECTURE: {prob["lecture"]}')
            #     print(f'SOLUTION: {prob["solution"]}')
            #     print('=====================')
        else:
            # continue
            pass
        # gpt4_pred_idx = our_pred_idx

        if gpt4_pred_idx == prob['answer']:
            results['correct'] += 1
        else:
            results['incorrect'] += 1


        if gpt4_pred_idx == prob['answer'] or our_pred_idx == prob['answer']:
            results['correct_upperbound'] += 1

    correct = results['correct']
    total = results['correct'] + results['incorrect']
    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%')
    print(f'Total: {total}, Correct (upper): {results["correct_upperbound"]}, Accuracy: {results["correct_upperbound"] / total * 100:.2f}%')
    print(f'Total: {total}, GPT-4 NO-ANS (RANDOM): {results["gpt4_failed"]}, Percentage: {results["gpt4_failed"] / total * 100:.2f}%')

