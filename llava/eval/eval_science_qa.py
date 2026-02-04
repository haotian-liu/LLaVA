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
        return -1
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'correct': [], 'incorrect': []}
    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}

    for prob_id, prob in split_problems.items():
        if prob_id not in predictions:
            pred = {'text': 'FAILED', 'prompt': 'Unknown'}
            pred_text = 'FAILED'
        else:
            pred = predictions[prob_id]
            pred_text = pred['text']

        ipred_text = pred_text.strip()

        pred_text = pred_text or ""
        ipred_text = pred_text.strip()

        answer = "FAILED"

        # 1) Exact match (after strip)
        if ipred_text in args.options:
            answer = ipred_text

        # 2) Leading letter + optional punctuation or whitespace/newline, then more text
        #    Examples: "A\n...", "A. ...", "A) ...", "A: ...", "A, ..."
        elif len(ipred_text) >= 1 and ipred_text[0] in args.options:
            if len(ipred_text) == 1:
                answer = ipred_text[0]
            else:
                # allow punctuation OR whitespace/newline immediately after the letter
                if ipred_text[1] in [".", ")", ":", ",", " ", "\n", "\t", "\r"]:
                    answer = ipred_text[0]

        # 3) "Final answer: X" anywhere (tolerate punctuation after X)
        if answer == "FAILED":
            m = re.findall(r'final answer\s*:\s*([A-Z])[\.\)\:\,]?', ipred_text, flags=re.IGNORECASE)
            if m:
                cand = m[-1].upper()
                if cand in args.options:
                    answer = cand

        # 4) Fallback: first standalone option letter on its own line
        #    This catches:
        #      "A\n\nThe continent..."
        #      "B\nReasoning: ..."
        if answer == "FAILED":
            m = re.search(r'^\s*([A-Z])\s*$', pred_text, flags=re.MULTILINE)
            if m:
                cand = m.group(1).upper()
                if cand in args.options:
                    answer = cand

        # 5) Final fallback: first occurrence of a valid option letter as a standalone token
        #    Avoids grabbing letters inside words like "Africa"
        if answer == "FAILED":
            m = re.search(r'\b([A-Z])\b', ipred_text)
            if m:
                cand = m.group(1).upper()
                if cand in args.options:
                    answer = cand



        pred_idx = get_pred_idx(answer, prob['choices'], args.options)

        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': args.options[prob['answer']],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        sqa_results['results'][prob_id] = get_pred_idx(answer, prob['choices'], args.options)
        sqa_results['outputs'][prob_id] = pred_text

        if pred_idx == prob['answer']:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%')

    # ###### IMG ######
    # multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    # multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    # multimodal_total = multimodal_correct + multimodal_incorrect
    # print(f'IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')
    # ###### IMG ######

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)
