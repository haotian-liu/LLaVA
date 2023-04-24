import json
import os
import fire
import re
from convert_sqa_to_llava_base_prompt import build_prompt_chatbot


def convert_to_llava(base_dir, split, prompt_format="QCM-LEPA"):
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    split_problems = build_prompt_chatbot(
        problems, split_indices, prompt_format,
        use_caption=False, is_test=False)

    target_format = []
    for prob_id, (input, output) in split_problems.items():
        if input.startswith('Question: '):
            input = input.replace('Question: ', '')
        if output.startswith('Answer: '):
            output = output.replace('Answer: ', '')

        raw_prob_data = problems[prob_id]
        if raw_prob_data['image'] is None:
            target_format.append({
                "id": prob_id,
                "conversations": [
                    {'from': 'human', 'value': f"{input}"},
                    {'from': 'gpt', 'value': f"{output}"},
                ],
            })

        else:
            target_format.append({
                "id": prob_id,
                "image": os.path.join(prob_id, raw_prob_data['image']),
                "conversations": [
                    {'from': 'human', 'value': f"{input}\n<image>"},
                    {'from': 'gpt', 'value': f"{output}"},
                ],
            })

    print(f'Number of samples: {len(target_format)}')

    with open(os.path.join(base_dir, f"llava_{split}_{prompt_format}.json"), "w") as f:
        json.dump(target_format, f, indent=2)


def convert_to_jsonl(base_dir, split, prompt_format="QCM-LEPA"):
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    split_problems = build_prompt_chatbot(
        problems, split_indices, prompt_format,
        use_caption=False, is_test=False)

    writer = open(os.path.join(base_dir, f"scienceqa_{split}_{prompt_format}.jsonl"), "w")
    for prob_id, (input, output) in split_problems.items():
        if input.startswith('Question: '):
            input = input.replace('Question: ', '')
        if output.startswith('Answer: '):
            output = output.replace('Answer: ', '')

        raw_prob_data = problems[prob_id]
        if raw_prob_data['image'] is None:
            data = {
                "id": prob_id,
                "instruction": f"{input}",
                "output": f"{output}",
            }

        else:
            data = {
                "id": prob_id,
                "image": os.path.join(prob_id, raw_prob_data['image']),
                "instruction": f"{input}\n<image>",
                "output": f"{output}",
            }
        writer.write(json.dumps(data) + '\n')
    writer.close()


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
