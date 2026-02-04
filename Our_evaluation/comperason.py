import json
import pandas as pd
import re
import argparse


def load_preds(path):
    rows = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)

            qtext = ex["question"]

            # count answer options by detecting (A), (B), ...
            options = re.findall(r"\([A-E]\)", qtext)
            n_options = len(options)

            rows.append({
                "qid": ex["question_id"],
                "pred": ex["parsed_ans"],
                "gt": ex["ground_truth"],
                "correct": ex["parsed_ans"] == ex["ground_truth"],
                "n_options": n_options,
                "setting": "AE"  # change per file
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--output-format", type=str, default="letter", choices=["letter", "cot_reason_first", "cot_answer_first"], help="Force model output format.")
    args = parser.parse_args()

    eval_model(args)