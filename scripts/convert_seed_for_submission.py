import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-upload-file", type=str)
    return parser.parse_args()


def eval_single(result_file, eval_only_type=None):
    results = {}
    for line in open(result_file):
        row = json.loads(line)
        results[row['question_id']] = row

    type_counts = {}
    correct_counts = {}
    for question_data in data['questions']:
        if eval_only_type is not None and question_data['data_type'] != eval_only_type: continue
        data_type = question_data['question_type_id']
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        try:
            question_id = int(question_data['question_id'])
        except:
            question_id = question_data['question_id']
        if question_id not in results:
            correct_counts[data_type] = correct_counts.get(data_type, 0)
            continue
        row = results[question_id]
        if row['text'] == question_data['answer']:
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

    total_count = 0
    total_correct = 0
    for data_type in sorted(type_counts.keys()):
        accuracy = correct_counts[data_type] / type_counts[data_type] * 100
        if eval_only_type is None:
            print(f"{ques_type_id_to_name[data_type]}: {accuracy:.2f}%")

        total_count += type_counts[data_type]
        total_correct += correct_counts[data_type]

    total_accuracy = total_correct / total_count * 100
    if eval_only_type is None:
        print(f"Total accuracy: {total_accuracy:.2f}%")
    else:
        print(f"{eval_only_type} accuracy: {total_accuracy:.2f}%")

    return results

if __name__ == "__main__":
    args = get_args()
    data = json.load(open(args.annotation_file))
    ques_type_id_to_name = {id:n for n,id in data['question_type'].items()}

    results = eval_single(args.result_file)
    eval_single(args.result_file, eval_only_type='image')
    eval_single(args.result_file, eval_only_type='video')

    with open(args.result_upload_file, 'w') as fp:
        for question in data['questions']:
            qid = question['question_id']
            if qid in results:
                result = results[qid]
            else:
                result = results[int(qid)]
            fp.write(json.dumps({
                'question_id': qid,
                'prediction': result['text']
            }) + '\n')
