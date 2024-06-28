import os
import json
import uuid

def generate_sample(image_path, task, question, response):
    return {
        "id": str(uuid.uuid4()),
        "image": image_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": response}
        ]
    }

def collect_samples(base_dir, task, question_template, label_mapping):
    samples = []
    total_images = 0
    used_images = 0
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                total_images += 1
                label = file.split('_')[0]
                
                if label in label_mapping:
                    used_images += 1
                    question = question_template
                    response = f"Driver is {label_mapping[label]}."
                    full_image_path = os.path.join(root, file)
                    sample = generate_sample(full_image_path, task, question, response)
                    samples.append(sample)
    
    print(f"Total images in {base_dir}: {total_images}")
    print(f"Used images in {base_dir}: {used_images}")
    
    return samples




def main():
    emotion_dir = "/home/users/ntu/chih0001/scratch/data/emotion/FED_2/train"
    
    emotion_question = "<image>\nAnalyze the person's facial expression in the uploaded image. Look closely at the facial features such as the eyebrows, eyes, mouth, and overall facial tension."


    emotion_labels = {
        "AN": "Angry", "DI": "Disgust", "FE": "Fear", "HA": "Happy", "SA": "Sad", "SU": "Surprise"
    }

    samples = (collect_samples(emotion_dir, "emotion", emotion_question, emotion_labels))
    

    output_path = "/home/users/ntu/chih0001/scratch/VLM/LLaVA/train/lora_emotion.json"
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=4)

    print(f"JSON file has been saved to {output_path}")

if __name__ == "__main__":
    main()
