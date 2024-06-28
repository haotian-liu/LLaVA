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
                label = None
                
                if task == "distraction":
                    label = file.split('_')[0]
                elif task == "emotion":
                    label = file.split('_')[1]
                elif task == "drowsiness":
                    label = file.split('_')[-1].split('.')[0]
                
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
    distraction_dir = "/home/users/ntu/chih0001/scratch/data/distraction/SAM-DD/train"
    emotion_dir = "/home/users/ntu/chih0001/scratch/data/emotion/FED_2/train"
    drowsiness_dir = "/home/users/ntu/chih0001/scratch/data/drowsiness/NTHUDDD_2/train"

    distraction_question = "<image>\n**Identify Driver Behavior from Image**\n\n**Task:** What behavior is the driver exhibiting?\n\n**Options:**\n\n- Normal Driving\n- Drinking\n- Phoning Left\n- Phoning Right\n- Texting Left\n- Texting Right\n- Touching Hairs & Makeup\n- Adjusting Glasses\n- Reaching Behind\n- Dropping\n\n**Instructions:** Analyze the image and identify the driver's behavior."
    emotion_question = "<image>\nAnalyze the person's facial expression in the uploaded image. Look closely at the facial features such as the eyebrows, eyes, mouth, and overall facial tension."
    drowsiness_question = "<image>\nThis image captures a crucial moment in a driver's journey. Your task is to observe the driver's posture, facial expression, and overall appearance carefully."

    distraction_labels = {
        "0": "Normal Driving", "1": "Drinking", "2": "Phoning Left", "3": "Phoning Right",
        "4": "Texting Left", "5": "Texting Right", "6": "Touching Hairs & Makeup",
        "7": "Adjusting Glasses", "8": "Reaching Behind", "9": "Dropping"
    }
    emotion_labels = {
        "AN": "Angry", "DI": "Disgust", "FE": "Fear", "HA": "Happy", "SA": "Sad", "SU": "Surprise"
    }
    drowsiness_labels = {
        "notdrowsy": "Non Drowsy", "drowsy": "Drowsy"
    }

    samples = []
    samples.extend(collect_samples(distraction_dir, "distraction", distraction_question, distraction_labels))
    samples.extend(collect_samples(emotion_dir, "emotion", emotion_question, emotion_labels))
    samples.extend(collect_samples(drowsiness_dir, "drowsiness", drowsiness_question, drowsiness_labels))

    output_path = "/home/users/ntu/chih0001/scratch/VLM/LLaVA/train/lora.json"
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=4)

    print(f"JSON file has been saved to {output_path}")

if __name__ == "__main__":
    main()
