import json
import requests
import base64


BASE_URI = 'https://example.com'


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return str(base64.b64encode(image_file.read()).decode('utf-8'))


if __name__ == '__main__':
    payload = {
        'model_path': 'liuhaotian/llava-v1.5-7b',
        'image_base64': encode_image_to_base64('examples/waterview.jpg'),
        'prompt': 'What are the things I should be cautious about when I visit here?',
        'temperature': 0.2,
        'max_new_tokens': 512
    }

    r = requests.post(
        f'{BASE_URI}/inference',
        json=payload
    )

    print(r.status_code)
    resp_json = r.json()
    print(json.dumps(resp_json, indent=4, default=str))
