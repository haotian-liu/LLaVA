import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import argparse
import requests
import base64
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from flask import Flask, request, jsonify, make_response


app = Flask(__name__)


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def get_args():
    parser = argparse.ArgumentParser(
        description='LLaVA Flask API'
    )

    parser.add_argument(
        '-p', '--port',
        help='Port to listen on',
        type=int,
        default=5000
    )

    parser.add_argument(
        '-H', '--host',
        help='Host to bind to',
        default='0.0.0.0'
    )

    return parser.parse_args()


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_image_from_base64(base64_str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return image


def run_inference(data):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(data['model_path'])

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        data['model_path'],
        data['model_base'],
        model_name,
        data['load_8bit'],
        data['load_4bit'],
        device='cuda'
    )

    if 'llama-2' in model_name.lower():
        conv_mode = 'llava_llama_2'
    elif 'v1' in model_name.lower():
        conv_mode = 'llava_v1'
    elif 'mpt' in model_name.lower():
        conv_mode = 'mpt'
    else:
        conv_mode = 'llava_v0'

    if data['conv_mode'] is not None and conv_mode != data['conv_mode']:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
            conv_mode,
            data['conv_mode'],
            data['conv_mode']
        ))
    else:
        data['conv_mode'] = conv_mode

    conv = conv_templates[data['conv_mode']].copy()

    if 'mpt' in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image_from_base64(data['image_base64'])
    image_tensor = process_images([image], image_processor, DictToObject(data))

    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = data['prompt']

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if data['stream']:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=data['temperature'],
            max_new_tokens=data['max_new_tokens'],
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    return outputs


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': f'{request.url} not found',
            'detail': str(error)
        }
    ), 404)


@app.errorhandler(500)
def internal_server_error(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': 'Internal Server Error',
            'detail': str(error)
        }
    ), 500)


@app.route('/')
def ping():
    return make_response(jsonify(
        {
            'status': 'ok'
        }
    ), 200)


@app.route('/inference', methods=['POST'])
def process_image():
    try:
        payload = request.get_json()

        data = {
            'model_path': payload.get('model_path', 'liuhaotian/llava-v1.5-13b'),
            'model_base': payload.get('model_base', None),
            'image_base64': payload.get('image_base64'),
            'prompt': payload.get('prompt'),
            'conv_mode': payload.get('conv_mode', None),
            'temperature': payload.get('temperature', 0.2),
            'max_new_tokens': payload.get('max_new_tokens', 512),
            'load_8bit': payload.get('load_8bit', False),
            'load_4bit': payload.get('load_4bit', False),
            'image_aspect_ratio': payload.get('image_aspect_ratio', 'pad'),
            'stream': payload.get('stream', False)
        }

        outputs = run_inference(data)

        return make_response(jsonify(
            {
                'status': 'ok',
                'response': outputs.replace('</s>', '')
            }
        ), 200)
    except Exception as e:
        return make_response(jsonify(
            {
                'status': 'error',
                'msg': 'Internal Server Error',
                'error': str(e)
            }
        ), 500)


if __name__ == '__main__':
    args = get_args()
    app.run(host=args.host, port=args.port)
