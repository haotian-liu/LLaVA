# Fast API Hippo Llava API on Port

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Extra
from typing import TYPE_CHECKING, List, Optional, Union

from .cli import load_image
import argparse
import torch
import uvicorn

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.utils import build_logger, server_error_msg

logger = build_logger("hippoapi", "hippoapi.log")
from PIL import Image

from transformers import TextStreamer
app = FastAPI()

# request model
class QueryModel(BaseModel):
    temperature: Optional[float] = 0.2
    max_new_tokens: Optional[int] = 512
    load_8bit: Optional[bool] = True
    image_url: str
    prompt: str

# Query AI modol with request body include image url and text (prompt)
@app.post("/query")
async def query(request: QueryModel):
    disable_torch_init()
    # convert request to dict
    data = request.dict()
    # Hard code
    data['load_4bit'] = False
    data['model_base'] = None
    data['device'] = "cuda"
    data['conv_mode'] = None
    # Path on server
    model_path = "liuhaotian/llava-v1.5-13b"
    model_name = get_model_name_from_path(model_path)
    # Model
    print("Model name: ", model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, data['model_base'], model_name, data['load_8bit'], data['load_4bit'], device=data['device'])
    print("Model loaded")
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if data['conv_mode'] is not None and conv_mode != data['conv_mode']:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, data['conv_mode'], data['conv_mode']))
    else:
        data['conv_mode'] = conv_mode

    conv = conv_templates[data['conv_mode']].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(data['image_url'])
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    # Return text simulate user input and image
    try:
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
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if data['temperature'] > 0 else False,
                temperature=data['temperature'],
                max_new_tokens=data['max_new_tokens'],
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()         
        conv.messages[-1][-1] = outputs
        return {"message": outputs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()
    logger.info(f"args: {args}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
