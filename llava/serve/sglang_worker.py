"""
A model worker executes the model.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import re
import uvicorn
from functools import partial

from llava.constants import WORKER_HEART_BEAT_INTERVAL
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, expand2square
from llava.constants import DEFAULT_IMAGE_TOKEN

import sglang as sgl
from sglang.backend.runtime_endpoint import RuntimeEndpoint


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


@sgl.function
def pipeline(s, prompt, max_tokens):
    for p in prompt:
        if type(p) is str:
            s += p
        else:
            s += sgl.image(p)
    s += sgl.gen("response", max_tokens=max_tokens)


class ModelWorker:
    def __init__(self, controller_addr, worker_addr, sgl_endpoint,
                 worker_id, no_register, model_name):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id

        # Select backend
        backend = RuntimeEndpoint(sgl_endpoint)
        sgl.set_default_backend(backend)
        model_path = backend.model_info["model_path"]

        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        logger.info(f"Loading the SGLANG model {self.model_name} on worker {worker_id} ...")

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True)
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    async def generate_stream(self, params):
        ori_prompt = prompt = params["prompt"]
        images = params.get("images", None)
        if images is not None and len(images) > 0:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]

                # FIXME: for image-start/end token
                # replace_token = DEFAULT_IMAGE_TOKEN
                # if getattr(self.model.config, 'mm_use_im_start_end', False):
                #     replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                # prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                prompt = prompt.replace(' ' + DEFAULT_IMAGE_TOKEN + '\n', DEFAULT_IMAGE_TOKEN)
                prompt_split = prompt.split(DEFAULT_IMAGE_TOKEN)
                prompt = []
                for i in range(len(prompt_split)):
                    prompt.append(prompt_split[i])
                    if i < len(images):
                        prompt.append(images[i])
        else:
            prompt = [prompt]

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        # max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        stop_str = [stop_str] if stop_str is not None else None

        print({'prompt': prompt, 'max_new_tokens': max_new_tokens, 'temperature': temperature, 'top_p': top_p})
        state = pipeline.run(prompt, max_new_tokens, temperature=temperature, top_p=top_p, stream=True)

        generated_text = ori_prompt
        async for text_outputs in state.text_async_iter(var_name="response"):
            generated_text += text_outputs
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    async def generate_stream_gate(self, params):
        try:
            async for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--sgl-endpoint", type=str)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         args.sgl_endpoint,
                         worker_id,
                         args.no_register,
                         args.model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
