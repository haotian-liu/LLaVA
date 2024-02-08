import argparse
import ast
import datetime
import json
import os
import time

import gradio as gr
import requests

from llava.conversation import (default_conversation, conv_templates,
                                SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
                         violates_moderation, moderation_msg)
import hashlib

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(
                value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, chat_history, image, image_process_mode, include_image, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(include_image=include_image), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(include_image=include_image), moderation_msg, None) + (
                no_change_btn,) * 5

    # handle passed-in chat history
    if not chat_history:
        chat_history = []
    if isinstance(chat_history, str):
        chat_history = ast.literal_eval(chat_history)
        assert isinstance(chat_history, list), "Chat history must be a list: %s" % chat_history

    chat_history0 = chat_history.copy()
    chat_history = []
    for chati, chat in enumerate(chat_history0):
        if chat and chat[0] and isinstance(chat[0], str) and chat[1] and isinstance(chat[1], str):
            chat_history.append(chat)
        elif chat and chat[0] and isinstance(chat[0], str):
            chat_history.append([chat[0], "Image Generated"])
        elif chat and chat[1] and isinstance(chat[1], str):
            chat_history.append(["User Uploaded Image", chat[1]])

    if chat_history and chat_history[0] and chat_history[0][0]:
        in_history = True
        text_with_image = chat_history[0][0]
    else:
        in_history = False
        text_with_image = text

    if image is not None:
        text_with_image = text_with_image[:1200]  # Hard cut-off for images
    else:
        text_with_image = text_with_image[:1536]  # Hard cut-off

    if image is not None:
        if '<image>' not in text_with_image:
            # text = '<Image><image></Image>' + text
            text_with_image += '\n<image>'
        text_with_image = (text_with_image, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    if in_history:
        chat_history[0][0] = text_with_image
    else:
        text = text_with_image

    if chat_history:
        for chat in chat_history:
            if chat and chat[0]:
                state.append_message(state.roles[0], chat[0])
            if chat and chat[1]:
                state.append_message(state.roles[1], chat[1])

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)

    state.skip_next = False
    return (state, state.to_gradio_chatbot(include_image=include_image), "", None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, include_image, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        if include_image:
            yield (state, state.to_gradio_chatbot(include_image=include_image)) + (no_change_btn,) * 5
        else:
            yield state, state.to_gradio_chatbot(include_image=include_image)
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        if include_image:
            yield (state, state.to_gradio_chatbot(include_image=include_image), disable_btn, disable_btn, disable_btn,
                   enable_btn, enable_btn)
        else:
            yield state, state.to_gradio_chatbot(include_image=include_image)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    # stream_marker = "▌"
    stream_marker = ""
    state.messages[-1][-1] = stream_marker
    if include_image:
        yield (state, state.to_gradio_chatbot(include_image=include_image)) + (disable_btn,) * 5
    else:
        yield state, state.to_gradio_chatbot(include_image=include_image)

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + stream_marker
                    if include_image:
                        yield (state, state.to_gradio_chatbot(include_image=include_image)) + (disable_btn,) * 5
                    else:
                        yield state, state.to_gradio_chatbot(include_image=include_image)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    if include_image:
                        yield (state, state.to_gradio_chatbot(include_image=include_image)) + (
                        disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    else:
                        yield state, state.to_gradio_chatbot(include_image=include_image)
                    return
                time.sleep(0.01)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        if include_image:
            yield (state, state.to_gradio_chatbot(include_image=include_image)) + (
            disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        else:
            yield state, state.to_gradio_chatbot(include_image=include_image)
        return

    # state.messages[-1][-1] = state.messages[-1][-1][:-1]
    if include_image:
        yield (state, state.to_gradio_chatbot(include_image=include_image)) + (enable_btn,) * 5
    else:
        yield state, state.to_gradio_chatbot(include_image=include_image)

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def build_demo(concurrency_count=10):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    textbox_api = gr.Textbox(visible=False)
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=True)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image?"],
                    [f"{cur_dir}/examples/waterview.jpg",
                     "What are the things I should be cautious about when I visit here?"],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                            label="Temperature", )
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                                  label="Max output tokens", )
                    chat_history = gr.Textbox(value='[]', show_label=True, label="Enter chat_history as [['human', 'bot']]")

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLaVA Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                        submit_api_btn = gr.Button(value="Send", variant="primary", visible=False)
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            #queue=False,
            api_name='upvote_click',
            concurrency_limit=None,
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            #queue=False,
            api_name='downvote_click',
            concurrency_limit=None,
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            #queue=False,
            api_name='flag_click',
            concurrency_limit=None,
        )

        include_image = gr.Checkbox(value=True, label="Include Image in Chat")

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            #queue=False,
            concurrency_limit=None,
            api_name='regenerate_btn',
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, include_image],
            [state, chatbot] + btn_list,
            api_name='regenerate_click',
            concurrency_limit=concurrency_count,
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            #queue=False,
            concurrency_limit=None,
            api_name='clear',
        )

        textbox.submit(
            add_text,
            [state, textbox, chat_history, imagebox, image_process_mode, include_image],
            [state, chatbot, textbox, imagebox] + btn_list,
            #queue=False,
            concurrency_limit=None,
            api_name='textbox_btn',
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, include_image],
            [state, chatbot] + btn_list,
            api_name='textbox_submit',
            concurrency_limit=concurrency_count,
        )

        textbox_api.submit(
            add_text,
            [state, textbox, chat_history, imagebox, image_process_mode, include_image],
            [state, chatbot],
            #queue=False,
            concurrency_limit=None,
            api_name='textbox_api_btn',
            # preprocess=False,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, include_image],
            [state, chatbot],
            api_name='textbox_api_submit',
            concurrency_limit=concurrency_count,
            # preprocess=False,
            # postprocess=False,
        )

        submit_btn.click(
            add_text,
            [state, textbox, chat_history, imagebox, image_process_mode, include_image],
            [state, chatbot, textbox, imagebox] + btn_list,
            #queue=False,
            concurrency_limit=None,
            api_name='submit_btn',
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, include_image],
            [state, chatbot] + btn_list,
            api_name='submit_click',
            concurrency_limit=concurrency_count,
        )

        demo_setup_kwargs = dict(fn=load_demo_refresh_model_list,
                                 inputs=None,
                                 outputs=[state, model_selector])
        demo_btn = gr.Button(visible=False)
        demo_btn.click(**demo_setup_kwargs, api_name='demo_load')

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
#                js=get_window_url_params
#            )
#        elif args.model_list_mode == "reload":
#            demo.load(
#                load_demo_refresh_model_list,
#                None,
#                [state, model_selector],
                #concurrency_limit=None,
                #queue=False
            )
        elif args.model_list_mode == "reload":
            demo.load(**demo_setup_kwargs,
                      #concurrency_limit=None,
                      #queue=False
                      )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        # Handle uploads from API
        upload_api_btn = gr.UploadButton("Upload File Results", visible=False)
        file_upload_api = gr.File(visible=False)
        file_upload_text = gr.Textbox(visible=False)

        def upload_file(files):
            if isinstance(files, list):
                file_paths = [file.name for file in files]
            elif isinstance(files, dict):
                file_paths = files['name']
                assert os.path.isfile(file_paths)
            else:
                file_paths = files.name
            return file_paths, file_paths

        upload_api_btn.upload(fn=upload_file,
                              inputs=upload_api_btn,
                              outputs=[file_upload_api, file_upload_text],
                              api_name='upload_api',
                              preprocess=False)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
                        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    #parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo()
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        api_open=True
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
