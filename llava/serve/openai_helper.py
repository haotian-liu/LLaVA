from typing import List, Optional, Literal, Union, Dict, Any
from enum import Enum
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import requests

from io import BytesIO

from llava.utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)


from llava.mm_utils import load_image_from_base64


from llava.conversation import SeparatorStyle, conv_templates, default_conversation

logger = build_logger("openai-api", f"openai-api.log")


class TextMessage(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImageURLMessage(BaseModel):
    type: Literal["image_url"]
    image_url: Union[ImageURL, str]


class AssistantMessage(BaseModel):
    content: str


class GPTVMessage(BaseModel):
    role: str
    content: Union[List[Union[TextMessage, ImageURLMessage]], str]


class ConcatOptions(Enum):
    Horizontal = "horizontal"
    Vertical = "vertical"
    No = "no"


class GPTVChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[GPTVMessage]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    concat: Optional[ConcatOptions] = ConcatOptions.No


def get_template(model_name):
    if "llava" in model_name.lower():
        if "llama-2" in model_name.lower():
            template_name = "llava_llama_2"
        elif "v1" in model_name.lower():
            if "mmtag" in model_name.lower():
                template_name = "v1_mmtag"
            elif "plain" in model_name.lower() and "finetune" not in model_name.lower():
                template_name = "v1_mmtag"
            else:
                template_name = "llava_v1"
        elif "mpt" in model_name.lower():
            template_name = "mpt"
        else:
            if "mmtag" in model_name.lower():
                template_name = "v0_mmtag"
            elif "plain" in model_name.lower() and "finetune" not in model_name.lower():
                template_name = "v0_mmtag"
            else:
                template_name = "llava_v0"
    elif "mpt" in model_name:
        template_name = "mpt_text"
    elif "llama-2" in model_name:
        template_name = "llama_2"
    else:
        template_name = "vicuna_v1"

    return conv_templates[template_name].copy()


def load_image_from_url(image_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    if image_url.startswith("http"):
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()

        # Open the image using PIL
        img = Image.open(BytesIO(response.content))
    elif image_url.startswith("data:image"):
        img = load_image_from_base64(image_url.split(',')[1])

    return img


def safe_append(history, text, image, image_process_mode="Default"):
    if len(text) <= 0 and image is None:
        history.skip_next = True
        return history

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            text = text + "\n<image>"
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            history = default_conversation.copy()
    history.append_message(history.roles[0], text)
    history.skip_next = False

    return history

def concat_images(image_list, dimension=ConcatOptions.Horizontal):

    widths, heights = zip(*(img.size for img in image_list))

    max_height = max(heights)
    max_width = max(widths)

    image_list = [img.resize((max_width, max_height)) for img in image_list]

    # Get the dimensions of the new images
    widths, heights = zip(*(img.size for img in image_list))

    # Calculate the total width and height for the concatenated image
    total_width = sum(widths)
    total_height = sum(heights)

    # Create a new blank image with the required width and height
    concat_image = Image.new('RGB', (total_width, total_height))

    # Paste the images onto the concatenated image
    x_offset = 0
    y_offset = 0
    for i, img in enumerate(image_list):
        # Create a draw object to write text on the concatenated image
        draw = ImageDraw.Draw(img)
        # Write the big text image inside the current image
        text = f"{i+1}"  # Text to be written
        text_position = (10, 10)  # Center the text
        font = ImageFont.load_default(size=20)
        draw.text(text_position, text, fill=(255, 0, 0), font=font)  # Write the text in black color

        if dimension == ConcatOptions.Horizontal:
            concat_image.paste(img, (x_offset, 0))
            x_offset += img.width
        elif dimension == ConcatOptions.Vertical:
            concat_image.paste(img, (0, y_offset))
            y_offset += img.height

    return concat_image