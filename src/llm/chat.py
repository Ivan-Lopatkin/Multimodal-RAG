import os
from typing import Dict, List

from dotenv import load_dotenv
from mistralai import Mistral

from src.utils import encode_image, load_prompts_from_yaml

load_dotenv()

prompts = load_prompts_from_yaml(os.getenv("PROMPTS_PATH"))
model = os.getenv("MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)


def chat(chat_history: List[Dict], images: List[str] = None):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompts["system"]}]}
    ]
    messages += chat_history
    if images:
        for image_path in images:
            base64_image = encode_image(image_path=image_path)
            messages[-1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                }
            )
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content


def summarize_image(image_path: str):
    """
    Отправляет изображение в модель Pixtral-12B для суммаризации.
    """
    base64_image = encode_image(image_path)
    messages = [
        {"role": "system", "content": prompts["summary"]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summary"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]

    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content
