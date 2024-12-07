from typing import List, Dict
import os
from mistralai import Mistral
from utils import encode_image

with open("system_message.txt", "r") as f:
    system_message = f.read()

model = os.getenv("MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)


def chat(chat_history: List[Dict], image_path: str):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        }
    ]
    messages += chat_history
    if image_path:
        base64_image = encode_image(image_path=image_path)
        messages[-1]["content"].append({"type": "image_url",
                                       "image_url": f"data:image/jpeg;base64,{base64_image}"})
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content
