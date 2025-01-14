import base64
import os

import yaml
from mistralai import Mistral
from pdf2image import convert_from_path

model = os.getenv("MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)


def encode_image(image_path) -> str | None:
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def pdf_to_images(pdf_path: str) -> None:
    pdf_name = pdf_path.split("/")[-1].replace(".pdf", "")
    out_folder = "data/images/" + pdf_name
    os.makedirs(out_folder, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=100)

    for i, page in enumerate(images, start=1):
        output_path = os.path.join(out_folder, f"{pdf_name}_page{i}.jpg")
        page.save(output_path, "JPEG")


def load_prompts_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
