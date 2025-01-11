import base64
from pdf2image import convert_from_path
import tempfile
import os
from mistralai import Mistral


model = os.getenv("MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)
summary_prompt = '''
                        Проанализируй изображение и извлеки текстовое описание, максимально полезное для систем Retrieval-Augmented Generation (RAG). Описание должно быть представлено в связной текстовой форме и содержать:
                        
                        1. Все текстовые данные, извлеченные из изображения, с учетом их структуры и контекста (например, заголовки, абзацы, примечания, подписи к элементам).
                        2. Информацию о графическом содержании (графики, таблицы, изображения) в виде краткого описания, где это возможно. Опиши ключевые данные, тренды или характеристики, которые можно понять из визуальных элементов.
                        3. Контекст изображения: если изображение похоже на часть документа (например, отчет, статья, техническая документация), опиши общий смысл содержимого.
                        4. Любую дополнительную информацию, которая может быть полезна для поиска (например, упоминания ключевых терминов, даты, цифры, имена или названия).
                        
                        Ответ должен быть максимально информативным и релевантным для поиска, при этом не теряя связности и понятности. Строго придерживайся примера вывода.
                        Не пиши конкретные цифры, показатели, ответь лишь на один вопрос "что изображено на картинке?". коротко и ясно.
                        Пример вывода:
                        "На изображении представлена страница документа с заголовком 'Отчет о продажах за 2024 год'. Текст включает описание квартальных результатов, где основной акцент сделан на рост выручки на 15% в Q2. Присутствует линейный график, показывающий динамику продаж по месяцам, с заметным скачком в апреле. Также есть таблица с детализацией по регионам, где выделяются три основных региона: Северная Америка, Европа и Азия. На изображении также содержится примечание о планах на следующий квартал."
                        '''

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def pdf_to_images(pdf_path):
    pdf_name = pdf_path.split('/')[-1].replace('.pdf', '')
    out_folder = 'data/images/' + pdf_name
    os.makedirs(out_folder, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=100)
    
    for i, page in enumerate(images, start=1):
            output_path = os.path.join(out_folder, f"{pdf_name}_page{i}.jpg")
            page.save(output_path, "JPEG")

def summarize_image(image_path, summary_prompt=summary_prompt):
    """
    Отправляет изображение в модель Pixtral-12B для суммаризации.
    """
    base64_image = encode_image(image_path)
    messages = [
        {
            "role": "system",
            "content": summary_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summary"},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
            ]
        }
    ]

    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content