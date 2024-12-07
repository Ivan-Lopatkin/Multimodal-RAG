import streamlit as st
from mistralai import Mistral
model = "pixtral-12b-2409"
system_message = "Ты виртуальный ассистент, твоя задача отвечать на вопросы пользователей и быть дружелюбным."


def add_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://your-image-url.com/background.jpg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


api_key = st.secrets['MISTRAL_API_KEY']

client = Mistral(api_key=api_key)


def chat(user_message: str):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content


def main():
    # Добавляем фон
    add_background()

    # Заголовок страницы
    st.title("Мультимодальный RAG - Интерфейс")

    # Сайдбар с описанием
    st.sidebar.header("Мультимодальный RAG")
    st.sidebar.text("Загрузите файлы и введите запрос для поиска.")

    # Загрузка файлов
    st.header("Загрузите файлы для индексации")
    uploaded_files = st.file_uploader(
        "Выберите файлы (PDF, DOCX, изображения)", accept_multiple_files=True)

    # Проверка, загружены ли файлы
    if uploaded_files:
        st.write("Вы загрузили следующие файлы:")
        for file in uploaded_files:
            st.write(f"- {file.name}")
    else:
        st.write("Пожалуйста, загрузите файлы для индексации.")

    # Поле для ввода запроса
    st.header("Введите запрос для мультимодального поиска")
    query = st.text_input("Ваш вопрос")
    answer = chat(query)
    st.write(answer)


# Запуск приложения
if __name__ == "__main__":
    main()
