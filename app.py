from typing import List, Dict
import streamlit as st
from mistralai import Mistral
model = "pixtral-12b-2409"
system_message = "Ты виртуальный ассистент, твоя задача отвечать на вопросы пользователей и быть дружелюбным."


chat_css = """
<style>
.chat-container {
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #DCF8C6;
    align-self: flex-start;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    max-width: 60%;
}
.bot-message {
    background-color: #F1F0F0;
    align-self: flex-end;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    max-width: 60%;
}
</style>
"""

st.markdown(chat_css, unsafe_allow_html=True)

api_key = st.secrets['MISTRAL_API_KEY']

client = Mistral(api_key=api_key)


def chat(chat_history: List[Dict]) -> str:
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
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    st.session_state["chat_history"].append(
        {"role": "ai", "content": [
            {"type": "text",
             "text": chat_response.choices[0].message.content}
        ]})


def main():
    # Добавляем фон

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
    if st.button("Отправить"):
        if query:
            st.session_state["chat_history"].append(
                {"role": "user", "content": [
                    {"type": "text",
                     "text": query}
                ]})
            chat(st.session_state["chat_history"])

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">{msg["content"]["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="bot-message">{msg["content"]["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Запуск приложения
if __name__ == "__main__":
    main()
