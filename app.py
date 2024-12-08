import os
from dotenv import load_dotenv
import streamlit as st
from src.llm import chat
from src.retrievers import RetrievePipeline
system_message = "Ты виртуальный ассистент, твоя задача отвечать на вопросы пользователей и быть дружелюбным."

load_dotenv()
retrieve_pipe = RetrievePipeline()

chat_css = """
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.user-message {
    background-color: #e8f5e9; /* Светло-зеленый фон */
    color: #1b5e20; /* Темно-зеленый текст */
    align-self: flex-start;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    max-width: 60%;
    font-size: 16px;
}
.bot-message {
    background-color: #e3f2fd; /* Светло-голубой фон */
    color: #0d47a1; /* Темно-синий текст */
    align-self: flex-end;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    max-width: 60%;
    font-size: 16px;
    font-style: italic;
}
</style>
"""


st.markdown(chat_css, unsafe_allow_html=True)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if st.button("Очистить контекст"):
    st.session_state["chat_history"] = []


def main():

    st.title("Мультимодальный RAG - Интерфейс")

    st.sidebar.header("Мультимодальный RAG")
    st.sidebar.text("Загрузите файлы и введите запрос для поиска.")

    st.header("Загрузите файлы для индексации")
    uploaded_files = st.file_uploader(
        "Выберите файлы (PDF, DOCX, изображения)", accept_multiple_files=True)

    if uploaded_files:
        st.write("Вы загрузили следующие файлы:")
        for file in uploaded_files:
            st.write(f"- {file.name}")
    else:
        st.write("Пожалуйста, загрузите файлы для индексации.")
    
    st.header("Введите запрос для мультимодального поиска")
    query = st.text_input("Ваш вопрос")
    if st.button("Отправить"):
        if query:
            st.session_state["chat_history"].append(
                {"role": "user", "content": [
                    {"type": "text",
                     "text": query}
                ]})
            images = retrieve_pipe.retrieve(
                query,
                strategy="Intersection")
            answer = chat(st.session_state["chat_history"], images)
            st.session_state['chat_history'].append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer}
                    ]
                }
            )

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">{msg["content"][0]["text"]}</div >', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="bot-message">{msg["content"][0]["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
