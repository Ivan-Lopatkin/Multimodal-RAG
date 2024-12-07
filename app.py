import os
import logging
from dotenv import load_dotenv
import streamlit as st
from src.llm import chat
from src.retrievers import RetrievePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
    background-color: #e8f5e9; 
    color: #1b5e20; 
    align-self: flex-start;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    max-width: 60%;
    font-size: 16px;
}
.bot-message {
    background-color: #e3f2fd; 
    color: #0d47a1; 
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


def initialize_session():
    """Инициализация сессии и очистка контекста при необходимости."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        logger.info("Инициализация состояния чата.")

    if st.button("Очистить контекст"):
        st.session_state["chat_history"] = []
        logger.info("Контекст чата был очищен пользователем.")


def display_upload_section():
    """Отображение секции загрузки файлов."""
    st.header("Загрузите файлы для индексации")
    uploaded_files = st.file_uploader(
        "Выберите файлы (PDF, DOCX, изображения)", accept_multiple_files=True
    )

    if uploaded_files:
        st.write("Вы загрузили следующие файлы:")
        for file in uploaded_files:
            st.write(f"- {file.name}")
        logger.info(f"Пользователь загрузил файлы: {[f.name for f in uploaded_files]}")
    else:
        st.write("Пожалуйста, загрузите файлы для индексации.")
        logger.info("Файлы не были загружены.")

    return uploaded_files


def display_query_section():
    """Отображение секции ввода запроса и обработка ответа."""
    st.header("Введите запрос для мультимодального поиска")
    query = st.text_input("Ваш вопрос")
    if st.button("Отправить"):
        logger.info(f"Пользователь отправил запрос: {query}")
        if query:
            # Добавление запроса в историю
            st.session_state["chat_history"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            )
            # Получение связанных изображений
            images = retrieve_pipe.retrieve(query, strategy="Intersection")
            logger.info(f"Получены изображения для запроса: {query} - {images}")

            # Получение ответа от модели
            answer = chat(st.session_state["chat_history"], images)
            logger.info(f"Ответ от модели для запроса '{query}': {answer}")

            # Добавление ответа в историю
            st.session_state['chat_history'].append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer
                        }
                    ]
                }
            )
        else:
            logger.warning("Пользователь отправил пустой запрос.")


def display_chat_history():
    """Отображение истории чата."""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">{msg["content"][0]["text"]}</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-message">{msg["content"][0]["text"]}</div>', 
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.title("Мультимодальный RAG - Интерфейс")
    st.sidebar.header("Мультимодальный RAG")
    st.sidebar.text("Загрузите файлы и введите запрос для поиска.")

    initialize_session()
    uploaded_files = display_upload_section()
    display_query_section()
    display_chat_history()


if __name__ == "__main__":
    logger.info("Запуск приложения.")
    main()
