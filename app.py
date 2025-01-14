import os

import streamlit as st
from dotenv import load_dotenv

from src.llm import chat
from src.retrievers import RetrievePipeline
from src.utils import pdf_to_images

SAVE_DIR = "data/pdf_files"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "mps"

load_dotenv()


@st.cache_resource
def init_retrieve_pipeline(device: str) -> RetrievePipeline:
    return RetrievePipeline(device=device)


retrieve_pipe = init_retrieve_pipeline(device)


def initialize_session_states() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "answered" not in st.session_state:
        st.session_state["answered"] = True


def sidebar_file_uploader() -> None:
    st.sidebar.header("Загрузка файлов")

    with st.sidebar.form("Загрузка файлов", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Выберите PDF-файлы", accept_multiple_files=True
        )
        submitted = st.form_submit_button("Рассчитать эмбеддинги")

        if submitted and uploaded_files:
            for file in uploaded_files:
                pdf_path = os.path.join(SAVE_DIR, file.name)
                with open(pdf_path, "wb") as f:
                    f.write(file.getbuffer())

                pdf_to_images(pdf_path)

                retrieve_pipe.add_to_index(pdf_path)


def sidebar_strategy_selector() -> str:
    st.sidebar.header("Выбор стратегии")
    return st.sidebar.selectbox(
        "Выберите стратегию поиска:", ["SummaryEmb", "ColQwen", "ColQwen+SummaryEmb"]
    )


def display_chat_history() -> None:
    for message in st.session_state["messages"]:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                answer_text, image_paths = content
                st.markdown("Релевантные изображения по запросу:")
                for path in image_paths:
                    st.image(
                        path,
                        caption="Релевантное изображение",
                        use_container_width=True,
                    )
                st.markdown("Ответ:\n" + answer_text)


def handle_user_query(query: str, strategy: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    image_paths = retrieve_pipe.retrieve(query, strategy)

    structured_query = [{"role": "user", "content": [{"type": "text", "text": query}]}]

    answer_text = chat(structured_query, image_paths)

    st.session_state["messages"].append(
        {"role": "assistant", "content": (answer_text, image_paths)}
    )
    with st.chat_message("assistant"):
        st.markdown("Релевантные изображения по запросу:")
        for path in image_paths:
            st.image(path, caption="Релевантное изображение", use_container_width=True)
        st.markdown("Ответ:\n" + answer_text)


def main():
    st.title("Мультимодальный RAG - Интерфейс")

    initialize_session_states()

    sidebar_file_uploader()
    strategy = sidebar_strategy_selector()

    display_chat_history()

    user_query = st.chat_input("Введите запрос для мультимодального поиска")

    if user_query and st.session_state["answered"]:
        st.session_state["answered"] = False
        handle_user_query(user_query, strategy)
        st.session_state["answered"] = True


if __name__ == "__main__":
    main()
