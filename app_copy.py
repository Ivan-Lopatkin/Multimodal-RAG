import os
from dotenv import load_dotenv
import streamlit as st
from src.llm import chat
from src.retrievers import RetrievePipeline
from src.utils import pdf_to_images
import torch

SAVE_DIR = "data/pdf_files"
os.makedirs(SAVE_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

system_message = "Ты виртуальный ассистент, твоя задача отвечать на вопросы пользователей и быть дружелюбным."

load_dotenv()
retrieve_pipe = RetrievePipeline(device=device)

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

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "answered" not in st.session_state:
    st.session_state["answered"] = True

def main():

    st.title("Мультимодальный RAG - Интерфейс")

    # Sidebar для загрузки файлов и выбора стратегии
    with st.sidebar.form("Загрузка файлов", clear_on_submit=True):
        uploaded_files = st.sidebar.file_uploader(
            "Выберите PDF-файлы", accept_multiple_files=True
        )

        submitted = st.form_submit_button("Рассчитать эмбеддинги загруженных файлов")

        if submitted and uploaded_files:
            st.sidebar.write("Вы загрузили следующие файлы:")
            for file in uploaded_files:
                st.sidebar.write(f"- {file.name}")
                
                with open(SAVE_DIR + '/' + file.name, "wb") as f:
                    f.write(file.getbuffer())

                
                pdf_to_images(SAVE_DIR + '/' + file.name)
                retrieve_pipe.add_to_index(SAVE_DIR + '/' + file.name)
                

                    
            # st.sidebar.success(f"Файлы сохранены в папку {SAVE_DIR}.")
        else:
            st.sidebar.write("Пожалуйста, загрузите файлы для индексации.")

    # Меню выбора стратегии
    st.sidebar.header("Выбор стратегии")
    strategy = st.sidebar.selectbox(
        "Выберите стратегию поиска:",
        ["SummaryEmb", "ColQwen", "ColQwen+SummaryEmb"]
    )

    # Отображение чата

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == 'user':
                st.markdown(message["content"])
            else:
                st.markdown('Релевантные изображения по запросу:')
                for image in message["content"][1]:
                    st.image(image, caption="Релевантное изображение", use_container_width=True)
                st.markdown('Ответ:\n' + message["content"][0])

    # React to user input
    if (query := st.chat_input("Введите запрос для мультимодального поиска")) and st.session_state["answered"]:
        
        st.session_state["answered"] = False

        # Display user message in chat message container
        st.chat_message("user").markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        images = retrieve_pipe.retrieve(
                query,
                strategy=strategy)
        
        structured_query = [
                {"role": "user", "content": [
                    {"type": "text",
                     "text": query}
                ]}]
        

        answer = chat(structured_query, images)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown('Релевантные изображения по запросу:')
            for image in images:
                st.image(image, caption="Релевантное изображение", use_container_width=True)
            st.markdown('Ответ:\n' + answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": (answer, images)})
        st.session_state["answered"] = True


if __name__ == "__main__":
    main()