import streamlit as st

# Функция для добавления CSS стилей
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
    uploaded_files = st.file_uploader("Выберите файлы (PDF, DOCX, изображения)", accept_multiple_files=True)

    # Проверка, загружены ли файлы
    if uploaded_files:
        st.write("Вы загрузили следующие файлы:")
        for file in uploaded_files:
            st.write(f"- {file.name}")
    else:
        st.write("Пожалуйста, загрузите файлы для индексации.")

    # Поле для ввода запроса
    st.header("Введите запрос для мультимодального поиска")
    query = st.text_input("Ваш запрос")

    # Если введен запрос
    if query:
        st.write(f"Вы ищете информацию по запросу: **{query}**")
        # Здесь можно подключить модель и выполнить поиск по данным, но пока это заглушка
        st.write("Результаты поиска будут отображены здесь.")
    
    # Если запрос не введен
    else:
        st.write("Введите запрос, чтобы начать поиск.")

# Запуск приложения
if __name__ == "__main__":
    main()
