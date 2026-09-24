"""ProcessMind AI — точка входа Streamlit-приложения (страница «Главная»)."""

import streamlit as st

from processmind.config import get_settings

st.set_page_config(page_title="ProcessMind AI", page_icon="🧠", layout="wide")

st.title("🧠 ProcessMind AI")
st.subheader("Интеллектуальный анализ и автоматизация бизнес-процессов")

st.markdown(
    """
ProcessMind AI — учебный MVP-проект финального курса по LLM Engineering.

Пользователь загружает описание бизнес-процесса (текст, PDF или изображение схемы),
а система:

1. извлекает шаги процесса в единую структурированную модель **ProcessSpec**;
2. анализирует процесс и находит возможности автоматизации;
3. формирует черновик целевой модели **TO-BE**;
4. визуализирует AS-IS / TO-BE в виде диаграмм.

⚠️ **Статус MVP:** распознавание процесса из PDF/PNG/JPG работает на реальных вызовах OpenAI
(нужен `OPENAI_API_KEY`); анализ, TO-BE и оценка пока на детерминированных правилах.
"""
)

st.markdown("### Навигация")
st.markdown(
    "- **Распознавание процесса** — загрузка PDF/PNG/JPG, правка и подтверждение ProcessSpec
- **Анализ процесса** — запуск пайплайна анализа над подтверждённым процессом\n"
    "- **База знаний** — семантический поиск по загруженным документам (RAG)\n"
    "- **О проекте** — архитектура, стек и ограничения MVP"
)

with st.expander("Статус конфигурации"):
    settings = get_settings()
    st.json(
        {
            "OPENAI_API_KEY задан": bool(settings.openai_api_key),
            "OPENAI_MODEL": settings.openai_model,
            "EMBEDDING_MODEL": settings.embedding_model,
            "CHROMA_PERSIST_DIR": settings.chroma_persist_dir,
            "LANGCHAIN_PROJECT": settings.langchain_project,
        }
    )
