"""ProcessMind AI — точка входа Streamlit-приложения (страница «Главная»)."""

import streamlit as st

from processmind.analysis.report import DATA_NOTICE
from processmind.config import get_settings
from processmind.ui.style import apply_style, nav_link, notice, stepper

st.set_page_config(page_title="ProcessMind AI", page_icon="🧠", layout="wide")
apply_style()

st.markdown(
    '<div class="pm-hero"><h1>🧠 ProcessMind AI</h1>'
    "<p>Загрузите описание бизнес-процесса — получите схему AS-IS, метрики, обоснованные рекомендации по автоматизации "
    "и модельную оценку целевого процесса TO-BE.</p></div>",
    unsafe_allow_html=True,
)
stepper()

left, right = st.columns([3, 2])
with left:
    st.markdown("#### Как это работает")
    st.markdown(
        "1. **Документ** — загрузите PDF или PNG (схема или регламент).\n"
        "2. **Операции** — система извлечёт процесс в структуру ProcessSpec; проверьте и поправьте операции.\n"
        "3. **AI-анализ** — метрики считает MCP-сервер, паттерны ищет RAG, рекомендации формирует LLM по методике Skill.\n"
        "4. **Выбор** — вы решаете, какие рекомендации включить в TO-BE, и задаёте допущения о длительности.\n"
        "5. **Отчёт** — сравнение AS-IS и TO-BE, схемы процесса и отчёт в Markdown."
    )
    nav_link("pages/1_Распознавание_процесса.py", "Начать: загрузить документ →", "📄")
    nav_link("pages/2_Анализ_процесса.py", "Перейти к AI-анализу →", "🔍")
with right:
    st.markdown("#### Что важно знать")
    notice(DATA_NOTICE)
    st.markdown(
        "- Значения, которых нет в документе (участник, длительность, тип выполнения), **не выдумываются**.\n"
        "- Показатели TO-BE — **модельная оценка** по вашим допущениям, а не прогноз.\n"
        "- Рекомендации проходят программную проверку и ваше подтверждение."
    )

settings = get_settings()
with st.expander("Статус конфигурации"):
    st.json(
        {
            "OPENAI_API_KEY задан": bool(settings.openai_api_key),
            "OPENAI_MODEL": settings.openai_model,
            "EMBEDDING_MODEL": settings.embedding_model,
            "CHROMA_PERSIST_DIR": settings.chroma_persist_dir,
        }
    )
