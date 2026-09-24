"""Страница «База знаний»: поиск применимых способов автоматизации (RAG над ChromaDB)."""

import streamlit as st

from processmind.config import get_settings
from processmind.rag.errors import KnowledgeBaseError
from processmind.ui.style import apply_style, page_header
from processmind.rag.knowledge_base import (
    build_knowledge_base,
    build_query_from_process,
    get_pattern_by_id,
    list_indexed_patterns,
    search_automation_patterns,
)

RESULTS_KEY = "kb_results"
SHOWN_CHUNKS = 2  # сколько лучших разделов показывать как объяснение релевантности

st.set_page_config(page_title="База знаний — ProcessMind AI", page_icon="📚", layout="wide")
apply_style()
page_header(
    "База знаний: паттерны автоматизации",
    "Семантический поиск (RAG) по синтетической базе: эмбеддинги OpenAI text-embedding-3-small + ChromaDB. "
    "Запрос сравнивается по смыслу с разделами паттернов, а не по совпадению слов.",
)

settings = get_settings()
if not settings.openai_api_key:
    st.warning("OPENAI_API_KEY не задан в .env — индексация и поиск недоступны (просмотр уже построенного индекса работает).")

# ---------- индекс ----------
col_status, col_button = st.columns([3, 1])
build_clicked = col_button.button("Построить / обновить базу знаний", disabled=not settings.openai_api_key)

build_message = None  # (тип, текст) — показывается под статусом
if build_clicked:
    try:
        with st.spinner("Индексирую паттерны…"):
            report = build_knowledge_base()
    except KnowledgeBaseError as exc:
        build_message = ("error", f"Не удалось построить базу знаний: {exc}")
    else:
        build_message = (
            "success",
            f"Готово: паттернов {report.total_patterns}, фрагментов {report.total_chunks}. "
            f"Добавлено {len(report.added)}, обновлено {len(report.updated)}, без изменений {len(report.unchanged)}, "
            f"удалено {len(report.removed)}. Отправлено на эмбеддинг: {report.embedded_chunks} фрагментов.",
        )

indexed = list_indexed_patterns()  # после сборки: статус отражает актуальный индекс
if indexed:
    col_status.success(f"В индексе паттернов: {len(indexed)}")
else:
    col_status.info("Индекс пуст. Нажмите «Построить / обновить базу знаний».")
if build_message:
    getattr(st, build_message[0])(build_message[1])

if indexed:
    with st.expander(f"Паттерны в базе ({len(indexed)})"):
        for item in indexed:
            st.markdown(f"- **{item['pattern_id']}** — {item['title']} · `{item['source']}`")

st.divider()

# ---------- запрос ----------
confirmed = st.session_state.get("confirmed_process_spec")
modes = ["Свободный запрос"] + (["Из подтверждённого процесса"] if confirmed is not None else [])
mode = st.radio("Источник запроса", modes, horizontal=True)

default_query, key_suffix = "", "free"
if mode == "Из подтверждённого процесса":
    options = {"Весь процесс": None} | {f"{i}. {s.name}": s.id for i, s in enumerate(confirmed.steps, start=1)}
    label = st.selectbox("Что искать", list(options))
    default_query = build_query_from_process(confirmed, options[label])
    key_suffix = f"{confirmed.process_id}_{options[label]}"

query = st.text_area(
    "Запрос (можно отредактировать)",
    value=default_query,
    placeholder="Например: сотрудники вручную перепечатывают данные из сканов договоров в учётную систему",
    key=f"kb_query_{key_suffix}",
    height=110,
)

if st.button("Найти способы автоматизации", type="primary", disabled=not query.strip()):
    try:
        with st.spinner("Ищу…"):
            st.session_state[RESULTS_KEY] = (query, search_automation_patterns(query, top_k=3))
    except KnowledgeBaseError as exc:
        st.session_state.pop(RESULTS_KEY, None)
        st.error(str(exc))

# ---------- результаты ----------
if RESULTS_KEY in st.session_state:
    searched_query, matches = st.session_state[RESULTS_KEY]
    st.markdown(f"#### Top-{len(matches)} для запроса: «{searched_query}»")
    st.caption(
        "Сходство — косинусная близость эмбеддингов запроса и раздела паттерна. Шкала не калибрована: "
        "сравнивайте значения между собой и читайте найденные фрагменты."
    )
    for rank, match in enumerate(matches, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {match.pattern_id} — {match.title}**  ·  сходство {match.score:.3f}")
            st.caption(f"Документ: knowledge_base/automation_patterns/{match.source}")

            st.markdown("**Почему релевантен** — наиболее близкие разделы:")
            for hit in match.chunks[:SHOWN_CHUNKS]:
                st.markdown(f"- **{hit.section}** (сходство {hit.score:.3f})")
                st.markdown(f"> {hit.text}")

            with st.expander("Все разделы паттерна и их сходство с запросом"):
                for hit in match.chunks:
                    st.markdown(f"**{hit.section}** — {hit.score:.3f}  ·  `{hit.chunk_id}`")
            with st.expander("Полное описание паттерна"):
                pattern = get_pattern_by_id(match.pattern_id)
                if pattern is None:
                    st.info("Паттерн не найден в индексе.")
                else:
                    for section, body in pattern.sections.items():
                        st.markdown(f"**{section}**")
                        st.markdown(body)
