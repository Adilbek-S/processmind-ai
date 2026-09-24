"""Страница «База знаний»: добавление и поиск документов в RAG (ChromaDB)."""

import streamlit as st

from processmind.rag.engine import RAGEngine

st.set_page_config(page_title="База знаний — ProcessMind AI", page_icon="📚", layout="wide")
st.title("📚 База знаний (RAG)")

st.markdown(
    "Здесь можно добавить синтетические документы о бизнес-процессах в векторную базу "
    "(ChromaDB) и выполнить по ним семантический поиск. Без ключа OPENAI_API_KEY "
    "используется офлайн-эмбеддинг (для демонстрации архитектуры RAG)."
)

engine = RAGEngine()

with st.form("add_doc_form", clear_on_submit=True):
    doc_text = st.text_area("Текст документа для добавления в базу знаний", height=150)
    submitted = st.form_submit_button("Добавить в базу знаний")
    if submitted and doc_text.strip():
        ids = engine.add_documents([doc_text])
        st.success(f"Документ добавлен, id: {ids[0]}")

st.divider()

query = st.text_input("Поисковый запрос")
if st.button("Искать") and query.strip():
    results = engine.query(query)
    if results:
        for i, doc in enumerate(results, start=1):
            st.markdown(f"**{i}.** {doc}")
    else:
        st.info("По запросу ничего не найдено. Добавьте документы в базу знаний.")
