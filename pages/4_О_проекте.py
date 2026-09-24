"""Страница «О проекте»: описание архитектуры, стека и ограничений MVP."""

import streamlit as st

from processmind.ui.style import apply_style, page_header

st.set_page_config(page_title="О проекте — ProcessMind AI", page_icon="ℹ️", layout="wide")
apply_style()
page_header("О проекте", "Архитектура, стек и ограничения MVP.")

st.markdown(
    """
### ProcessMind AI

Финальный проект курса по LLM Engineering.

**Цель:** интеллектуальный анализ бизнес-процессов и генерация целевой (TO-BE) модели
на основе описания, загруженного пользователем.

### Технологический стек
- Python 3.11+, Streamlit — пользовательский интерфейс
- LangGraph — оркестрация пайплайна анализа
- OpenAI GPT-4o-mini (Structured Outputs + Vision) — извлечение процесса из PDF и изображений
- ChromaDB + OpenAI Embeddings (text-embedding-3-small) — RAG по базе знаний
- Python MCP SDK / FastMCP — собственный MCP-сервер с инструментами ProcessMind
- PyMuPDF — парсинг PDF-документов
- Graphviz — визуализация AS-IS / TO-BE
- LangSmith — трассировка и оценка (следующая итерация)
- Pytest — тесты

### Архитектура (модули)
- `processmind.models` — единая модель процесса **ProcessSpec** (Pydantic)
- `processmind.parsing` — Document Parser (PyMuPDF), извлечение из текста и Vision Parser (GPT-4o-mini), построение и валидация ProcessSpec
- `processmind.workflow` — главный граф LangGraph (validate_process → calculate_metrics → retrieve_automation_patterns → generate_recommendations → validate_recommendations → human_approval → simulate_automation → generate_report)
- `processmind.analysis` — Skill process-analysis (методика → промпт LLM), рекомендации и их валидация, AnalysisReport
- `processmind.rag` — Automation Knowledge Base: паттерны (Markdown), эмбеддинги, ChromaDB, поиск Top-3
- `processmind.mcp` — собственный MCP-сервер (validate_process, calculate_process_metrics, simulate_automation и др.) и MCP-клиент (stdio) для LangGraph
- `processmind.visualization` — построение AS-IS/TO-BE диаграмм (Graphviz)
- `processmind.evaluation` — Evaluation Pipeline (структурные проверки, заготовка под LangSmith)

### Ограничения текущего MVP
- Извлечение из PDF/PNG/JPG требует OPENAI_API_KEY; PDF без текстового слоя (сканы) не
  поддерживаются — загрузите их как изображение.
- Анализ работает только над подтверждённым процессом со страницы «Распознавание процесса».
- Качество рекомендаций зависит от модели (GPT-4o-mini); результат всегда проходит программную валидацию и подтверждение пользователя.
- Все оценки эффекта автоматизации — модельные, на основе заданных предположений.
- Индексация и поиск по базе знаний требуют OPENAI_API_KEY (офлайн-подстановки эмбеддингов нет); база знаний — 14 синтетических паттернов.
- Авторизация, внешние интеграции и сложные хранилища данных не предусмотрены.
- Все данные для демонстрации — синтетические.
"""
)
