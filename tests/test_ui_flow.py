"""Сквозной тест страницы «AI-анализ» через Streamlit AppTest.

Страница работает по-настоящему: граф LangGraph, checkpointer с паузой, MCP-сервер в отдельном процессе.
Подменены только внешние модели (LLM, эмбеддинги) и запуск MCP-процесса приложением (используется тот же
клиент, что и в остальных тестах). Проверяется реальная последовательность действий пользователя:
запуск -> метрики и рекомендации -> выбор -> допущение -> подтверждение -> TO-BE и отчёт.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from processmind.analysis.recommendations import LLMRecommendationSet
from processmind.analysis.report import DATA_NOTICE, MODEL_NOTICE
from processmind.models import ProcessSpec
from processmind.ui.state import CONFIRMED_KEY, RUN_KEY
from tests.helpers_workflow import PROCESS, FakeLLMClient, context_of, rec
from tests.test_analysis_workflow import Harness, kb, mcp_client  # noqa: F401 (фикстуры)

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "2_Анализ_процесса.py")


def make_page(monkeypatch, harness, process=PROCESS) -> AppTest:
    monkeypatch.setattr("processmind.workflow.graph.create_default_deps", lambda: None)
    monkeypatch.setattr("processmind.workflow.graph.build_workflow", lambda deps: harness.graph)
    st.cache_resource.clear()
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state[CONFIRMED_KEY] = ProcessSpec.model_validate(process)
    return at.run()


def html_of(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def checkbox_for(at: AppTest, rec_id: str):
    return next(c for c in at.checkbox if c.key.endswith(f"_{rec_id}"))


def duration_input(at: AppTest, rec_id: str):
    return next(n for n in at.number_input if n.key and n.key.endswith(f"_{rec_id}") and n.key.startswith("dur_"))


@pytest.fixture
def harness(mcp_client, kb):  # noqa: F811
    return Harness(mcp_client, kb)


def test_page_without_confirmed_process_guides_the_user():
    st.cache_resource.clear()
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception and any("Сначала распознайте" in i.value for i in at.info)
    assert not at.button  # запускать нечего


def test_full_user_flow_from_start_to_report(monkeypatch, harness):
    at = make_page(monkeypatch, harness)
    assert not at.exception
    assert "pm-stepper" in html_of(at) and "Этап 3" in html_of(at)  # подтверждённый процесс -> этап AI-анализа
    at.number_input[0].set_value(40).run()

    # 4-5. запуск AI-анализа, просмотр исходных метрик и рекомендаций
    button(at, "Запустить AI-анализ").click().run()
    assert not at.exception
    page = html_of(at)
    assert "Исходные метрики (AS-IS)" in page and "pm-metric" in page and "Рекомендации по автоматизации" in page
    for column in ("Операция", "Проблема", "Предложение", "Паттерн автоматизации", "Ожидаемый эффект"):
        assert f"<th>{column}</th>" in page
    assert any("приостановлен" in i.value for i in at.info)  # честная пауза workflow
    assert harness.mcp.calls == ["validate_process", "calculate_process_metrics"]  # симуляции ещё нет

    # 6-7. выбор рекомендаций; у R3 (операция без длительности) выбор недоступен
    assert checkbox_for(at, "R3").disabled and not checkbox_for(at, "R1").disabled
    checkbox_for(at, "R1").check().run()
    checkbox_for(at, "R2").check().run()
    duration_input(at, "R2").set_value(30).run()  # допущение пользователя

    # 8. формирование TO-BE
    button(at, "Сформировать TO-BE и отчёт").click().run()
    assert not at.exception
    assert harness.mcp.calls == ["validate_process", "calculate_process_metrics", "simulate_automation", "calculate_process_metrics"]

    # 9. итоговый отчёт: сравнение, схемы, таблица рекомендаций, скачивание
    page = html_of(at)
    assert "Сравнение AS-IS и TO-BE" in page and "TO-BE (модельная оценка)" in page
    for label in ("Общее время процесса", "Ручных операций", "Доля ручных операций", "Условная месячная трудоёмкость"):
        assert label in page
    assert MODEL_NOTICE in page and DATA_NOTICE in page  # оговорки видны и в отчёте, и рядом с показателями
    assert "включено в TO-BE" in page and "<th>Решение</th>" in page
    assert len(at.get("graphviz_chart")) == 2  # AS-IS и TO-BE рядом
    downloads = [d for d in at.get("download_button")]
    assert {d.proto.label for d in downloads} >= {"Скачать отчёт (Markdown)"}
    assert at.session_state[RUN_KEY].status == "finished" and at.session_state[RUN_KEY].report.status == "completed"


def test_missing_duration_assumption_blocks_confirmation(monkeypatch, mcp_client, kb):  # noqa: F811
    def responder(messages):  # как реальная gpt-4o-mini: рекомендации есть, длительности после автоматизации нет
        pattern = context_of(messages)["knowledge_base_patterns"][0]["pattern_id"]
        return LLMRecommendationSet(recommendations=[rec("s1", pattern)], notes=[])

    harness = Harness(mcp_client, kb, llm=FakeLLMClient(responder))
    at = make_page(monkeypatch, harness)
    button(at, "Запустить AI-анализ").click().run()

    checkbox_for(at, "R1").check().run()
    assert any("Укажите допущение" in w.value for w in at.warning)
    assert button(at, "Сформировать TO-BE и отчёт").disabled  # процент сокращения не подставляется — нужен ввод пользователя

    duration_input(at, "R1").set_value(5).run()
    assert not button(at, "Сформировать TO-BE и отчёт").disabled
    button(at, "Сформировать TO-BE и отчёт").click().run()
    assert at.session_state[RUN_KEY].report.approved[0].duration_source == "user"


def test_selecting_nothing_finishes_without_tobe(monkeypatch, harness):
    at = make_page(monkeypatch, harness)
    button(at, "Запустить AI-анализ").click().run()
    button(at, "Сформировать TO-BE и отчёт").click().run()
    assert not at.exception and any("TO-BE не формировался" in i.value for i in at.info)
    assert not at.get("graphviz_chart") and "Сравнение AS-IS и TO-BE" not in html_of(at)
    assert {d.proto.label for d in at.get("download_button")} >= {"Скачать отчёт (Markdown)"}  # отчёт есть при любом исходе


def test_invalid_process_shows_errors_to_fix(monkeypatch, harness):
    bad = {**PROCESS, "steps": [PROCESS["steps"][0], {**PROCESS["steps"][1], "id": "s1"}]}
    at = make_page(monkeypatch, harness, bad)
    button(at, "Запустить AI-анализ").click().run()
    assert not at.exception and any("Процесс некорректен" in e.value for e in at.error)
    assert "уже использован" in html_of(at) and harness.llm.calls == []


def test_changed_process_resets_previous_run(monkeypatch, harness):
    at = make_page(monkeypatch, harness)
    button(at, "Запустить AI-анализ").click().run()
    assert at.session_state[RUN_KEY].status == "awaiting_approval"

    changed = ProcessSpec.model_validate({**PROCESS, "name": "Другой процесс"})
    at.session_state[CONFIRMED_KEY] = changed
    at.run()
    assert RUN_KEY not in at.session_state  # прежний запуск неактуален
    assert not any(b.label == "Сформировать TO-BE и отчёт" for b in at.button)
