"""Тесты форматирования, сравнения AS-IS/TO-BE, Markdown-отчёта и компонентов интерфейса.

Отчёты строит настоящий workflow (MCP-сервер по stdio, граф, checkpointer); LLM и эмбеддинги подменены.
"""

import copy

import pytest

from processmind.analysis.report import (
    DATA_NOTICE,
    MODEL_NOTICE,
    AnalysisReport,
    metrics_comparison,
    report_to_markdown,
)
from processmind.formatting import format_duration, format_hours, format_share
from processmind.ui.components import (
    changes_from_report,
    comparison_table_html,
    initial_metric_items,
    recommendations_table_html,
)
from processmind.ui.state import CONFIRMED_KEY, DRAFT_KEY, RUN_KEY, current_stage
from processmind.workflow.graph import AnalysisRun, resume_analysis
from tests.helpers_workflow import PROCESS, FakeLLMClient, rec
from tests.test_analysis_workflow import Harness, kb, mcp_client  # noqa: F401 (фикстуры)

RUNS = 40


# ---------- форматирование ----------


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(None, "не указано"), (0, "0 мин"), (30, "30 мин"), (59.5, "59,5 мин"), (60, "1 ч"), (90, "1,5 ч"), (1440, "1 дн"), (4320, "3 дн")],
)
def test_format_duration(minutes, expected):
    assert format_duration(minutes) == expected


def test_format_duration_with_exact_minutes_for_totals():
    assert format_duration(4535, with_minutes=True) == "4535 мин (≈ 3,1 дн)"
    assert format_duration(30, with_minutes=True) == "30 мин"


def test_format_share_and_hours():
    assert format_share(0.25) == "25%" and format_share(1) == "100%"
    assert format_hours(250) == "250 ч" and format_hours(1.25) == "1,2 ч" and format_hours(None) == "не указано"


# ---------- сравнение метрик ----------

AS_IS = {
    "total_time_minutes": 450, "manual_steps": 3, "manual_share": 0.6, "manual_share_upper_bound": 0.8, "unknown_type_steps": 1,
    "time_is_complete": False, "monthly_effort_hours": 300.0, "runs_per_month": 40,
}
TO_BE = {**AS_IS, "total_time_minutes": 335, "manual_steps": 1, "manual_share": 0.2, "manual_share_upper_bound": 0.4, "monthly_effort_hours": 223.3}


def test_metrics_comparison_covers_the_four_required_indicators():
    rows = metrics_comparison(AS_IS, TO_BE)
    assert [r.label for r in rows] == [
        "Общее время процесса (один экземпляр)", "Ручных операций", "Доля ручных операций", "Условная месячная трудоёмкость",
    ]
    time_row, count_row, share_row, monthly_row = rows
    assert time_row.as_is.startswith("450 мин") and time_row.to_be.startswith("335 мин")
    assert time_row.delta == "−1,9 ч (−25,56%)"  # 115 мин; 115/450 = 25,56%
    assert (count_row.as_is, count_row.to_be, count_row.delta) == ("3", "1", "−2")
    assert (share_row.as_is, share_row.to_be, share_row.delta) == ("60%", "20%", "−40 п.п.")
    assert (monthly_row.as_is, monthly_row.to_be) == ("300 ч", "223,3 ч") and monthly_row.delta.startswith("−")


def test_metrics_comparison_states_data_gaps_and_missing_runs():
    rows = metrics_comparison(AS_IS, {**TO_BE, "monthly_effort_hours": None})
    assert "неполные" in rows[0].note and "не считаются ручными" in rows[1].note and "Если все операции" in rows[2].note
    monthly = metrics_comparison({**AS_IS, "monthly_effort_hours": None, "runs_per_month": None}, {**TO_BE, "monthly_effort_hours": None})[3]
    assert monthly.as_is == monthly.to_be == "не указано" and "не задано" in monthly.note


def test_metrics_comparison_no_change_and_growth():
    same = metrics_comparison(AS_IS, AS_IS)
    assert same[0].delta == "без изменений" and same[1].delta == "без изменений"
    worse = metrics_comparison(AS_IS, {**AS_IS, "total_time_minutes": 500})
    assert worse[0].delta.startswith("+")


def test_comparison_table_html_colors_improvements_and_escapes():
    rows = metrics_comparison(AS_IS, TO_BE)
    html_text = comparison_table_html(rows)
    assert html_text.count("pm-good") >= 3 and "AS-IS" in html_text and "TO-BE (модельная оценка)" in html_text
    evil = rows[0].model_copy(update={"label": "<script>alert(1)</script>"})
    assert "<script>" not in comparison_table_html([evil]) and "&lt;script&gt;" in comparison_table_html([evil])


# ---------- отчёты настоящего workflow ----------


@pytest.fixture
def harness(mcp_client, kb):  # noqa: F811
    return Harness(mcp_client, kb)


def completed_report(harness, selection=None) -> AnalysisReport:
    run = harness.start()
    r = {x["step_id"]: x["rec_id"] for x in run.approval_request["recommendations"]}
    selection = selection or [{"rec_id": r["s1"]}, {"rec_id": r["s2"], "new_duration_minutes": 30}]
    return resume_analysis(harness.graph, run.thread_id, {"selections": selection}).report


def test_tobe_metrics_are_calculated_by_mcp_and_consistent_with_simulation(harness):
    report = completed_report(harness)
    assert harness.mcp.calls == ["validate_process", "calculate_process_metrics", "simulate_automation", "calculate_process_metrics"]
    tm, sim, m = report.tobe_metrics, report.simulation, report.metrics
    assert tm["total_time_minutes"] == sim["proposed_time_minutes"] == 335 and m["total_time_minutes"] == sim["baseline_time_minutes"] == 450
    assert tm["manual_steps"] == 1 and m["manual_steps"] == 3  # s1 и s2 автоматизированы, ручным остаётся s4
    assert tm["manual_share"] == pytest.approx(sim["proposed_manual_share"]) == pytest.approx(0.2)
    assert tm["monthly_effort_minutes"] == 335 * RUNS and m["monthly_effort_minutes"] == 450 * RUNS


def test_report_contains_process_problems_and_notices(harness):
    report = completed_report(harness)
    assert report.process.name == "Согласование закупки" and len(report.process.steps) == 5
    assert {(p["step_id"], p["kind"]) for p in report.problems} >= {("s1", "manual_step"), ("s4", "missing_duration"), ("s5", "unknown_type")}
    assert report.data_notice == DATA_NOTICE and report.model_notice == MODEL_NOTICE


def test_markdown_report_has_all_required_sections_and_warnings(harness):
    text = report_to_markdown(completed_report(harness))

    for heading in (
        "# Отчёт по анализу процесса: Согласование закупки",
        "## 1. Описание процесса", "## 2. Исходные метрики (AS-IS)", "## 3. Выявленные проблемы",
        "## 4. Рекомендации по автоматизации", "## 5. Использованные паттерны автоматизации", "## 6. Модельные показатели TO-BE",
    ):
        assert heading in text, heading
    # предупреждение о синтетическом характере данных и оценочном характере эффекта — вверху и внизу
    assert text.count(DATA_NOTICE) == 2 and text.count(MODEL_NOTICE) >= 2
    assert text.index(DATA_NOTICE) < text.index("## 1.")
    # таблица рекомендаций с нужными столбцами
    assert "| Операция | Проблема | Предложение | Паттерн автоматизации | Ожидаемый эффект | Решение пользователя |" in text
    # описание процесса, метрики, проблемы, паттерны и модельные показатели
    assert "| 1 | Принять заявку | Менеджер | вручную | 30 мин |" in text
    assert "Суммарное время одного экземпляра: **450 мин (≈ 7,5 ч)**" in text
    assert "Ручные операции" in text and "Длительность не указана" in text
    assert "| Показатель | AS-IS | TO-BE (модель) | Изменение |" in text and "335 мин" in text
    assert "| Принять заявку | 30 мин | 5 мин | LLM |" in text and "| Согласовать бюджет | 2 ч | 30 мин | пользователь |" in text
    assert "порядок операций и связи между ними не изменены" in text


def test_markdown_references_patterns_and_source_documents(harness):
    report = completed_report(harness)
    text = report_to_markdown(report)
    used = {r.pattern_id for r in report.recommendations if r.pattern_id}
    assert used and all(f"{pid} —" in text for pid in used)
    assert all(p.source in text for p in report.patterns)
    assert "включено в TO-BE" in text and "не выбрано" in text


def test_markdown_escapes_pipes_and_newlines_in_table_cells(mcp_client, kb):  # noqa: F811
    from tests.helpers_workflow import honest_llm, context_of
    from processmind.analysis.recommendations import LLMRecommendationSet

    def responder(messages):
        pattern = context_of(messages)["knowledge_base_patterns"][0]["pattern_id"]
        return LLMRecommendationSet(recommendations=[rec("s1", pattern, 5, problem="Есть | разделитель\nи перенос")], notes=[])

    h = Harness(mcp_client, kb, llm=FakeLLMClient(responder))
    run = h.start()
    text = report_to_markdown(resume_analysis(h.graph, run.thread_id, {"selections": [{"rec_id": "R1"}]}).report)
    assert "Есть \\| разделитель и перенос" in text


def test_markdown_for_non_completed_outcomes(mcp_client, kb):  # noqa: F811
    bad = copy.deepcopy(PROCESS)
    bad["steps"][1]["id"] = "s1"
    invalid = report_to_markdown(Harness(mcp_client, kb).start(bad).report)
    assert "`invalid`" in invalid and "## Ошибки" in invalid and "уже использован" in invalid and "## 6." not in invalid and DATA_NOTICE in invalid

    h = Harness(mcp_client, kb)
    run = h.start()
    no_selection = report_to_markdown(resume_analysis(h.graph, run.thread_id, {"selections": []}).report)
    assert "Рекомендации не выбраны — TO-BE не формировался" in no_selection and "## 5." in no_selection


def test_invalid_process_report_has_no_process_and_no_diagrams_data(mcp_client, kb):  # noqa: F811
    bad = {"process_id": "p", "name": "Плохой", "steps": [{"id": "a", "name": "A", "duration_minutes": -5}]}
    report = Harness(mcp_client, kb).start(bad).report
    assert report.status == "invalid" and report.process is None and report.to_be is None and report.tobe_metrics is None


# ---------- компоненты ----------


def test_recommendations_table_has_required_columns_for_dicts_and_objects(harness):
    run = harness.start()
    table = recommendations_table_html(run.approval_request["recommendations"])
    for column in ("Операция", "Проблема", "Предложение", "Паттерн автоматизации", "Ожидаемый эффект"):
        assert f"<th>{column}</th>" in table
    assert "<th>Решение</th>" not in table and "R1. Принять заявку" in table and "AP-" in table

    report = resume_analysis(harness.graph, run.thread_id, {"selections": [{"rec_id": "R1"}]}).report
    final = recommendations_table_html(report.recommendations, show_decision=True)
    assert "<th>Решение</th>" in final and "включено в TO-BE" in final and "не выбрано" in final


def test_recommendations_table_escapes_llm_text_and_marks_missing_pattern():
    item = {"rec_id": "R1", "step_name": "<i>x</i>", "problem": "<script>1</script>", "solution": "a & b", "pattern_id": None, "pattern_title": None, "expected_effect": "ok"}
    table = recommendations_table_html([item])
    assert "<script>" not in table and "&lt;script&gt;" in table and "a &amp; b" in table and "не опирается на базу знаний" in table


def test_changes_for_tobe_diagram_come_from_approved_recommendations(harness):
    report = completed_report(harness)
    changes = {c.step_id: c for c in changes_from_report(report)}
    assert set(changes) == {"s1", "s2"}
    assert (changes["s1"].original_minutes, changes["s1"].new_minutes) == (30, 5)
    assert (changes["s2"].original_minutes, changes["s2"].new_minutes) == (120, 30)
    assert changes["s1"].pattern_id and changes["s1"].pattern_id.startswith("AP-")


def test_initial_metric_items_are_four_cards(harness):
    run = harness.start()
    items = initial_metric_items(harness.graph.get_state({"configurable": {"thread_id": run.thread_id}}).values["metrics"])
    assert [i[0] for i in items] == ["Операций", "Доля ручных", "Время процесса", "Месячная трудоёмкость"]
    assert items[0][1] == "5" and items[1][1] == "60%" and items[3][1] == "300 ч"


def test_current_stage_follows_user_progress():
    finished = AnalysisRun(thread_id="t", status="finished")
    waiting = AnalysisRun(thread_id="t", status="awaiting_approval")
    assert current_stage({}) == 0
    assert current_stage({DRAFT_KEY: object()}) == 1
    assert current_stage({DRAFT_KEY: object(), CONFIRMED_KEY: object()}) == 2
    assert current_stage({CONFIRMED_KEY: object(), RUN_KEY: waiting}) == 3
    assert current_stage({CONFIRMED_KEY: object(), RUN_KEY: finished}) == 4
    assert current_stage({RUN_KEY: finished}) == 0  # запуск без подтверждённого процесса не считается
