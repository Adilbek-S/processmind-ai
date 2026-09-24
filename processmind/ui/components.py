"""Компоненты интерфейса: карточки метрик, таблица сравнения AS-IS/TO-BE, таблица рекомендаций.

HTML собирается из данных процесса и ответа LLM, поэтому весь текст экранируется.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

import streamlit as st

from processmind.analysis.report import AnalysisReport, MetricRow
from processmind.formatting import format_duration, format_hours, format_share
from processmind.visualization.diagrams import StepChange


def _e(value: Any) -> str:
    return html.escape(str(value))


def metric_card_html(label: str, value: str, hint: str | None = None) -> str:
    hint_html = f'<div class="hint">{_e(hint)}</div>' if hint else ""
    return f'<div class="pm-card pm-metric"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>{hint_html}</div>'


def metric_cards(items: Sequence[tuple[str, str, str | None]]) -> None:
    """Ряд карточек: (подпись, значение, подсказка)."""
    for column, (label, value, hint) in zip(st.columns(len(items)), items):
        column.markdown(metric_card_html(label, value, hint), unsafe_allow_html=True)


def initial_metric_items(m: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """Четыре карточки исходных метрик из результата MCP calculate_process_metrics."""
    unknown = m["unknown_type_steps"]
    return [
        ("Операций", str(m["total_steps"]), f"ручных {m['manual_steps']}, автоматизированных {m['automated_steps']}, тип не указан {unknown}"),
        ("Доля ручных", format_share(m["manual_share"]), f"до {format_share(m['manual_share_upper_bound'])}, если операции с неуказанным типом ручные" if unknown else None),
        ("Время процесса", format_duration(m["total_time_minutes"]), f"{m['total_time_minutes']:g} мин · " + ("неполное: у части операций длительность не указана" if not m["time_is_complete"] else f"ручных операций: {format_duration(m['manual_time_minutes'])}")),
        ("Месячная трудоёмкость", format_hours(m["monthly_effort_hours"]), f"условная, {m['runs_per_month']} запусков в месяц" if m["runs_per_month"] is not None else "число запусков не задано"),
    ]


def comparison_table_html(rows: Sequence[MetricRow]) -> str:
    """AS-IS и TO-BE рядом: показатель | AS-IS | TO-BE (модель) | изменение."""
    body = []
    for r in rows:
        css = "pm-good" if r.delta.startswith("−") else ("pm-bad" if r.delta.startswith("+") else "pm-neutral")
        note = f'<span class="pm-note">{_e(r.note)}</span>' if r.note else ""
        body.append(
            f'<tr><td class="pm-key">{_e(r.label)}{note}</td><td>{_e(r.as_is)}</td><td><b>{_e(r.to_be)}</b></td>'
            f'<td class="{css}">{_e(r.delta)}</td></tr>'
        )
    return (
        '<table class="pm-table"><thead><tr><th>Показатель</th><th>AS-IS</th><th>TO-BE (модельная оценка)</th><th>Изменение</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def _get(item: Any, key: str) -> Any:
    """Рекомендация приходит как словарь (запрос подтверждения) или как объект (отчёт)."""
    return item[key] if isinstance(item, dict) else getattr(item, key)


def recommendations_table_html(recommendations: Sequence[Any], *, show_decision: bool = False) -> str:
    """Таблица: Операция | Проблема | Предложение | Паттерн автоматизации | Ожидаемый эффект (+ решение пользователя)."""
    head = ["Операция", "Проблема", "Предложение", "Паттерн автоматизации", "Ожидаемый эффект"] + (["Решение"] if show_decision else [])
    decisions = {
        "approved": '<span class="pm-badge">включено в TO-BE</span>',
        "not_selected": '<span class="pm-badge off">не выбрано</span>',
        "not_offered": '<span class="pm-badge off">не предлагалось</span>',
    }
    rows = []
    for r in recommendations:
        if _get(r, "pattern_id"):
            pattern = f"<b>{_e(_get(r, 'pattern_id'))}</b><br>{_e(_get(r, 'pattern_title'))}"
        else:
            pattern = '<span class="pm-neutral">не опирается на базу знаний</span>'
        cells = [
            f'<td class="pm-key">{_e(_get(r, "rec_id"))}. {_e(_get(r, "step_name"))}</td>',
            f'<td>{_e(_get(r, "problem"))}</td>',
            f'<td>{_e(_get(r, "solution"))}</td>',
            f"<td>{pattern}</td>",
            f'<td>{_e(_get(r, "expected_effect"))}</td>',
        ]
        if show_decision:
            cells.append(f'<td>{decisions[_get(r, "decision")]}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{h}</th>" for h in head)
    return f'<table class="pm-table"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def changes_from_report(report: AnalysisReport) -> list[StepChange]:
    """Принятые изменения для TO-BE-диаграммы: было → стало и применённый паттерн."""
    by_id = {r.rec_id: r for r in report.recommendations}
    return [
        StepChange(
            step_id=a.step_id,
            original_minutes=by_id[a.rec_id].original_duration_minutes,
            new_minutes=a.new_duration_minutes,
            pattern_id=by_id[a.rec_id].pattern_id,
        )
        for a in report.approved
    ]
