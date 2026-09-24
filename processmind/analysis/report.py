"""AnalysisReport — структурированный итог workflow, сравнение AS-IS/TO-BE и Markdown-отчёт."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from processmind.analysis.recommendations import Recommendation, RejectedRecommendation
from processmind.formatting import MISSING, format_duration, format_hours, format_share
from processmind.models import ProcessSpec

MODEL_NOTICE = (
    "Все показатели «после автоматизации» — модельная оценка на основе заданных предположений "
    "(длительностей операций после автоматизации и числа запусков процесса), а не измерение и не прогноз."
)
DATA_NOTICE = (
    "Данные процесса и база знаний паттернов в этом MVP синтетические (учебные): отчёт не описывает реальную "
    "организацию, а оценка эффекта носит оценочный характер и требует проверки на реальных данных."
)

ReportStatus = Literal["completed", "no_recommendations", "no_selection", "invalid", "failed"]

_KIND_TITLES = {
    "manual_step": "Ручные операции",
    "unknown_type": "Тип выполнения не указан",
    "missing_duration": "Длительность не указана",
    "long_step": "Наиболее длительные операции",
    "pain_point": "Проблемы, указанные в описании процесса",
}


class PatternRef(BaseModel):
    pattern_id: str
    title: str
    source: str
    score: float
    matched_for: list[str]


class ApprovedChange(BaseModel):
    """Рекомендация, подтверждённая пользователем, с длительностью, которая пошла в симуляцию."""

    rec_id: str
    step_id: str
    new_duration_minutes: float
    duration_source: Literal["llm", "user"]  # user — значение задано/изменено пользователем


class ReportRecommendation(Recommendation):
    decision: Literal["approved", "not_selected", "not_offered"] = "not_offered"
    # Длительность, которая реально пошла в симуляцию (для одобренных): значение LLM или допущение пользователя
    approved_duration_minutes: float | None = None
    duration_source: Literal["llm", "user"] | None = None


class SkillRef(BaseModel):
    name: str
    sha256: str


class AnalysisReport(BaseModel):
    process_id: str
    process_name: str
    status: ReportStatus
    process: ProcessSpec | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validation: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    problems: list[dict[str, str]] = Field(default_factory=list, description="Выявленные проблемы — факты из данных процесса")
    patterns: list[PatternRef] = Field(default_factory=list)
    recommendations: list[ReportRecommendation] = Field(default_factory=list)
    rejected_recommendations: list[RejectedRecommendation] = Field(default_factory=list)
    llm_notes: list[str] = Field(default_factory=list)
    approved: list[ApprovedChange] = Field(default_factory=list)
    simulation: dict[str, Any] | None = None
    tobe_metrics: dict[str, Any] | None = None
    to_be: ProcessSpec | None = None
    skill: SkillRef | None = None
    llm_model: str | None = None
    embedding_model: str | None = None
    model_notice: str = MODEL_NOTICE
    data_notice: str = DATA_NOTICE


def build_to_be(spec: ProcessSpec, approved: list[ApprovedChange]) -> ProcessSpec:
    """TO-BE: подтверждённые операции становятся автоматизированными с предполагаемой длительностью.

    Порядок операций и связи между ними не меняются.
    """
    changes = {c.step_id: c.new_duration_minutes for c in approved}
    steps = [
        s.model_copy(update={"is_manual": False, "duration_minutes": changes[s.id]}) if s.id in changes else s
        for s in spec.steps
    ]
    return spec.model_copy(
        update={
            "process_id": f"{spec.process_id}-tobe",
            "name": f"{spec.name} (TO-BE, модельная оценка)",
            "steps": steps,
        }
    )


# ---------- сравнение AS-IS / TO-BE ----------


class MetricRow(BaseModel):
    label: str
    as_is: str
    to_be: str
    delta: str
    note: str | None = None


def _signed(value: float, fmt) -> str:
    if value == 0:
        return "без изменений"
    return f"{'−' if value < 0 else '+'}{fmt(abs(value))}"


def metrics_comparison(metrics: dict[str, Any], tobe_metrics: dict[str, Any]) -> list[MetricRow]:
    """Четыре показателя рядом: общее время, число и доля ручных операций, условная месячная трудоёмкость."""
    rows: list[MetricRow] = []

    before, after = metrics["total_time_minutes"], tobe_metrics["total_time_minutes"]
    pct = f" ({'−' if after < before else '+'}{abs(after - before) / before * 100:.2f}%)".replace(".", ",") if before and after != before else ""
    rows.append(
        MetricRow(
            label="Общее время процесса (один экземпляр)",
            as_is=format_duration(before, with_minutes=True),
            to_be=format_duration(after, with_minutes=True),
            delta=_signed(after - before, lambda v: format_duration(v)) + pct,
            note=None if metrics["time_is_complete"] else "Суммы неполные: у части операций длительность не указана",
        )
    )

    rows.append(
        MetricRow(
            label="Ручных операций",
            as_is=str(metrics["manual_steps"]),
            to_be=str(tobe_metrics["manual_steps"]),
            delta=_signed(tobe_metrics["manual_steps"] - metrics["manual_steps"], str),
            note=None if not metrics["unknown_type_steps"] else f"Тип выполнения не указан у {metrics['unknown_type_steps']} операций — они не считаются ручными",
        )
    )

    share_before, share_after = metrics["manual_share"], tobe_metrics["manual_share"]
    rows.append(
        MetricRow(
            label="Доля ручных операций",
            as_is=format_share(share_before),
            to_be=format_share(share_after),
            delta=_signed((share_after - share_before) * 100, lambda v: f"{v:.0f} п.п."),
            note=None
            if not metrics["unknown_type_steps"]
            else f"Если все операции с неуказанным типом ручные: {format_share(metrics['manual_share_upper_bound'])} → {format_share(tobe_metrics['manual_share_upper_bound'])}",
        )
    )

    monthly_before, monthly_after = metrics["monthly_effort_hours"], tobe_metrics["monthly_effort_hours"]
    if monthly_before is None or monthly_after is None:
        rows.append(MetricRow(label="Условная месячная трудоёмкость", as_is=MISSING, to_be=MISSING, delta="—", note="Число запусков процесса в месяц не задано"))
    else:
        rows.append(
            MetricRow(
                label="Условная месячная трудоёмкость",
                as_is=format_hours(monthly_before),
                to_be=format_hours(monthly_after),
                delta=_signed(monthly_after - monthly_before, format_hours),
                note=f"Время процесса × число запусков в месяц ({metrics['runs_per_month']})",
            )
        )
    return rows


# ---------- Markdown ----------


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip() or "—"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _kind_label(step) -> str:
    return {True: "вручную", False: "автоматизирована", None: "не указан"}[step.is_manual]


def report_to_markdown(report: AnalysisReport) -> str:
    """Итоговый отчёт: процесс, исходные метрики, проблемы, рекомендации, паттерны, модельные показатели TO-BE."""
    out: list[str] = [f"# Отчёт по анализу процесса: {report.process_name}", ""]
    out += [f"> ⚠️ **{report.data_notice}**", ">", f"> **{report.model_notice}**", "", f"**Статус анализа:** `{report.status}`", ""]

    if report.errors:
        out += ["## Ошибки", *[f"- {e}" for e in report.errors], ""]

    step_name = {s.id: s.name for s in report.process.steps} if report.process else {}

    # 1. Описание процесса
    if report.process:
        p = report.process
        out += ["## 1. Описание процесса", "", f"- **Название:** {p.name}", f"- **Цель:** {p.goal or MISSING}"]
        if p.description:
            out.append(f"- **Описание:** {p.description}")
        out += [f"- **Участники:** {', '.join(p.actors) if p.actors else MISSING}", f"- **Операций:** {len(p.steps)}", ""]
        out += _table(
            ["№", "Операция", "Исполнитель", "Тип выполнения", "Длительность"],
            [[i, s.name, s.actor or MISSING, _kind_label(s), format_duration(s.duration_minutes)] for i, s in enumerate(p.steps, start=1)],
        )

    # 2. Исходные метрики
    if report.metrics:
        m = report.metrics
        out += [
            "## 2. Исходные метрики (AS-IS)",
            "",
            f"- Операций всего: **{m['total_steps']}** (ручных {m['manual_steps']}, автоматизированных {m['automated_steps']}, тип не указан {m['unknown_type_steps']})",
            f"- Доля ручных операций: **{format_share(m['manual_share'])}**"
            + (f" (не более {format_share(m['manual_share_upper_bound'])}, если операции с неуказанным типом ручные)" if m["unknown_type_steps"] else ""),
            f"- Суммарное время одного экземпляра: **{format_duration(m['total_time_minutes'], with_minutes=True)}**"
            + ("" if m["time_is_complete"] else " — неполное: у части операций длительность не указана"),
            f"- Время ручных операций: {format_duration(m['manual_time_minutes'], with_minutes=True)}",
            f"- Условная месячная трудоёмкость: {format_hours(m['monthly_effort_hours'])}"
            + ("" if m["runs_per_month"] is None else f" (при {m['runs_per_month']} запусках в месяц)"),
            "",
        ]

    # 3. Выявленные проблемы
    if report.problems:
        out += ["## 3. Выявленные проблемы", "", "Факты, выявленные по данным процесса (без домыслов):", ""]
        for kind, title in _KIND_TITLES.items():
            items = [x for x in report.problems if x["kind"] == kind]
            if items:
                out.append(f"**{title}**")
                out += [f"- {step_name.get(x['step_id'], x['step_id'])} (`{x['step_id']}`): {x['text']}" for x in items]
                out.append("")

    # 4. Рекомендации
    if report.recommendations:
        out += ["## 4. Рекомендации по автоматизации", ""]
        out += _table(
            ["Операция", "Проблема", "Предложение", "Паттерн автоматизации", "Ожидаемый эффект", "Решение пользователя"],
            [
                [
                    r.step_name,
                    r.problem,
                    r.solution,
                    f"{r.pattern_id} — {r.pattern_title}" if r.pattern_id else "не опирается на базу знаний",
                    r.expected_effect,
                    {"approved": "включено в TO-BE", "not_selected": "не выбрано", "not_offered": "не предлагалось"}[r.decision],
                ]
                for r in report.recommendations
            ],
        )
        for r in report.recommendations:
            out += [f"### {r.rec_id}. {r.step_name} (`{r.step_id}`)", f"- **Факты об операции:** {r.process_facts}", f"- **Обоснование:** {r.rationale}"]
            if r.conditions_confirmed:
                out.append(f"- **Условия применимости, подтверждённые данными:** {'; '.join(r.conditions_confirmed)}")
            if r.conditions_unverified:
                out.append(f"- **Не подтверждено данными (нужно проверить):** {'; '.join(r.conditions_unverified)}")
            if r.approved_duration_minutes is not None:
                who = "пользователя" if r.duration_source == "user" else "LLM"
                out.append(f"- **Длительность:** {format_duration(r.original_duration_minutes)} → {format_duration(r.approved_duration_minutes)} (допущение {who}, использовано в симуляции)")
            elif r.new_duration_minutes is not None:
                out.append(f"- **Длительность:** {format_duration(r.original_duration_minutes)} → {format_duration(r.new_duration_minutes)} (допущение LLM: {r.duration_basis})")
            else:
                out.append(f"- **Длительность после автоматизации:** не задана (исходная: {format_duration(r.original_duration_minutes)})")
            out.append("")
    elif report.status in ("no_recommendations", "no_selection", "completed"):
        out += ["## 4. Рекомендации по автоматизации", "", "Обоснованных рекомендаций не сформировано.", ""]

    # 5. Паттерны
    if report.patterns:
        used: dict[str, list[str]] = {}
        for r in report.recommendations:
            if r.pattern_id:
                used.setdefault(r.pattern_id, []).append(r.rec_id)
        out += ["## 5. Использованные паттерны автоматизации", "", "Паттерны найдены семантическим поиском (RAG) по синтетической базе знаний:", ""]
        out += _table(
            ["Паттерн", "Документ базы знаний", "Сходство", "В рекомендациях"],
            [[f"{p.pattern_id} — {p.title}", p.source, f"{p.score:.3f}", ", ".join(used.get(p.pattern_id, [])) or "не использован"] for p in report.patterns],
        )

    # 6. TO-BE
    if report.simulation and report.tobe_metrics and report.metrics:
        s = report.simulation
        rows = metrics_comparison(report.metrics, report.tobe_metrics)
        out += ["## 6. Модельные показатели TO-BE", "", f"> {report.model_notice}", ""]
        out += _table(["Показатель", "AS-IS", "TO-BE (модель)", "Изменение"], [[r.label, r.as_is, r.to_be, r.delta] for r in rows])
        notes = [r for r in rows if r.note]
        out += [f"- {r.label}: {r.note}" for r in notes] + ([""] if notes else [])
        if s["monthly_saving_hours"] is not None:
            out += [f"Условная экономия времени за месяц: **{format_hours(s['monthly_saving_hours'])}** (при {s['runs_per_month']} запусках).", ""]
        rec_by_id = {r.rec_id: r for r in report.recommendations}
        out += ["**Применённые изменения** (порядок операций и связи между ними не изменены):", ""]
        out += _table(
            ["Операция", "Было", "Стало (допущение)", "Источник допущения", "Паттерн"],
            [
                [
                    step_name.get(a.step_id, a.step_id),
                    format_duration(rec_by_id[a.rec_id].original_duration_minutes),
                    format_duration(a.new_duration_minutes),
                    "пользователь" if a.duration_source == "user" else "LLM",
                    rec_by_id[a.rec_id].pattern_id or "нет",
                ]
                for a in report.approved
            ],
        )
    elif report.status == "no_selection":
        out += ["## 6. Модельные показатели TO-BE", "", "Рекомендации не выбраны — TO-BE не формировался.", ""]

    # 7. Предупреждения
    if report.warnings or report.rejected_recommendations or report.llm_notes:
        out += ["## 7. Предупреждения и ограничения", ""]
        out += [f"- {w}" for w in report.warnings]
        out += [f"- Замечание LLM: {note}" for note in report.llm_notes]
        out += [f"- Отклонена валидацией рекомендация по операции `{rej.recommendation.step_id}`: {'; '.join(rej.reasons)}" for rej in report.rejected_recommendations]
        out.append("")

    footer = ""
    if report.skill:
        footer = f"Методика: Skill `{report.skill.name}` (sha256 {report.skill.sha256[:12]}…); LLM: {report.llm_model or '—'}; эмбеддинги: {report.embedding_model or '—'}."
    out += ["---", footer, "", f"_{report.data_notice}_"]
    return "\n".join(out).strip() + "\n"
