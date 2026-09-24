"""Рекомендации по автоматизации: схемы, контекст для LLM, вызов модели и программная валидация.

LLM предлагает, код проверяет. Всё, что модель не вправе выдумывать (ID операций, ссылки на паттерны,
длительности, количественные утверждения об эффекте), проверяется обычным кодом в
`validate_recommendations`; непрошедшие проверку рекомендации отклоняются с причиной и в отчёте
показываются отдельно.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from processmind.analysis.skill import Skill
from processmind.models import ProcessSpec, ProcessStep
from processmind.parsing.llm import request_structured

MAX_LONG_STEP_SIGNALS = 3
# Признаки того, что «подтверждённое» условие на самом деле данными не подтверждается.
_NEGATIVE_EVIDENCE = (
    "нет данных", "данных нет", "нет информации", "не указан", "отсутству", "не подтвержд", "неизвестн", "не известн",
)
# Явные маркеры ручной работы в описании операции (одно лишь наличие описания признаком не является).
_MANUAL_MARKERS = ("вручную", "ручн", "перепечат", "пересыл", "бумаж", "на бумаге")
_QUANTITATIVE_EFFECT = re.compile(r"\d+\s*%|\bв\s+\d+(?:[.,]\d+)?\s+раз", re.IGNORECASE)


class RecommendationError(Exception):
    """LLM не смогла сформировать рекомендации (сообщение показывается пользователю)."""


# ---------- схемы ----------


class LLMRecommendation(BaseModel):
    """Рекомендация в том виде, в каком её возвращает LLM (без значений по умолчанию — строгий режим)."""

    step_id: str = Field(description="ID операции из process.steps[].id")
    problem: str = Field(description="Обнаруженная проблема (только из фактов данных)")
    solution: str = Field(description="Предлагаемое решение для этой операции")
    pattern_id: str | None = Field(description="ID найденного паттерна из knowledge_base_patterns; null, если рекомендация не опирается на базу знаний")
    rationale: str = Field(description="Обоснование: факты процесса, признак проблемы, почему подходит паттерн, релевантные ограничения")
    conditions_confirmed: list[str] = Field(description="Условия применимости паттерна, ПОДТВЕРЖДЁННЫЕ данными процесса; каждое — со ссылкой на факт из данных. Пустой список, если паттерна нет")
    conditions_unverified: list[str] = Field(description="Условия применимости и ограничения паттерна, которые данные подтвердить НЕ позволяют (их нужно проверить). Пустой список, если паттерна нет")
    expected_effect: str = Field(description="Ожидаемый качественный эффект — словами, без чисел и процентов")
    new_duration_minutes: float | None = Field(description="Предполагаемая длительность операции после автоматизации (допущение), минуты. ОБЯЗАТЕЛЬНА, если у операции известна исходная длительность; null только если исходная длительность неизвестна")
    duration_basis: str | None = Field(description="Обоснование допущения: какая часть работы остаётся за человеком, какая выполняется системой; явно назови это допущением. null, если new_duration_minutes = null")


class LLMRecommendationSet(BaseModel):
    recommendations: list[LLMRecommendation]
    notes: list[str] = Field(description="Пробелы в данных, отвергнутые кандидаты и причины")


class Recommendation(BaseModel):
    """Рекомендация, прошедшая программную валидацию."""

    rec_id: str
    step_id: str
    step_name: str
    problem: str
    solution: str
    pattern_id: str | None
    pattern_title: str | None
    pattern_source: str | None
    kb_based: bool
    process_facts: str = Field(description="Факты об операции из ProcessSpec — подставляются кодом, не LLM")
    rationale: str
    conditions_confirmed: list[str]
    conditions_unverified: list[str]
    expected_effect: str
    original_duration_minutes: float | None
    new_duration_minutes: float | None
    duration_basis: str | None
    simulatable: bool = Field(description="Можно включить в симуляцию: известны исходная и предполагаемая длительности")
    warnings: list[str] = Field(default_factory=list)


class RejectedRecommendation(BaseModel):
    recommendation: LLMRecommendation
    reasons: list[str]


class ValidationOutcome(BaseModel):
    accepted: list[Recommendation]
    rejected: list[RejectedRecommendation]


# ---------- контекст для LLM ----------


def has_explicit_manual_signs(step: ProcessStep) -> bool:
    """Явный признак ручной работы: pain_points или маркер («вручную», «пересылка» и т.п.) в описании."""
    description = (step.description or "").casefold()
    return bool(step.pain_points) or any(marker in description for marker in _MANUAL_MARKERS)


def describe_step_facts(step: ProcessStep) -> str:
    """Факты об операции строкой — из данных, а не из ответа LLM (LLM их не пересказывает и не может исказить)."""
    kind = {True: "ручной", False: "автоматизированный", None: "не указан"}[step.is_manual]
    duration = "не указана" if step.duration_minutes is None else f"{step.duration_minutes:g} мин"
    return f"Исполнитель: {step.actor or 'не указан'}; тип выполнения: {kind}; длительность: {duration}"


def build_problem_signals(spec: ProcessSpec, metrics: dict[str, Any]) -> list[dict[str, str]]:
    """«Выявленные проблемы» — только факты из данных, без интерпретаций и без домыслов."""
    signals: list[dict[str, str]] = []
    for step in spec.steps:
        for pain in step.pain_points:
            signals.append({"step_id": step.id, "kind": "pain_point", "text": pain})
        if step.is_manual is True:
            signals.append({"step_id": step.id, "kind": "manual_step", "text": "Операция выполняется вручную"})
        elif step.is_manual is None:
            signals.append({"step_id": step.id, "kind": "unknown_type", "text": "Тип выполнения не указан"})
        if step.duration_minutes is None:
            signals.append({"step_id": step.id, "kind": "missing_duration", "text": "Длительность не указана"})

    total = metrics.get("total_time_minutes") or 0
    known = sorted((s for s in spec.steps if s.duration_minutes), key=lambda s: s.duration_minutes, reverse=True)
    if total > 0 and len(spec.steps) >= MAX_LONG_STEP_SIGNALS:
        for step in known[:MAX_LONG_STEP_SIGNALS]:
            signals.append(
                {
                    "step_id": step.id,
                    "kind": "long_step",
                    "text": f"Известная длительность {step.duration_minutes:g} мин из {total:g} мин суммарно по операциям с известной длительностью",
                }
            )
    return signals


def build_llm_context(
    spec: ProcessSpec,
    validation: dict[str, Any],
    metrics: dict[str, Any],
    patterns: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "process": {
            "name": spec.name,
            "goal": spec.goal,
            "steps": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "actor": s.actor,
                    "is_manual": s.is_manual,
                    "duration_minutes": s.duration_minutes,
                    "pain_points": s.pain_points,
                    "systems": s.systems,
                    "inputs": s.inputs,
                    "outputs": s.outputs,
                    "next_steps": s.next_steps,
                }
                for s in spec.steps
            ],
        },
        "metrics": metrics,
        "problem_signals": build_problem_signals(spec, metrics),
        "knowledge_base_patterns": [
            {
                "pattern_id": p["pattern_id"],
                "title": p["title"],
                "relevant_for": p["matched_for"],
                "similarity": p["score"],
                "sections": p["sections"],
            }
            for p in patterns
        ],
        "validation_warnings": [w["message"] if isinstance(w, dict) else str(w) for w in validation.get("warnings", [])],
    }


def build_messages(skill: Skill, context: dict[str, Any]) -> list[dict[str, str]]:
    """Системный промпт = методика Skill целиком; пользовательское сообщение = данные процесса."""
    return [
        {"role": "system", "content": skill.body},
        {
            "role": "user",
            "content": "Проанализируй процесс по методике и верни рекомендации. Данные:\n\n"
            + json.dumps(context, ensure_ascii=False, indent=1),
        },
    ]


def generate_recommendations(client: OpenAI, model: str, skill: Skill, context: dict[str, Any]) -> LLMRecommendationSet:
    return request_structured(client, model, build_messages(skill, context), LLMRecommendationSet, RecommendationError)


# ---------- программная валидация ----------


def _blank(value: str | None) -> bool:
    return value is None or not value.strip()


def validate_recommendations(
    raw: LLMRecommendationSet,
    spec: ProcessSpec,
    patterns: dict[str, dict[str, Any]],
) -> ValidationOutcome:
    """Отсеивает рекомендации с несуществующими ID, ложными ссылками на паттерны и выдуманными числами.

    `patterns` — найденные при поиске паттерны: {pattern_id: {"title", "source", ...}}.
    """
    steps = {s.id: s for s in spec.steps}
    accepted: list[Recommendation] = []
    rejected: list[RejectedRecommendation] = []
    used_steps: set[str] = set()

    for rec in raw.recommendations:
        reasons: list[str] = []
        warnings: list[str] = []
        step = steps.get(rec.step_id)

        if step is None:
            reasons.append(f"операции «{rec.step_id}» нет в процессе")
        else:
            if step.is_manual is False:
                reasons.append("операция уже автоматизирована")
            if step.is_manual is None and not has_explicit_manual_signs(step):
                reasons.append("тип выполнения операции не указан и нет явных признаков ручной работы (pain_points или маркеры ручной работы в описании) — по методике такая операция не рекомендуется")
            if step.id in used_steps:
                reasons.append("для этой операции уже есть рекомендация")

        for label, value in (("problem", rec.problem), ("solution", rec.solution), ("rationale", rec.rationale), ("expected_effect", rec.expected_effect)):
            if _blank(value):
                reasons.append(f"пустое поле {label}")

        pattern_id = None if _blank(rec.pattern_id) else rec.pattern_id.strip()
        if pattern_id is not None and pattern_id not in patterns:
            reasons.append(f"паттерн «{pattern_id}» не найден среди результатов поиска по базе знаний")

        confirmed = [c.strip() for c in rec.conditions_confirmed if not _blank(c)]
        unverified = [c.strip() for c in rec.conditions_unverified if not _blank(c)]
        contradictory = [c for c in confirmed if any(marker in c.casefold() for marker in _NEGATIVE_EVIDENCE)]
        if contradictory:  # «подтверждено: нет данных» — это не подтверждение, а пробел
            confirmed = [c for c in confirmed if c not in contradictory]
            unverified = [*unverified, *contradictory]
            warnings.append("условия, для которых в данных нет фактов, перенесены из «подтверждённых» в «непроверенные»")
        if pattern_id is not None and not confirmed and not unverified:
            reasons.append("не проверены условия применимости паттерна (нет ни подтверждённых, ни непроверенных условий)")

        if not _blank(rec.expected_effect) and _QUANTITATIVE_EFFECT.search(rec.expected_effect):
            reasons.append("ожидаемый эффект содержит количественное утверждение (проценты или кратность), которого нет в данных")

        duration = rec.new_duration_minutes
        basis = None if _blank(rec.duration_basis) else rec.duration_basis.strip()
        if duration is not None:
            if not math.isfinite(duration) or duration < 0:
                reasons.append("предполагаемая длительность должна быть конечным неотрицательным числом")
            elif basis is None:
                warnings.append("предполагаемая длительность отброшена: не указано основание допущения")
                duration = None
            elif step is not None and step.duration_minutes is None:
                warnings.append("предполагаемая длительность отброшена: у операции не указана исходная длительность")
                duration, basis = None, None
            elif step is not None and duration > step.duration_minutes:
                warnings.append(
                    f"предполагаемая длительность ({duration:g} мин) больше исходной ({step.duration_minutes:g} мин) — отброшена"
                )
                duration, basis = None, None
        else:
            basis = None

        if reasons:
            rejected.append(RejectedRecommendation(recommendation=rec, reasons=reasons))
            continue

        assert step is not None  # reasons пуст => операция существует
        used_steps.add(step.id)
        pattern = patterns.get(pattern_id) if pattern_id else None
        accepted.append(
            Recommendation(
                rec_id=f"R{len(accepted) + 1}",
                step_id=step.id,
                step_name=step.name,
                problem=rec.problem.strip(),
                solution=rec.solution.strip(),
                pattern_id=pattern_id,
                pattern_title=pattern["title"] if pattern else None,
                pattern_source=pattern["source"] if pattern else None,
                kb_based=pattern_id is not None,
                process_facts=describe_step_facts(step),
                rationale=rec.rationale.strip(),
                conditions_confirmed=confirmed,
                conditions_unverified=unverified,
                expected_effect=rec.expected_effect.strip(),
                original_duration_minutes=step.duration_minutes,
                new_duration_minutes=duration,
                duration_basis=basis,
                simulatable=duration is not None and step.duration_minutes is not None,
                warnings=warnings,
            )
        )
    return ValidationOutcome(accepted=accepted, rejected=rejected)
