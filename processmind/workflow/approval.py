"""Решение пользователя на шаге human_approval: какие рекомендации включить в TO-BE и с какой длительностью."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from processmind.analysis.report import ApprovedChange


class ApprovalError(ValueError):
    """Решение пользователя некорректно. Workflow остаётся на паузе — можно отправить исправленное решение."""


class Selection(BaseModel):
    rec_id: str
    # None — принять предполагаемую длительность LLM; число — предположение пользователя (обязательно, если LLM её не дала)
    new_duration_minutes: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ApprovalDecision(BaseModel):
    selections: list[Selection] = Field(default_factory=list)  # пусто — ничего не включать в TO-BE


def parse_decision(decision: Any, recommendations: list[dict[str, Any]]) -> list[ApprovedChange]:
    """Проверяет решение относительно предложенных рекомендаций и возвращает подтверждённые изменения."""
    try:
        parsed = ApprovalDecision.model_validate(decision)
    except ValidationError as exc:
        raise ApprovalError(f"Некорректное решение: {exc}") from exc

    by_id = {r["rec_id"]: r for r in recommendations}
    approved: list[ApprovedChange] = []
    seen: set[str] = set()
    for sel in parsed.selections:
        rec = by_id.get(sel.rec_id)
        if rec is None:
            raise ApprovalError(f"Рекомендации «{sel.rec_id}» не было среди предложенных")
        if sel.rec_id in seen:
            raise ApprovalError(f"Рекомендация «{sel.rec_id}» выбрана несколько раз")
        seen.add(sel.rec_id)
        if rec["original_duration_minutes"] is None:
            raise ApprovalError(
                f"Рекомендацию «{sel.rec_id}» нельзя смоделировать: у операции «{rec['step_id']}» не указана исходная длительность"
            )
        duration = sel.new_duration_minutes if sel.new_duration_minutes is not None else rec["new_duration_minutes"]
        if duration is None:
            raise ApprovalError(f"Укажите предполагаемую длительность после автоматизации для «{sel.rec_id}»")
        user_defined = sel.new_duration_minutes is not None and sel.new_duration_minutes != rec["new_duration_minutes"]
        approved.append(
            ApprovedChange(
                rec_id=sel.rec_id,
                step_id=rec["step_id"],
                new_duration_minutes=duration,
                duration_source="user" if user_defined else "llm",
            )
        )
    return approved
