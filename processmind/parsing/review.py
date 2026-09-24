"""Применение пользовательских правок к распознанному ProcessSpec (шаг подтверждения)."""

from __future__ import annotations

import math
from typing import Any

from pydantic import ValidationError

from processmind.models import ProcessSpec

EXEC_MANUAL = "Ручная"
EXEC_AUTOMATED = "Автоматизированная"
EXEC_UNKNOWN = "Не указано"
EXEC_OPTIONS = [EXEC_MANUAL, EXEC_AUTOMATED, EXEC_UNKNOWN]

_IS_MANUAL_BY_LABEL = {EXEC_MANUAL: True, EXEC_AUTOMATED: False, EXEC_UNKNOWN: None}
_LABEL_BY_IS_MANUAL = {True: EXEC_MANUAL, False: EXEC_AUTOMATED, None: EXEC_UNKNOWN}


class ReviewError(ValueError):
    """Правки пользователя некорректны (сообщение показывается в интерфейсе)."""


def spec_to_rows(spec: ProcessSpec) -> list[dict[str, Any]]:
    """Строки таблицы для редактора. Отсутствующие значения — пустые ячейки (None)."""
    number_by_id = {step.id: i + 1 for i, step in enumerate(spec.steps)}
    return [
        {
            "№": i + 1,
            "Операция": step.name,
            "Участник": step.actor,
            "Длительность, мин": step.duration_minutes,
            "Тип выполнения": _LABEL_BY_IS_MANUAL[step.is_manual],
            "Следующие": ", ".join(str(number_by_id[n]) for n in step.next_steps) or "—",
        }
        for i, step in enumerate(spec.steps)
    ]


def _blank_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def apply_edits(spec: ProcessSpec, rows: list[dict[str, Any]]) -> ProcessSpec:
    """Возвращает новый ProcessSpec с учтёнными правками названия, участника, длительности и типа.

    Число и порядок операций и связи между ними не меняются. Результат проходит валидацию Pydantic.
    """
    if len(rows) != len(spec.steps):
        raise ReviewError("Число операций в таблице не совпадает с распознанным процессом.")

    new_steps = []
    for i, (step, row) in enumerate(zip(spec.steps, rows), start=1):
        name = _blank_to_none(row.get("Операция"))
        if name is None:
            raise ReviewError(f"Операция №{i}: название не может быть пустым.")

        actor = _blank_to_none(row.get("Участник"))

        duration = _blank_to_none(row.get("Длительность, мин"))
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError) as exc:
                raise ReviewError(f"Операция №{i}: длительность должна быть числом.") from exc
            if duration < 0:
                raise ReviewError(f"Операция №{i}: длительность не может быть отрицательной.")

        label = _blank_to_none(row.get("Тип выполнения")) or EXEC_UNKNOWN
        if label not in _IS_MANUAL_BY_LABEL:
            raise ReviewError(f"Операция №{i}: неизвестный тип выполнения «{label}».")

        new_steps.append(
            step.model_copy(
                update={
                    "name": str(name).strip(),
                    "actor": str(actor).strip() if actor is not None else None,
                    "duration_minutes": duration,
                    "is_manual": _IS_MANUAL_BY_LABEL[label],
                }
            )
        )

    old_step_actors = {s.actor for s in spec.steps if s.actor}
    extra_participants = [a for a in spec.actors if a not in old_step_actors]
    actors = list(dict.fromkeys([s.actor for s in new_steps if s.actor] + extra_participants))

    try:
        return ProcessSpec.model_validate({**spec.model_dump(), "steps": [s.model_dump() for s in new_steps], "actors": actors})
    except ValidationError as exc:
        raise ReviewError(f"Правки не прошли проверку: {exc}") from exc
