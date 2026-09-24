"""Схемы «сырого» вывода LLM и результат извлечения.

`RawProcess` — то, что модель возвращает (Structured Outputs). Он намеренно отделён от
`ProcessSpec`: сырой вывод LLM не доверенный, и перед попаданием в ProcessSpec проходит
нормализацию и проверки (см. `process_builder.py`).

Все поля без значений по умолчанию — этого требует строгий режим Structured Outputs.
Неизвестное значение модель обязана вернуть как `null` / `"unknown"`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from processmind.models import ProcessSpec


class ExtractionError(Exception):
    """Не удалось получить пригодный ProcessSpec (сообщение показывается пользователю)."""


class RawStep(BaseModel):
    order: int = Field(description="Порядковый номер операции, начиная с 1, в порядке чтения источника")
    name: str = Field(description="Короткое название операции (глагол + объект)")
    description: str | None = Field(description="Уточнение к операции, если оно есть в источнике, иначе null")
    actor: str | None = Field(description="Исполнитель, ЯВНО названный в источнике, иначе null")
    duration_value: float | None = Field(description="Длительность числом, ЯВНО указанная в источнике, иначе null")
    duration_unit: Literal["seconds", "minutes", "hours", "days"] | None = Field(
        description="Единица длительности; null, если длительность не указана"
    )
    duration_quote: str | None = Field(
        description="Дословный фрагмент источника, из которого взята длительность; null, если её нет"
    )
    execution_type: Literal["manual", "automated", "unknown"] = Field(
        description="manual — вручную, automated — системой/автоматически, unknown — не указано"
    )
    next_orders: list[int] = Field(
        description="Номера операций (order), которые следуют сразу за этой; пустой список у завершающих"
    )


class RawProcess(BaseModel):
    name: str | None = Field(description="Название процесса, если оно указано в источнике, иначе null")
    goal: str | None = Field(description="Цель процесса, если она указана в источнике, иначе null")
    participants: list[str] = Field(description="Участники, явно названные в источнике; пустой список, если нет")
    steps: list[RawStep]
    uncertainties: list[str] = Field(
        description="Что распознано неуверенно или нечитаемо (плохое качество, обрезанный текст и т.п.)"
    )


class ExtractionResult(BaseModel):
    """Итог извлечения: провалидированный ProcessSpec и предупреждения для пользователя."""

    spec: ProcessSpec
    warnings: list[str] = Field(default_factory=list)
