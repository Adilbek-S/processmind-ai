"""Единые структурированные модели данных ProcessMind (Pydantic).

ProcessSpec — центральная модель бизнес-процесса, используемая всеми остальными
компонентами (парсеры, LangGraph-пайплайн, анализатор, визуализация, оценка).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessStep(BaseModel):
    """Один шаг (операция) бизнес-процесса."""

    id: str
    name: str
    description: str | None = None
    actor: str | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    systems: list[str] = Field(default_factory=list)
    # None — значение в источнике не указано (не ноль и не «по умолчанию»).
    duration_minutes: float | None = Field(default=None, ge=0)
    # True — ручная, False — автоматизированная, None — тип выполнения не указан.
    is_manual: bool | None = None
    pain_points: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class ProcessSpec(BaseModel):
    """Единая структурированная модель бизнес-процесса (AS-IS или TO-BE)."""

    process_id: str
    name: str
    description: str | None = None
    goal: str | None = None
    steps: list[ProcessStep] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    systems: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
