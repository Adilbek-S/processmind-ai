"""Помощники для тестов главного workflow.

Настоящими остаются граф LangGraph, checkpointer и MCP-сервер (отдельный процесс по stdio). Подменены
только внешние модели: LLM (GPT-4o-mini) и эмбеддинги — лексическим двойником. Поэтому тесты проверяют
логику workflow, паузу/продолжение и валидацию, а не качество рекомендаций модели.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Callable

from processmind.analysis.recommendations import LLMRecommendation, LLMRecommendationSet
from processmind.mcp.client import McpToolError

# Процесс с известными числами. Известные длительности: 30 + 120 + 60 + 240 = 450 мин (у s4 длительности нет).
# Ручные: s1, s2, s4; автоматизирована: s3; тип не указан: s5.
PROCESS = {
    "process_id": "demo",
    "name": "Согласование закупки",
    "goal": "Оформить закупку",
    "steps": [
        {"id": "s1", "name": "Принять заявку", "actor": "Менеджер", "is_manual": True, "duration_minutes": 30, "next_steps": ["s2"]},
        {"id": "s2", "name": "Согласовать бюджет", "actor": "Финансовый директор", "is_manual": True, "duration_minutes": 120, "next_steps": ["s3"]},
        {"id": "s3", "name": "Оплатить счёт", "actor": "Бухгалтер", "is_manual": False, "duration_minutes": 60, "next_steps": ["s4"]},
        {"id": "s4", "name": "Выбрать поставщика", "actor": "Менеджер", "is_manual": True, "next_steps": ["s5"]},
        {"id": "s5", "name": "Заключить договор", "duration_minutes": 240, "pain_points": ["согласование договора пересылается по почте"]},
    ],
}


def rec(step_id: str, pattern_id: str | None, new_duration: float | None = None, basis: str | None = None, **over) -> LLMRecommendation:
    data = dict(
        step_id=step_id,
        problem=f"Проблема операции {step_id}",
        solution=f"Автоматизировать {step_id}",
        pattern_id=pattern_id,
        rationale=f"Обоснование для {step_id}",
        conditions_confirmed=["Заявки повторяются (подтверждено данными процесса)"] if pattern_id else [],
        conditions_unverified=["Доступ исполнителей к системе (данными не подтверждён)"] if pattern_id else [],
        expected_effect="Заявка обрабатывается без ручной пересылки",
        new_duration_minutes=new_duration,
        duration_basis=basis if basis is not None else ("допущение" if new_duration is not None else None),
    )
    data.update(over)
    return LLMRecommendation(**data)


def context_of(messages: list[dict]) -> dict:
    """JSON-контекст из пользовательского сообщения, которое получила LLM."""
    text = messages[1]["content"]
    return json.loads(text[text.index("{"):])


def honest_llm(messages: list[dict]) -> LLMRecommendationSet:
    """Ведёт себя как добросовестная модель: ссылается только на паттерны, которые ей действительно передали."""
    context = context_of(messages)
    patterns = [p["pattern_id"] for p in context["knowledge_base_patterns"]]
    pattern = patterns[0] if patterns else None
    return LLMRecommendationSet(
        recommendations=[
            rec("s1", pattern, 5, "автоматическая обработка занимает несколько минут (допущение)"),
            rec("s2", pattern, 20, "согласование в системе (допущение)"),
            rec("s4", pattern),  # исходной длительности нет — без предполагаемой
            rec("s5", pattern, 60, "шаблон договора (допущение)"),
        ],
        notes=["У операции s4 не указана длительность"],
    )


class FakeLLMClient:
    """Замена openai.OpenAI: запоминает запросы, ответ строит функция `responder(messages)`."""

    def __init__(self, responder: Callable[[list[dict]], LLMRecommendationSet] = honest_llm, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._responder, self._error = responder, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        message = SimpleNamespace(parsed=self._responder(kwargs["messages"]), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RecordingMcp:
    """Настоящий MCP-клиент (stdio) с журналом вызовов; может имитировать сбой конкретного инструмента."""

    def __init__(self, client, fail_on: set[str] | None = None) -> None:
        self._client = client
        self.calls: list[str] = []
        self.fail_on = fail_on or set()

    def _call(self, name: str, *args):
        self.calls.append(name)
        if name in self.fail_on:
            raise McpToolError(f"имитация сбоя {name}")
        return getattr(self._client, name)(*args)

    def validate_process(self, process):
        return self._call("validate_process", process)

    def calculate_process_metrics(self, process, runs_per_month=None):
        return self._call("calculate_process_metrics", process, runs_per_month)

    def simulate_automation(self, process, automation, runs_per_month=None):
        return self._call("simulate_automation", process, automation, runs_per_month)
