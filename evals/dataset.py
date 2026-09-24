"""Golden Dataset: модели, загрузка и строгая валидация."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from processmind.mcp.process_tools import validate_process_data
from processmind.rag.knowledge_base import DEFAULT_KB_DIR
from processmind.rag.patterns import load_patterns

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"

Kind = Literal["positive", "negative", "ambiguous"]


class RecommendationRule(BaseModel):
    """Допустимая или недопустимая рекомендация: операция и (необязательно) паттерны; None — любые паттерны."""

    step_id: str
    pattern_ids: list[str] | None = None
    reason: str | None = None


class GoldenExample(BaseModel):
    test_id: str
    category: str
    kind: Kind
    description: str
    process: dict
    question: str
    target_step_id: str | None
    expected_pattern_ids: list[str] = Field(default_factory=list)
    acceptable_recommendations: list[RecommendationRule] = Field(default_factory=list)
    unacceptable_recommendations: list[RecommendationRule] = Field(default_factory=list)

    @property
    def step_ids(self) -> set[str]:
        return {s["id"] for s in self.process["steps"]}

    @property
    def acceptable_patterns(self) -> set[str]:
        """Паттерны, допустимые сверх ожидаемых (не поощряются и не штрафуются)."""
        return {p for rule in self.acceptable_recommendations for p in (rule.pattern_ids or [])} - set(self.expected_pattern_ids)

    @model_validator(mode="after")
    def _check(self) -> GoldenExample:
        report = validate_process_data(self.process)
        if not report.valid:
            raise ValueError(f"{self.test_id}: процесс некорректен: {[e.message for e in report.errors]}")
        for rule in (*self.acceptable_recommendations, *self.unacceptable_recommendations):
            if rule.step_id not in self.step_ids:
                raise ValueError(f"{self.test_id}: правило ссылается на несуществующую операцию {rule.step_id}")
        if self.kind == "positive":
            if not self.expected_pattern_ids or self.target_step_id not in self.step_ids:
                raise ValueError(f"{self.test_id}: positive-пример требует ожидаемых паттернов и целевой операции")
        elif self.expected_pattern_ids:
            raise ValueError(f"{self.test_id}: у {self.kind}-примера не должно быть ожидаемых паттернов")
        if self.kind == "negative" and self.target_step_id is not None:
            raise ValueError(f"{self.test_id}: у negative-примера нет целевой операции")
        if self.target_step_id is not None and self.target_step_id not in self.step_ids:
            raise ValueError(f"{self.test_id}: целевой операции {self.target_step_id} нет в процессе")
        return self


class GoldenDataset(BaseModel):
    examples: list[GoldenExample]
    sha256: str

    def by_id(self, test_id: str) -> GoldenExample:
        return next(e for e in self.examples if e.test_id == test_id)

    @property
    def positives(self) -> list[GoldenExample]:
        return [e for e in self.examples if e.kind == "positive"]

    @property
    def negatives(self) -> list[GoldenExample]:
        return [e for e in self.examples if e.kind == "negative"]

    @property
    def ambiguous(self) -> list[GoldenExample]:
        return [e for e in self.examples if e.kind == "ambiguous"]


def load_dataset(path: str | Path | None = None, *, kb_dir: str | Path | None = None) -> GoldenDataset:
    """Читает и проверяет датасет: уникальные ID, корректные процессы, существующие паттерны базы знаний."""
    path = Path(path or DATASET_PATH)
    raw = path.read_bytes()
    examples = [GoldenExample.model_validate(item) for item in json.loads(raw.decode("utf-8"))]
    ids = [e.test_id for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("test_id в датасете должны быть уникальными")

    known = {p.pattern_id for p in load_patterns(kb_dir or DEFAULT_KB_DIR)}
    for example in examples:
        used = set(example.expected_pattern_ids) | example.acceptable_patterns
        used |= {p for rule in example.unacceptable_recommendations for p in (rule.pattern_ids or [])}
        unknown = used - known
        if unknown:
            raise ValueError(f"{example.test_id}: неизвестные ID паттернов {sorted(unknown)}")
    return GoldenDataset(examples=examples, sha256=hashlib.sha256(raw).hexdigest())
