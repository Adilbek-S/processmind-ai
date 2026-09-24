"""Тестовые двойники для RAG.

`FakeEmbedder` — детерминированный «мешок основ слов». Он ЛЕКСИЧЕСКИЙ и нужен только чтобы проверять
обвязку (индексация, дедупликация, метаданные, группировка результатов) без сети. Семантическое
качество поиска он не доказывает — его проверяет `scripts/verify_kb_live.py` с настоящими эмбеддингами.
"""

from __future__ import annotations

import math
import re
import zlib

DIM = 512


def fake_vector(text: str) -> list[float]:
    vector = [0.0] * DIM
    for word in re.findall(r"\w+", text.casefold()):
        vector[zlib.crc32(word[:5].encode("utf-8")) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeEmbedder:
    """Считает вызовы и тексты — чтобы тесты могли доказать, что повторная индексация ничего не пересчитывает."""

    def __init__(self, model: str = "fake-embedding-v1", fail_after_calls: int | None = None) -> None:
        self.model = model
        self.calls: list[list[str]] = []
        self._fail_after = fail_after_calls

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._fail_after is not None and len(self.calls) >= self._fail_after:
            raise RuntimeError("embedding service unavailable")
        self.calls.append(list(texts))
        return [fake_vector(t) for t in texts]


def pattern_markdown(pattern_id: str, title: str, **overrides: str) -> str:
    """Минимальный корректный паттерн; overrides меняют текст раздела по ключам problem/applicability/..."""
    body = {
        "problem": "Проблема заявки вручную.",
        "applicability": "Есть повторяющиеся заявки.",
        "approach": "Настроить автоматическое правило.",
        "effect": "Заявка обрабатывается быстрее.",
        "limitations": "Нужны исходные данные.",
    }
    body.update(overrides)
    return (
        f"# {title}\n\n**ID:** {pattern_id}\n\n"
        f"## Описание проблемы\n{body['problem']}\n\n"
        f"## Условия применимости\n{body['applicability']}\n\n"
        f"## Предлагаемый подход\n{body['approach']}\n\n"
        f"## Ожидаемый качественный эффект\n{body['effect']}\n\n"
        f"## Ограничения применения\n{body['limitations']}\n"
    )
