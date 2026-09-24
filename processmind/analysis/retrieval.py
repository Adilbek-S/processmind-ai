"""Поиск паттернов автоматизации для процесса: общий код workflow и автоматизированных оценок."""

from __future__ import annotations

from typing import Any, Callable

from processmind.models import ProcessSpec
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import PatternMatch, build_query_from_process

PATTERNS_PER_QUERY = 3
MAX_STEP_QUERIES = 8


def retrieve_patterns_for_process(
    spec: ProcessSpec,
    retrieve: Callable[[str, int], list[PatternMatch]],
    get_pattern: Callable[[str], Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """Паттерны для всего процесса и для каждой операции-кандидата (не автоматизированной).

    Возвращает (паттерны с полными разделами по убыванию сходства, предупреждение либо None). Если база знаний
    недоступна, паттернов нет, а предупреждение объясняет причину — без базы рекомендации возможны, но без ссылок.
    """
    candidates = [s for s in spec.steps if s.is_manual is not False][:MAX_STEP_QUERIES]
    queries = [("process", build_query_from_process(spec))] + [(s.id, build_query_from_process(spec, s.id)) for s in candidates]

    found: dict[str, dict[str, Any]] = {}
    try:
        for label, query in queries:
            for match in retrieve(query, PATTERNS_PER_QUERY):
                entry = found.setdefault(
                    match.pattern_id,
                    {"pattern_id": match.pattern_id, "title": match.title, "source": match.source, "score": match.score, "matched_for": []},
                )
                entry["score"] = max(entry["score"], match.score)
                if label not in entry["matched_for"]:
                    entry["matched_for"].append(label)
    except KnowledgeBaseError as exc:
        return [], f"База знаний недоступна: {exc} Рекомендации сформированы без опоры на паттерны."

    patterns = []
    for entry in sorted(found.values(), key=lambda e: e["score"], reverse=True):
        pattern = get_pattern(entry["pattern_id"])
        entry["sections"] = pattern.sections if pattern is not None else {}
        patterns.append(entry)
    return patterns, None
