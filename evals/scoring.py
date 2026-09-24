"""Метрики оценки: чистые функции без обращений к сети (проверяются юнит-тестами на вручную посчитанных числах).

Определения
-----------
Retrieval Hit@3 — доля positive-примеров, у которых хотя бы один ожидаемый паттерн вошёл в Top-3 поиска по запросу
для целевой операции. Recall@3 — средняя доля ожидаемых паттернов, найденных в Top-3.

Recommendation F1 (по паттернам, на примере): предсказанное множество P, ожидаемое E, допустимые сверх ожидаемых A.
    TP = |P ∩ E|,  FP = |P − E − A|,  FN = |E − P|
    precision = TP/(TP+FP), recall = TP/|E|, F1 = 2·P·R/(P+R)  (0, если TP = 0)
Допустимые паттерны A не поощряются и не штрафуются. Macro-F1 — среднее F1 по positive-примерам; micro-F1 — по суммам TP/FP/FN.

Отрицательные примеры: корректны, если предложено 0 рекомендаций. Неоднозначные: корректны, если нет недопустимых
рекомендаций и все предложенные паттерны допустимы.

Процентиль p95 — метод ближайшего ранга: sorted[ceil(0.95·n) − 1].
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from evals.dataset import RecommendationRule


@dataclass(frozen=True)
class PredictedRecommendation:
    step_id: str
    pattern_id: str | None


@dataclass(frozen=True)
class PatternScore:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def pattern_score(predicted: Iterable[str], expected: Iterable[str], acceptable: Iterable[str] = ()) -> PatternScore:
    p, e, a = set(predicted), set(expected), set(acceptable) - set(expected)
    tp, fp, fn = len(p & e), len(p - e - a), len(e - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(e) if e else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PatternScore(tp, fp, fn, precision, recall, f1)


def macro_f1(scores: Sequence[PatternScore]) -> float:
    return statistics.fmean(s.f1 for s in scores) if scores else 0.0


def micro_f1(scores: Sequence[PatternScore]) -> float:
    tp, fp, fn = (sum(getattr(s, k) for s in scores) for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def hit_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int = 3) -> bool:
    return bool(set(retrieved[:k]) & set(expected))


def recall_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int = 3) -> float:
    e = set(expected)
    return len(set(retrieved[:k]) & e) / len(e) if e else 0.0


def find_violations(predicted: Sequence[PredictedRecommendation], rules: Sequence[RecommendationRule]) -> list[tuple[PredictedRecommendation, RecommendationRule]]:
    """Предсказанные рекомендации, попавшие под правило «недопустимо» (операция совпала и паттерн входит в список либо список пуст)."""
    hits = []
    for rec in predicted:
        for rule in rules:
            if rec.step_id == rule.step_id and (rule.pattern_ids is None or rec.pattern_id in rule.pattern_ids):
                hits.append((rec, rule))
                break
    return hits


def negative_correct(predicted: Sequence[PredictedRecommendation]) -> bool:
    return len(predicted) == 0


def ambiguous_correct(predicted: Sequence[PredictedRecommendation], violations: Sequence, allowed_patterns: Iterable[str]) -> bool:
    allowed = set(allowed_patterns)
    return not violations and all(r.pattern_id is None or r.pattern_id in allowed for r in predicted)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def mean_pairwise_jaccard(sets: Sequence[Iterable[str]]) -> float:
    """Устойчивость: средняя схожесть множеств паттернов между повторными запусками одного примера."""
    pairs = [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return statistics.fmean(jaccard(sets[i], sets[j]) for i, j in pairs) if pairs else 1.0


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    median: float
    p95: float
    minimum: float
    maximum: float
    total: float


def summarize(values: Sequence[float]) -> Summary:
    if not values:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return Summary(len(ordered), statistics.fmean(ordered), statistics.median(ordered), p95, ordered[0], ordered[-1], math.fsum(ordered))


def rate(flags: Sequence[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0
