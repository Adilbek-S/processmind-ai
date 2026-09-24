"""Тесты метрик оценок на вручную посчитанных числах."""

import math

import pytest

from evals.dataset import RecommendationRule
from evals.scoring import (
    PatternScore,
    PredictedRecommendation,
    ambiguous_correct,
    find_violations,
    hit_at_k,
    jaccard,
    macro_f1,
    mean_pairwise_jaccard,
    micro_f1,
    negative_correct,
    pattern_score,
    rate,
    recall_at_k,
    summarize,
)


def rec(step, pattern):
    return PredictedRecommendation(step, pattern)


# ---------- Recommendation F1 ----------


def test_perfect_match():
    s = pattern_score({"AP-1"}, {"AP-1"})
    assert (s.tp, s.fp, s.fn, s.precision, s.recall, s.f1) == (1, 0, 0, 1.0, 1.0, 1.0)


def test_extra_and_missing_patterns_hand_calculated():
    # предсказано {1, 3}, ожидалось {1, 2}: TP=1 (AP-1), FP=1 (AP-3), FN=1 (AP-2) -> P=R=0.5, F1=0.5
    s = pattern_score({"AP-1", "AP-3"}, {"AP-1", "AP-2"})
    assert (s.tp, s.fp, s.fn) == (1, 1, 1) and s.precision == s.recall == s.f1 == 0.5


def test_acceptable_patterns_are_neither_rewarded_nor_penalised():
    with_acceptable = pattern_score({"AP-1", "AP-2"}, {"AP-1"}, acceptable={"AP-2"})
    assert (with_acceptable.tp, with_acceptable.fp, with_acceptable.f1) == (1, 0, 1.0)
    without = pattern_score({"AP-1", "AP-2"}, {"AP-1"})
    assert without.fp == 1 and without.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)  # P=0.5, R=1 -> 2/3
    only_acceptable = pattern_score({"AP-2"}, {"AP-1"}, acceptable={"AP-2"})
    assert (only_acceptable.tp, only_acceptable.fp, only_acceptable.fn, only_acceptable.f1) == (0, 0, 1, 0.0)


@pytest.mark.parametrize("predicted", [set(), {"AP-9"}])
def test_nothing_or_only_wrong_gives_zero_f1(predicted):
    assert pattern_score(predicted, {"AP-1"}).f1 == 0.0


def test_macro_and_micro_f1_differ_as_expected():
    scores = [PatternScore(1, 0, 0, 1.0, 1.0, 1.0), PatternScore(1, 1, 1, 0.5, 0.5, 0.5)]
    assert macro_f1(scores) == pytest.approx(0.75)  # (1 + 0.5) / 2
    assert micro_f1(scores) == pytest.approx(2 / 3)  # TP=2, FP=1, FN=1 -> P=R=2/3
    assert macro_f1([]) == 0.0 and micro_f1([]) == 0.0


# ---------- Retrieval ----------


def test_hit_and_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    assert hit_at_k(retrieved, {"C", "D"}, 3) is True and recall_at_k(retrieved, {"C", "D"}, 3) == 0.5
    assert hit_at_k(retrieved, {"C", "D"}, 2) is False and recall_at_k(retrieved, {"C", "D"}, 2) == 0.0
    assert hit_at_k(retrieved, {"A"}, 1) and recall_at_k(retrieved, set(), 3) == 0.0


# ---------- нарушения и отрицательные примеры ----------


def test_violations_match_step_and_optionally_pattern():
    rules = [
        RecommendationRule(step_id="s1", pattern_ids=None, reason="любая рекомендация на s1 недопустима"),
        RecommendationRule(step_id="s2", pattern_ids=["AP-7"], reason="OCR здесь не нужен"),
    ]
    predicted = [rec("s1", "AP-1"), rec("s1", None), rec("s2", "AP-7"), rec("s2", "AP-3"), rec("s3", "AP-7")]
    hits = find_violations(predicted, rules)
    assert [(r.step_id, r.pattern_id) for r, _ in hits] == [("s1", "AP-1"), ("s1", None), ("s2", "AP-7")]


def test_negative_example_is_correct_only_without_recommendations():
    assert negative_correct([]) is True
    assert negative_correct([rec("s1", None)]) is False


def test_ambiguous_example_correctness():
    assert ambiguous_correct([], [], {"AP-1"}) is True  # отказ от рекомендации допустим
    assert ambiguous_correct([rec("s1", "AP-1")], [], {"AP-1"}) is True
    assert ambiguous_correct([rec("s1", "AP-9")], [], {"AP-1"}) is False  # паттерн вне допустимых
    assert ambiguous_correct([rec("s1", "AP-1")], ["нарушение"], {"AP-1"}) is False


# ---------- устойчивость и сводки ----------


def test_jaccard_and_pairwise_stability():
    assert jaccard({"a"}, {"a"}) == 1.0 and jaccard({"a"}, {"b"}) == 0.0 and jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    # пары: (1,2)=1, (1,3)=0, (2,3)=0 -> среднее 1/3
    assert mean_pairwise_jaccard([{"a"}, {"a"}, {"b"}]) == pytest.approx(1 / 3)
    assert mean_pairwise_jaccard([set(), set(), set()]) == 1.0 and mean_pairwise_jaccard([{"a"}]) == 1.0


def test_summarize_uses_nearest_rank_p95():
    s = summarize(list(range(1, 21)))  # 1..20: p95 = sorted[ceil(19) - 1] = 19-е значение
    assert (s.n, s.mean, s.median, s.p95, s.minimum, s.maximum, s.total) == (20, 10.5, 10.5, 19, 1, 20, 210)
    single = summarize([4.0])
    assert single.p95 == single.median == single.mean == 4.0
    assert summarize([]).n == 0 and math.isfinite(summarize([]).mean)


def test_rate():
    assert rate([True, False, True, True]) == 0.75 and rate([]) == 0.0
