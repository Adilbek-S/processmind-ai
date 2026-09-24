"""Тесты Golden Dataset: состав, структура и корректность."""

import collections
import copy
import json
import re

import pytest

from evals.dataset import DATASET_PATH, GoldenExample, load_dataset
from evals.dataset_source import EXAMPLES

REQUIRED_CATEGORIES = {"manual_detection", "applicability", "routing", "validation", "deduplication", "notifications", "no_automation", "ambiguous", "other_patterns"}


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_dataset_has_30_examples_with_unique_ids(dataset):
    assert len(dataset.examples) == 30
    assert [e.test_id for e in dataset.examples] == [f"GD-{i:03d}" for i in range(1, 31)]


def test_all_required_query_types_are_covered(dataset):
    counts = collections.Counter(e.category for e in dataset.examples)
    assert set(counts) == REQUIRED_CATEGORIES
    assert counts["no_automation"] >= 3 and counts["ambiguous"] >= 3  # отрицательные и неоднозначные случаи
    for category in ("routing", "validation", "deduplication", "notifications", "manual_detection"):
        assert counts[category] >= 3


def test_every_example_has_required_fields(dataset):
    for e in dataset.examples:
        assert e.test_id and e.description.strip() and e.question.strip() and e.process["steps"]
        assert isinstance(e.expected_pattern_ids, list) and isinstance(e.acceptable_recommendations, list)
        assert isinstance(e.unacceptable_recommendations, list)


def test_kinds_are_consistent(dataset):
    assert len(dataset.positives) + len(dataset.negatives) + len(dataset.ambiguous) == 30
    assert all(e.expected_pattern_ids and e.target_step_id for e in dataset.positives)
    assert all(not e.expected_pattern_ids and e.target_step_id is None for e in dataset.negatives)
    assert all(not e.expected_pattern_ids for e in dataset.ambiguous)
    assert {e.category for e in dataset.negatives} == {"no_automation"} and {e.category for e in dataset.ambiguous} == {"ambiguous"}


def test_positive_examples_do_not_forbid_their_own_expected_answer(dataset):
    for e in dataset.positives:
        for rule in e.unacceptable_recommendations:
            forbids_target = rule.step_id == e.target_step_id and (rule.pattern_ids is None or set(rule.pattern_ids) & set(e.expected_pattern_ids))
            assert not forbids_target, e.test_id


def test_negative_examples_have_no_manual_step_with_a_problem_signal(dataset):
    """Отрицательный пример не должен содержать признаков проблемы — иначе «нет рекомендаций» было бы спорным."""
    for e in dataset.negatives:
        assert not any(s.get("pain_points") for s in e.process["steps"]), e.test_id


def test_json_file_is_generated_from_the_source():
    assert json.loads(DATASET_PATH.read_text(encoding="utf-8")) == json.loads(json.dumps(EXAMPLES, ensure_ascii=False))


def test_dataset_uses_only_synthetic_content_without_normative_references(dataset):
    forbidden = re.compile(r"\bГОСТ\b|\bФЗ\b|\bСНиП\b|\bISO\b|№\s*\d|\bприказ\b|\bпостановлен|\bзакон\b", re.IGNORECASE)
    for e in dataset.examples:
        text = json.dumps(e.model_dump(), ensure_ascii=False)
        assert not forbidden.search(text), (e.test_id, forbidden.search(text).group())


def test_expected_patterns_cover_a_variety_of_kb_patterns(dataset):
    assert len({p for e in dataset.positives for p in e.expected_pattern_ids}) >= 8


# ---------- строгая валидация загрузчика ----------


def _write(tmp_path, items):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def test_loader_rejects_unknown_pattern_ids(tmp_path):
    items = copy.deepcopy(EXAMPLES[:1])
    items[0]["expected_pattern_ids"] = ["AP-999"]
    with pytest.raises(ValueError, match="AP-999"):
        load_dataset(_write(tmp_path, items))


def test_loader_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="уникальными"):
        load_dataset(_write(tmp_path, copy.deepcopy(EXAMPLES[:1]) * 2))


def test_loader_rejects_rules_for_missing_steps(tmp_path):
    items = copy.deepcopy(EXAMPLES[:1])
    items[0]["unacceptable_recommendations"] = [{"step_id": "zzz", "pattern_ids": None, "reason": "x"}]
    with pytest.raises(ValueError, match="несуществующую операцию"):
        load_dataset(_write(tmp_path, items))


def test_loader_rejects_invalid_process(tmp_path):
    items = copy.deepcopy(EXAMPLES[:1])
    items[0]["process"]["steps"][0]["duration_minutes"] = -5
    with pytest.raises(ValueError, match="некорректен"):
        load_dataset(_write(tmp_path, items))


def test_loader_rejects_inconsistent_kinds():
    base = copy.deepcopy(EXAMPLES[0])
    with pytest.raises(ValueError, match="positive-пример"):
        GoldenExample.model_validate({**base, "expected_pattern_ids": []})
    with pytest.raises(ValueError, match="не должно быть ожидаемых"):
        GoldenExample.model_validate({**base, "kind": "negative", "target_step_id": None})


def test_dataset_hash_changes_with_content(tmp_path):
    a = load_dataset(_write(tmp_path, copy.deepcopy(EXAMPLES[:2])))
    items = copy.deepcopy(EXAMPLES[:2])
    items[0]["description"] += " (изменено)"
    b = load_dataset(_write(tmp_path, items))
    assert a.sha256 != b.sha256
