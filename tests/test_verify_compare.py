"""Проверка самого сравнения с эталоном (scripts.verify_live.compare) — что оно ловит выдумки и пропуски."""

from processmind.parsing.process_builder import build_process_spec
from scripts import demo_data
from scripts.verify_live import compare
from tests.test_extraction_service import perfect_raw_process


def _spec(raw=None):
    return build_process_spec(
        raw or perfect_raw_process(),
        source_name="x",
        source_type="image",  # без сверки с текстом — чтобы проверять именно compare
        source_bytes=b"x",
        model="m",
    ).spec


TRUTH = demo_data.truth_dict()


def test_perfect_result_has_no_problems():
    assert compare(_spec(), TRUTH) == []


def test_missing_operation_detected():
    raw = perfect_raw_process()
    raw.steps = raw.steps[:-1]
    problems = compare(_spec(raw), TRUTH)
    assert any("число операций" in p for p in problems)


def test_invented_duration_actor_and_type_detected():
    raw = perfect_raw_process()
    raw.steps[0] = raw.steps[0].model_copy(
        update={"duration_value": 10, "duration_unit": "minutes", "duration_quote": "10 мин", "execution_type": "manual"}
    )
    raw.steps[4] = raw.steps[4].model_copy(update={"actor": "Юрист"})
    problems = compare(_spec(raw), TRUTH)
    assert any("ВЫДУМАНА длительность" in p for p in problems)
    assert any("ВЫДУМАН участник" in p for p in problems)
    assert any("ВЫДУМАН тип" in p for p in problems)
