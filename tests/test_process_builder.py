"""Тесты нормализации сырого вывода LLM -> ProcessSpec и валидации Pydantic."""

import pytest
from pydantic import ValidationError

from processmind.models import ProcessStep
from processmind.parsing.process_builder import UNNAMED_PROCESS, build_process_spec
from processmind.parsing.schemas import ExtractionError, RawProcess, RawStep


def raw_step(order: int, name: str = "Операция", **kw) -> RawStep:
    base = dict(
        order=order,
        name=f"{name} {order}",
        description=None,
        actor=None,
        duration_value=None,
        duration_unit=None,
        duration_quote=None,
        execution_type="unknown",
        next_orders=[],
    )
    base.update(kw)
    return RawStep(**base)


def raw_process(steps, **kw) -> RawProcess:
    base = dict(name="Процесс", goal="Цель", participants=[], steps=steps, uncertainties=[])
    base.update(kw)
    return RawProcess(**base)


def build(raw, source_text=None, source_type="pdf"):
    return build_process_spec(
        raw, source_name="f.pdf", source_type=source_type, source_bytes=b"x", model="gpt-4o-mini", source_text=source_text
    )


# ---------- ProcessSpec (Pydantic) ----------


def test_process_step_rejects_negative_duration():
    with pytest.raises(ValidationError):
        ProcessStep(id="a", name="1", duration_minutes=-5)


def test_raw_step_rejects_unknown_execution_type():
    with pytest.raises(ValidationError):
        raw_step(1, execution_type="semi-automatic")


def test_raw_process_rejects_wrong_types():
    with pytest.raises(ValidationError):
        RawProcess.model_validate({"name": None, "goal": None, "participants": "Бухгалтер", "steps": [], "uncertainties": []})


def test_built_spec_has_consistent_links():
    steps = [raw_step(1, next_orders=[2]), raw_step(2, next_orders=[1, 3]), raw_step(3)]
    spec = build(raw_process(steps)).spec
    ids = {s.id for s in spec.steps}
    assert len(ids) == len(spec.steps)
    assert all(n in ids for s in spec.steps for n in s.next_steps)


# ---------- сборка ----------


def test_missing_values_stay_none():
    result = build(raw_process([raw_step(1)]))
    step = result.spec.steps[0]
    assert step.actor is None
    assert step.duration_minutes is None
    assert step.is_manual is None


def test_builds_ids_links_and_execution_type():
    steps = [
        raw_step(1, next_orders=[2], execution_type="manual"),
        raw_step(2, next_orders=[3], execution_type="automated"),
        raw_step(3),
    ]
    spec = build(raw_process(steps)).spec
    assert [s.id for s in spec.steps] == ["step-1", "step-2", "step-3"]
    assert spec.steps[0].next_steps == ["step-2"]
    assert spec.steps[2].next_steps == []
    assert [s.is_manual for s in spec.steps] == [True, False, None]


def test_steps_are_sorted_by_order():
    spec = build(raw_process([raw_step(2), raw_step(1)])).spec
    assert [s.name for s in spec.steps] == ["Операция 1", "Операция 2"]


def test_branching_links_are_kept():
    steps = [raw_step(1, next_orders=[2, 3]), raw_step(2), raw_step(3)]
    spec = build(raw_process(steps)).spec
    assert spec.steps[0].next_steps == ["step-2", "step-3"]


def test_linear_order_assumed_when_no_links_given_and_user_is_warned():
    result = build(raw_process([raw_step(1), raw_step(2), raw_step(3)]))
    assert [s.next_steps for s in result.spec.steps] == [["step-2"], ["step-3"], []]
    assert any("порядок следования" in w for w in result.warnings)


def test_link_to_nonexistent_operation_is_dropped_with_warning():
    result = build(raw_process([raw_step(1, next_orders=[2, 99]), raw_step(2)]))
    assert result.spec.steps[0].next_steps == ["step-2"]
    assert any("№99" in w for w in result.warnings)


def test_empty_steps_is_an_error_not_an_empty_spec():
    with pytest.raises(ExtractionError):
        build(raw_process([]))


def test_duplicate_orders_is_an_error():
    with pytest.raises(ExtractionError):
        build(raw_process([raw_step(1), raw_step(1)]))


def test_blank_step_name_is_an_error():
    with pytest.raises(ExtractionError):
        build(raw_process([raw_step(1).model_copy(update={"name": "  "})]))


def test_missing_process_name_and_goal_are_explicit():
    result = build(raw_process([raw_step(1)], name=None, goal=None))
    assert result.spec.name == UNNAMED_PROCESS
    assert result.spec.goal is None
    assert any("Название процесса" in w for w in result.warnings)
    assert any("Цель процесса" in w for w in result.warnings)


def test_uncertainties_become_warnings():
    result = build(raw_process([raw_step(1)], uncertainties=["блок 3 нечитаем"]))
    assert any("блок 3 нечитаем" in w for w in result.warnings)


def test_metadata_records_source():
    spec = build(raw_process([raw_step(1)]), source_type="image").spec
    assert spec.metadata["source_type"] == "image"
    assert spec.metadata["source_file"] == "f.pdf"
    assert spec.process_id.startswith("proc-")


# ---------- длительность ----------


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [(30, "minutes", 30), (2, "hours", 120), (3, "days", 4320), (90, "seconds", 1.5)],
)
def test_duration_converted_to_minutes(value, unit, expected):
    step = raw_step(1, duration_value=value, duration_unit=unit, duration_quote=f"{value} ед.")
    spec = build(raw_process([step]), source_text=f"Операция ({value} ед.)").spec
    assert spec.steps[0].duration_minutes == pytest.approx(expected)


def test_duration_not_present_in_source_text_is_dropped():
    step = raw_step(1, duration_value=15, duration_unit="minutes", duration_quote="15 минут")
    result = build(raw_process([step]), source_text="Менеджер проверяет заявку.")
    assert result.spec.steps[0].duration_minutes is None
    assert any("не найдена в тексте" in w for w in result.warnings)


def test_duration_without_quote_is_dropped():
    step = raw_step(1, duration_value=15, duration_unit="minutes", duration_quote=None)
    result = build(raw_process([step]), source_text="Менеджер проверяет заявку за 15 минут.")
    assert result.spec.steps[0].duration_minutes is None
    assert any("без цитаты" in w for w in result.warnings)


def test_duration_without_unit_is_dropped():
    step = raw_step(1, duration_value=15, duration_unit=None, duration_quote="15")
    assert build(raw_process([step]), source_text="15").spec.steps[0].duration_minutes is None


def test_quote_matching_ignores_case_and_whitespace():
    step = raw_step(1, duration_value=30, duration_unit="minutes", duration_quote="30  МИНУТ")
    spec = build(raw_process([step]), source_text="проверка (30 минут)").spec
    assert spec.steps[0].duration_minutes == 30


def test_image_duration_kept_but_flagged_for_review():
    step = raw_step(1, duration_value=30, duration_unit="minutes", duration_quote="30 мин")
    result = build(raw_process([step]), source_text=None, source_type="image")
    assert result.spec.steps[0].duration_minutes == 30
    assert any("проверьте" in w for w in result.warnings)


# ---------- участники ----------


def test_actor_not_in_text_is_dropped():
    step = raw_step(1, actor="Директор по маркетингу")
    result = build(raw_process([step]), source_text="Менеджер проверяет заявку.")
    assert result.spec.steps[0].actor is None
    assert result.spec.actors == []
    assert any("Директор по маркетингу" in w for w in result.warnings)


def test_actor_matches_russian_word_forms():
    step = raw_step(1, actor="Менеджер по закупкам")
    spec = build(raw_process([step]), source_text="Проверку выполняет менеджером отдела.").spec
    assert spec.steps[0].actor == "Менеджер по закупкам"


def test_hallucinated_participant_is_dropped_real_one_kept():
    result = build(
        raw_process([raw_step(1)], participants=["Бухгалтер", "Космонавт"]),
        source_text="Бухгалтер выставляет счёт.",
    )
    assert result.spec.actors == ["Бухгалтер"]
    assert any("Космонавт" in w for w in result.warnings)


def test_actors_are_unique_and_include_step_actors():
    steps = [raw_step(1, actor="Бухгалтер"), raw_step(2, actor="Бухгалтер")]
    spec = build(raw_process(steps, participants=["Бухгалтер"]), source_text="Бухгалтер").spec
    assert spec.actors == ["Бухгалтер"]
