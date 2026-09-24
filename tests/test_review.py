"""Тесты применения пользовательских правок (шаг подтверждения) и передачи spec в пайплайн."""

import pytest

from processmind.models import ProcessSpec, ProcessStep
from processmind.parsing.review import (
    EXEC_AUTOMATED,
    EXEC_MANUAL,
    EXEC_UNKNOWN,
    ReviewError,
    apply_edits,
    spec_to_rows,
)
from processmind.workflow.graph import build_workflow


def make_spec() -> ProcessSpec:
    return ProcessSpec(
        process_id="p",
        name="Процесс",
        actors=["Бухгалтер", "Аудитор"],
        steps=[
            ProcessStep(id="step-1", name="Первый", actor="Бухгалтер", duration_minutes=10, is_manual=True, next_steps=["step-2"]),
            ProcessStep(id="step-2", name="Второй", next_steps=[]),
        ],
    )


def test_rows_show_missing_values_as_empty():
    rows = spec_to_rows(make_spec())
    assert rows[0]["Участник"] == "Бухгалтер" and rows[0]["Следующие"] == "2"
    assert rows[1]["Участник"] is None
    assert rows[1]["Длительность, мин"] is None
    assert rows[1]["Тип выполнения"] == EXEC_UNKNOWN
    assert rows[1]["Следующие"] == "—"


def test_roundtrip_without_edits_changes_nothing():
    spec = make_spec()
    assert apply_edits(spec, spec_to_rows(spec)) == spec


def test_edits_applied_and_spec_stays_valid():
    spec = make_spec()
    rows = spec_to_rows(spec)
    rows[0].update({"Операция": "  Новое имя ", "Участник": "Директор", "Длительность, мин": 45, "Тип выполнения": EXEC_AUTOMATED})
    rows[1].update({"Длительность, мин": 5.5, "Тип выполнения": EXEC_MANUAL})

    edited = apply_edits(spec, rows)
    assert edited.steps[0].name == "Новое имя"
    assert edited.steps[0].actor == "Директор"
    assert edited.steps[0].duration_minutes == 45
    assert edited.steps[0].is_manual is False
    assert edited.steps[1].duration_minutes == 5.5 and edited.steps[1].is_manual is True
    assert edited.steps[0].next_steps == ["step-2"]  # связи не затронуты
    ProcessSpec.model_validate(edited.model_dump())


def test_clearing_cells_makes_values_absent():
    spec = make_spec()
    rows = spec_to_rows(spec)
    rows[0].update({"Участник": "  ", "Длительность, мин": float("nan"), "Тип выполнения": EXEC_UNKNOWN})
    step = apply_edits(spec, rows).steps[0]
    assert step.actor is None and step.duration_minutes is None and step.is_manual is None


def test_actors_list_follows_edits_but_keeps_participants_without_steps():
    spec = make_spec()  # «Аудитор» назван участником, но шагов у него нет
    rows = spec_to_rows(spec)
    rows[0]["Участник"] = "Директор"
    assert apply_edits(spec, rows).actors == ["Директор", "Аудитор"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Операция", "", "название"),
        ("Операция", None, "название"),
        ("Длительность, мин", -1, "отрицательной"),
        ("Длительность, мин", "abc", "числом"),
        ("Тип выполнения", "Полуавтоматически", "неизвестный тип"),
    ],
)
def test_invalid_edits_rejected(field, value, message):
    spec = make_spec()
    rows = spec_to_rows(spec)
    rows[0][field] = value
    with pytest.raises(ReviewError, match=message):
        apply_edits(spec, rows)


def test_row_count_mismatch_rejected():
    spec = make_spec()
    with pytest.raises(ReviewError):
        apply_edits(spec, spec_to_rows(spec)[:1])


def test_workflow_uses_confirmed_spec_instead_of_naive_extraction():
    spec = make_spec()
    result = build_workflow().invoke({"raw_text": "", "errors": [], "process_spec": spec})
    assert result["process_spec"] == spec
    assert result["analysis"].process_id == "p"
