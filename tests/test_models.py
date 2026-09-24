from processmind.models import ProcessSpec, ProcessStep


def test_process_step_defaults():
    step = ProcessStep(id="step-1", name="Проверка заявки")
    assert step.is_manual is None  # тип выполнения не указан — не выдумываем
    assert step.duration_minutes is None
    assert step.actor is None
    assert step.inputs == []
    assert step.pain_points == []


def test_process_spec_holds_steps():
    spec = ProcessSpec(
        process_id="p1",
        name="Обработка заявки",
        steps=[ProcessStep(id="step-1", name="Приём заявки")],
    )
    assert spec.name == "Обработка заявки"
    assert len(spec.steps) == 1
    assert spec.steps[0].id == "step-1"
