from processmind.analysis.process_analyzer import ProcessAnalyzer, generate_tobe_spec
from processmind.models import ProcessSpec, ProcessStep


def _sample_spec() -> ProcessSpec:
    return ProcessSpec(
        process_id="p1",
        name="Обработка заявки",
        steps=[
            ProcessStep(id="step-1", name="Приём заявки", is_manual=True, pain_points=["дублирование данных"]),
            ProcessStep(id="step-2", name="Автопроверка", is_manual=False),
            ProcessStep(id="step-3", name="Долгое согласование", is_manual=True, duration_minutes=45),
        ],
    )


def test_analyze_finds_opportunities_for_manual_steps():
    analysis = ProcessAnalyzer().analyze(_sample_spec())
    opportunity_ids = {o.step_id for o in analysis.opportunities}
    assert "step-1" in opportunity_ids
    assert "step-3" in opportunity_ids
    assert "step-2" not in opportunity_ids


def test_analyze_scores_pain_points_higher():
    analysis = ProcessAnalyzer().analyze(_sample_spec())
    by_id = {o.step_id: o for o in analysis.opportunities}
    assert by_id["step-1"].score >= 2


def test_generate_tobe_marks_top_opportunities_automated():
    spec = _sample_spec()
    analysis = ProcessAnalyzer().analyze(spec)
    tobe = generate_tobe_spec(spec, analysis, top_n=1)

    top_step_id = analysis.opportunities[0].step_id
    tobe_step = next(s for s in tobe.steps if s.id == top_step_id)

    assert tobe_step.is_manual is False
    assert tobe.process_id == "p1-tobe"
    # исходная спецификация не должна мутировать
    original_step = next(s for s in spec.steps if s.id == top_step_id)
    assert original_step.is_manual is True
