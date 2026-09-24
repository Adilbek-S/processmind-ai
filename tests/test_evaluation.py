from processmind.evaluation.pipeline import EvaluationPipeline
from processmind.models import ProcessSpec, ProcessStep


def test_evaluate_valid_spec_scores_full():
    spec = ProcessSpec(
        process_id="p1",
        name="Процесс",
        steps=[ProcessStep(id="step-1", name="Шаг 1"), ProcessStep(id="step-2", name="Шаг 2")],
    )
    result = EvaluationPipeline().evaluate_process_spec(spec)
    assert result.score == 1.0
    assert all(result.checks.values())


def test_evaluate_flags_duplicate_ids():
    spec = ProcessSpec(
        process_id="p1",
        name="Процесс",
        steps=[ProcessStep(id="dup", name="Шаг 1"), ProcessStep(id="dup", name="Шаг 2")],
    )
    result = EvaluationPipeline().evaluate_process_spec(spec)
    assert result.checks["step_ids_are_unique"] is False
    assert result.score < 1.0
