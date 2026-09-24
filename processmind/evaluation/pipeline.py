"""Evaluation Pipeline: структурные проверки качества ProcessSpec.

MVP-этап: реализованы простые структурные (rule-based) проверки. На следующей
итерации сюда добавится LLM-as-judge оценка и трассировка прогонов через
LangSmith (датасеты синтетических процессов, метрики точности извлечения и
релевантности предложений по автоматизации).
"""

from __future__ import annotations

from processmind.models import EvaluationResult, ProcessSpec


class EvaluationPipeline:
    """Проверяет структурную корректность и полноту ProcessSpec."""

    def evaluate_process_spec(self, spec: ProcessSpec) -> EvaluationResult:
        checks = {
            "has_steps": len(spec.steps) > 0,
            "steps_have_names": all(bool(s.name) for s in spec.steps),
            "step_ids_are_unique": len({s.id for s in spec.steps}) == len(spec.steps),
        }
        score = sum(checks.values()) / len(checks) if checks else 0.0

        return EvaluationResult(
            target_id=spec.process_id,
            checks=checks,
            score=score,
            notes=(
                "Rule-based структурная оценка (MVP). LLM-as-judge и LangSmith-датасеты "
                "будут добавлены на следующей итерации."
            ),
        )
