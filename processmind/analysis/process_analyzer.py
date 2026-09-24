"""Process Analyzer: эвристический (правило-ориентированный) анализ ProcessSpec.

MVP-этап: оценка возможностей автоматизации построена на явных правилах, а не на
вызове LLM — это осознанное решение для текущей итерации проекта. Интерфейс
`ProcessAnalyzer.analyze` рассчитан на то, чтобы позже правило-ориентированную
оценку можно было заменить или дополнить LLM-скорингом без изменения вызывающего кода.
"""

from __future__ import annotations

from processmind.models import AnalysisResult, AutomationOpportunity, ProcessSpec

LONG_STEP_THRESHOLD_MINUTES = 30.0


class ProcessAnalyzer:
    """Анализирует ProcessSpec и находит потенциальные возможности автоматизации."""

    def analyze(self, spec: ProcessSpec) -> AnalysisResult:
        opportunities: list[AutomationOpportunity] = []

        for step in spec.steps:
            score = 0
            reasons: list[str] = []

            if step.is_manual:
                score += 1
                reasons.append("шаг выполняется вручную")

            if step.pain_points:
                score += len(step.pain_points)
                reasons.append(f"выявлены проблемы: {', '.join(step.pain_points)}")

            if step.duration_minutes and step.duration_minutes > LONG_STEP_THRESHOLD_MINUTES:
                score += 1
                reasons.append(
                    f"длительность шага ({step.duration_minutes:.0f} мин) превышает порог "
                    f"{LONG_STEP_THRESHOLD_MINUTES:.0f} мин"
                )

            if score > 0:
                opportunities.append(
                    AutomationOpportunity(
                        step_id=step.id,
                        step_name=step.name,
                        score=score,
                        reasons=reasons,
                        suggestion=f"Рассмотреть автоматизацию шага «{step.name}»",
                    )
                )

        opportunities.sort(key=lambda o: o.score, reverse=True)

        manual_steps = sum(1 for s in spec.steps if s.is_manual)
        summary = (
            f"Процесс «{spec.name}» содержит {len(spec.steps)} шагов, из них "
            f"{manual_steps} ручных. Найдено {len(opportunities)} потенциальных "
            "возможностей автоматизации."
        )

        return AnalysisResult(process_id=spec.process_id, summary=summary, opportunities=opportunities)


def generate_tobe_spec(spec: ProcessSpec, analysis: AnalysisResult, top_n: int = 3) -> ProcessSpec:
    """Строит черновик TO-BE модели: помечает top-N приоритетных шагов как автоматизированные.

    MVP-этап: реальное LLM-предложение конкретного решения по автоматизации
    (RPA/интеграция/бот) будет добавлено на следующей итерации.
    """
    automate_ids = {o.step_id for o in analysis.opportunities[:top_n]}

    new_steps = []
    for step in spec.steps:
        if step.id in automate_ids:
            step = step.model_copy(
                update={
                    "is_manual": False,
                    "systems": list({*step.systems, "Automation Bot (TODO: LLM-предложение)"}),
                }
            )
        new_steps.append(step)

    return spec.model_copy(
        update={
            "process_id": f"{spec.process_id}-tobe",
            "name": f"{spec.name} (TO-BE, черновик)",
            "steps": new_steps,
        }
    )
