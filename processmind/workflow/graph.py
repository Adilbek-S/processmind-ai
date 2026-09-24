"""LangGraph Workflow: extract_process -> analyze -> generate_tobe -> evaluate.

MVP-этап: узел `node_extract_process` использует наивное построчное извлечение
шагов из текста вместо вызова LLM — осознанное решение для текущей итерации.
Граф уже целиком рабочий и детерминированный, поэтому его можно тестировать и
демонстрировать end-to-end без ключа OpenAI. На следующей итерации узел
извлечения будет заменён на вызов GPT-4o-mini со структурированным выводом
(Pydantic ProcessSpec).
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from processmind.analysis.process_analyzer import ProcessAnalyzer, generate_tobe_spec
from processmind.evaluation.pipeline import EvaluationPipeline
from processmind.models import AnalysisResult, ProcessSpec, ProcessStep

MAX_NAIVE_STEPS = 20


class ProcessMindState(TypedDict, total=False):
    raw_text: str
    process_spec: Optional[ProcessSpec]
    analysis: Optional[AnalysisResult]
    tobe_spec: Optional[ProcessSpec]
    evaluation: Optional[dict]
    errors: list[str]


def node_extract_process(state: ProcessMindState) -> dict:
    """Наивное построчное извлечение шагов процесса (заглушка под LLM-извлечение).

    Если в состоянии уже есть подтверждённый пользователем ProcessSpec (результат
    модуля распознавания), извлечение пропускается.
    """
    if state.get("process_spec") is not None:
        return {}
    text = state.get("raw_text", "")
    lines = [line.strip("-*\t ") for line in text.splitlines() if line.strip()]

    steps = [
        ProcessStep(id=f"step-{i + 1}", name=line[:80], is_manual=True)
        for i, line in enumerate(lines[:MAX_NAIVE_STEPS])
    ]
    if not steps:
        steps = [ProcessStep(id="step-1", name="Шаг не распознан (пустой ввод)", is_manual=True)]

    spec = ProcessSpec(
        process_id="demo-process",
        name="Демонстрационный процесс (черновик извлечения)",
        steps=steps,
    )
    return {"process_spec": spec}


def node_analyze(state: ProcessMindState) -> dict:
    analyzer = ProcessAnalyzer()
    analysis = analyzer.analyze(state["process_spec"])
    return {"analysis": analysis}


def node_generate_tobe(state: ProcessMindState) -> dict:
    tobe_spec = generate_tobe_spec(state["process_spec"], state["analysis"])
    return {"tobe_spec": tobe_spec}


def node_evaluate(state: ProcessMindState) -> dict:
    pipeline = EvaluationPipeline()
    result = pipeline.evaluate_process_spec(state["tobe_spec"])
    return {"evaluation": result.model_dump()}


def build_workflow():
    """Компилирует LangGraph-граф пайплайна анализа процесса."""
    graph = StateGraph(ProcessMindState)

    graph.add_node("extract_process", node_extract_process)
    graph.add_node("analyze", node_analyze)
    graph.add_node("generate_tobe", node_generate_tobe)
    graph.add_node("evaluate", node_evaluate)

    graph.set_entry_point("extract_process")
    graph.add_edge("extract_process", "analyze")
    graph.add_edge("analyze", "generate_tobe")
    graph.add_edge("generate_tobe", "evaluate")
    graph.add_edge("evaluate", END)

    return graph.compile()
