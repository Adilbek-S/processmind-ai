"""Главный LangGraph workflow ProcessMind AI: анализ процесса и рекомендации по автоматизации.

    validate_process -> calculate_metrics -> retrieve_automation_patterns -> generate_recommendations
        -> validate_recommendations -> human_approval -> simulate_automation -> generate_report

Один граф с условными переходами и настоящим human-in-the-loop:
- валидацию, метрики и симуляцию выполняет MCP-сервер (отдельный процесс, stdio) через MCP-клиент;
- паттерны ищет RAG над ChromaDB (эмбеддинги OpenAI);
- рекомендации формирует LLM (GPT-4o-mini) по методике Skill `process-analysis`, которая загружается из
  `.claude/skills/process-analysis/SKILL.md` и становится системным промптом;
- `human_approval` вызывает `interrupt()`: граф останавливается с сохранением состояния в checkpointer и
  продолжается только командой `Command(resume=...)` с выбором пользователя.

Любой исход (ошибка валидации, сбой, отсутствие рекомендаций, пустой выбор, успех) заканчивается узлом
`generate_report` и структурированным `AnalysisReport`.

Состояние графа — только JSON-совместимые данные (словари/списки), чтобы его мог сохранять checkpointer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from openai import OpenAI
from pydantic import BaseModel

from processmind.analysis.recommendations import (
    LLMRecommendationSet,
    Recommendation,
    RecommendationError,
    build_llm_context,
    build_problem_signals,
    generate_recommendations,
    validate_recommendations,
)
from processmind.analysis.report import (
    AnalysisReport,
    ApprovedChange,
    PatternRef,
    ReportRecommendation,
    SkillRef,
    build_to_be,
)
from processmind.analysis.skill import Skill, SkillError, load_skill
from processmind.config import get_settings
from processmind.mcp.client import McpClientError, McpToolError, SyncMCPClient
from processmind.models import ProcessSpec
from processmind.parsing.llm import get_client
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import PatternMatch, build_query_from_process, get_pattern_by_id, search_automation_patterns
from processmind.workflow.approval import ApprovalError, parse_decision

PATTERNS_PER_QUERY = 3
MAX_STEP_QUERIES = 8
MCP_ERRORS = (McpToolError, McpClientError)

NODE_NAMES = (
    "validate_process",
    "calculate_metrics",
    "retrieve_automation_patterns",
    "generate_recommendations",
    "validate_recommendations",
    "human_approval",
    "simulate_automation",
    "generate_report",
)


class McpTools(Protocol):
    def validate_process(self, process: dict[str, Any]) -> dict[str, Any]: ...
    def calculate_process_metrics(self, process: dict[str, Any], runs_per_month: int | None = None) -> dict[str, Any]: ...
    def simulate_automation(self, process: dict[str, Any], automation: list[dict[str, Any]], runs_per_month: int | None = None) -> dict[str, Any]: ...


@dataclass
class WorkflowDeps:
    """Внешние зависимости workflow. В приложении — настоящие; тесты подставляют свои через те же интерфейсы."""

    mcp: McpTools
    retrieve: Callable[[str, int], list[PatternMatch]]
    get_pattern: Callable[[str], Any]
    llm_model: str
    llm_client: OpenAI | None = None  # None — клиент создаётся при вызове из OPENAI_API_KEY
    embedding_model: str | None = None
    skill: Skill | None = None  # None — Skill читается с диска при КАЖДОМ запуске
    skill_path: Path | None = None
    owned_mcp: SyncMCPClient | None = field(default=None, repr=False)

    def close(self) -> None:
        if self.owned_mcp is not None:
            self.owned_mcp.close()


def create_default_deps() -> WorkflowDeps:
    """Настоящие зависимости: MCP-сервер отдельным процессом, RAG над ChromaDB, OpenAI из .env."""
    settings = get_settings()
    mcp = SyncMCPClient().start()
    return WorkflowDeps(
        mcp=mcp,
        retrieve=lambda query, k: search_automation_patterns(query, top_k=k),
        get_pattern=get_pattern_by_id,
        llm_model=settings.openai_model,
        embedding_model=settings.embedding_model,
        owned_mcp=mcp,
    )


class AnalysisState(TypedDict, total=False):
    process: dict[str, Any]
    runs_per_month: int | None
    validation: dict[str, Any]
    metrics: dict[str, Any]
    patterns: list[dict[str, Any]]
    raw_recommendations: dict[str, Any]
    llm_notes: list[str]
    recommendations: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    approved: list[dict[str, Any]]
    simulation: dict[str, Any]
    tobe_metrics: dict[str, Any]
    skill_ref: dict[str, str]
    outcome: str  # invalid | failed | no_recommendations | no_selection | completed
    errors: list[str]
    warnings: list[str]
    report: dict[str, Any]


def _fail(state: AnalysisState, message: str) -> dict[str, Any]:
    return {"outcome": "failed", "errors": [*state.get("errors", []), message]}


def _spec(state: AnalysisState) -> ProcessSpec:
    return ProcessSpec.model_validate(state["process"])


def build_workflow(deps: WorkflowDeps, checkpointer: Any | None = None):
    """Компилирует граф. Checkpointer обязателен для пауз human-in-the-loop (по умолчанию — MemorySaver)."""

    # 1. validate_process ------------------------------------------------------------------------
    def validate_process(state: AnalysisState) -> dict[str, Any]:
        try:
            report = deps.mcp.validate_process(state["process"])
        except MCP_ERRORS as exc:
            return _fail(state, f"MCP validate_process: {exc}")
        if not report["valid"]:
            errors = [f"{e['path'] or 'процесс'}: {e['message']}" for e in report["errors"]]
            return {"validation": report, "outcome": "invalid", "errors": errors}
        return {"validation": report}

    # 2. calculate_metrics -----------------------------------------------------------------------
    def calculate_metrics(state: AnalysisState) -> dict[str, Any]:
        try:
            metrics = deps.mcp.calculate_process_metrics(state["process"], state.get("runs_per_month"))
        except MCP_ERRORS as exc:
            return _fail(state, f"MCP calculate_process_metrics: {exc}")
        return {"metrics": metrics}

    # 3. retrieve_automation_patterns ------------------------------------------------------------
    def retrieve_automation_patterns(state: AnalysisState) -> dict[str, Any]:
        spec = _spec(state)
        candidates = [s for s in spec.steps if s.is_manual is not False][:MAX_STEP_QUERIES]
        queries = [("process", build_query_from_process(spec))] + [(s.id, build_query_from_process(spec, s.id)) for s in candidates]

        found: dict[str, dict[str, Any]] = {}
        warnings = list(state.get("warnings", []))
        try:
            for label, query in queries:
                for match in deps.retrieve(query, PATTERNS_PER_QUERY):
                    entry = found.setdefault(
                        match.pattern_id,
                        {"pattern_id": match.pattern_id, "title": match.title, "source": match.source, "score": match.score, "matched_for": []},
                    )
                    entry["score"] = max(entry["score"], match.score)
                    if label not in entry["matched_for"]:
                        entry["matched_for"].append(label)
        except KnowledgeBaseError as exc:
            # Без базы знаний рекомендации возможны, но без ссылок на паттерны — об этом честно сообщаем.
            warnings.append(f"База знаний недоступна: {exc} Рекомендации сформированы без опоры на паттерны.")
            found = {}

        patterns = []
        for entry in sorted(found.values(), key=lambda e: e["score"], reverse=True):
            pattern = deps.get_pattern(entry["pattern_id"])
            entry["sections"] = pattern.sections if pattern is not None else {}
            patterns.append(entry)
        return {"patterns": patterns, "warnings": warnings}

    # 4. generate_recommendations ----------------------------------------------------------------
    def generate_recommendations_node(state: AnalysisState) -> dict[str, Any]:
        try:
            skill = deps.skill or load_skill(deps.skill_path)
            client = deps.llm_client or get_client(error_cls=RecommendationError)
            context = build_llm_context(_spec(state), state["validation"], state["metrics"], state.get("patterns", []))
            result = generate_recommendations(client, deps.llm_model, skill, context)
        except SkillError as exc:
            return _fail(state, f"Skill: {exc}")
        except RecommendationError as exc:
            return _fail(state, f"Рекомендации: {exc}")
        return {
            "raw_recommendations": result.model_dump(),
            "llm_notes": result.notes,
            "skill_ref": {"name": skill.name, "sha256": skill.sha256},
        }

    # 5. validate_recommendations ----------------------------------------------------------------
    def validate_recommendations_node(state: AnalysisState) -> dict[str, Any]:
        outcome = validate_recommendations(
            LLMRecommendationSet.model_validate(state["raw_recommendations"]),
            _spec(state),
            {p["pattern_id"]: p for p in state.get("patterns", [])},
        )
        update: dict[str, Any] = {
            "recommendations": [r.model_dump() for r in outcome.accepted],
            "rejected": [r.model_dump() for r in outcome.rejected],
        }
        if not outcome.accepted:
            update["outcome"] = "no_recommendations"
        return update

    # 6. human_approval --------------------------------------------------------------------------
    def human_approval(state: AnalysisState) -> dict[str, Any]:
        # interrupt() останавливает граф и отдаёт payload вызывающему. При возобновлении узел выполняется заново
        # с начала (до interrupt() в нём нет побочных эффектов), а уже полученные ответы отдаются по порядку.
        # Некорректное решение не роняет граф: узел повторно запрашивает решение, добавив текст ошибки, —
        # пауза сохраняется, и пользователь может исправить выбор.
        request: dict[str, Any] = {
            "type": "approval_request",
            "message": "Выберите рекомендации для включения в TO-BE и при необходимости уточните предполагаемую длительность.",
            "recommendations": state["recommendations"],
        }
        while True:
            decision = interrupt(request)
            try:
                approved = parse_decision(decision, state["recommendations"])
                break
            except ApprovalError as exc:
                request = {**request, "error": str(exc)}
        update: dict[str, Any] = {"approved": [a.model_dump() for a in approved]}
        if not approved:
            update["outcome"] = "no_selection"
        return update

    # 7. simulate_automation ---------------------------------------------------------------------
    def simulate_automation(state: AnalysisState) -> dict[str, Any]:
        automation = [{"step_id": a["step_id"], "new_duration_minutes": a["new_duration_minutes"]} for a in state["approved"]]
        approved = [ApprovedChange.model_validate(a) for a in state["approved"]]
        try:
            simulation = deps.mcp.simulate_automation(state["process"], automation, state.get("runs_per_month"))
            # Метрики TO-BE считает тот же MCP-инструмент, что и метрики AS-IS: показатели сопоставимы.
            tobe = build_to_be(_spec(state), approved).model_dump(mode="json")
            tobe_metrics = deps.mcp.calculate_process_metrics(tobe, state.get("runs_per_month"))
        except MCP_ERRORS as exc:
            return _fail(state, f"MCP simulate_automation: {exc}")
        return {"simulation": simulation, "tobe_metrics": tobe_metrics, "outcome": "completed"}

    # 8. generate_report -------------------------------------------------------------------------
    def generate_report(state: AnalysisState) -> dict[str, Any]:
        # Процесс может быть невалидным (именно тогда нужен отчёт с ошибками): ProcessSpec строим только при успехе.
        outcome = state.get("outcome", "completed")
        raw_process = state["process"] if isinstance(state.get("process"), dict) else {}
        try:
            spec = ProcessSpec.model_validate(raw_process)
        except ValueError:  # некорректный процесс: в отчёте остаются только ошибки
            spec = None
        approved = [ApprovedChange.model_validate(a) for a in state.get("approved", [])]
        offered = "approved" in state  # человек видел рекомендации и принял решение
        approved_by_id = {a.rec_id: a for a in approved}

        recommendations = []
        for raw in state.get("recommendations", []):
            change = approved_by_id.get(raw["rec_id"])
            decision = "not_offered" if not offered else ("approved" if change else "not_selected")
            recommendations.append(
                ReportRecommendation(
                    **Recommendation.model_validate(raw).model_dump(),
                    decision=decision,
                    approved_duration_minutes=change.new_duration_minutes if change else None,
                    duration_source=change.duration_source if change else None,
                )
            )

        warnings = list(state.get("warnings", []))
        for rec in recommendations:
            warnings.extend(f"{rec.rec_id}: {w}" for w in rec.warnings)
        if outcome == "no_recommendations":
            warnings.append("По методике анализа не найдено обоснованных рекомендаций по автоматизации.")
        metrics = state.get("metrics")
        if metrics:
            warnings.extend(metrics.get("warnings", []))
        simulation = state.get("simulation")
        if simulation:
            warnings.extend(simulation.get("warnings", []))

        report = AnalysisReport(
            process_id=str(raw_process.get("process_id") or "unknown"),
            process_name=str(raw_process.get("name") or "Без названия"),
            status=outcome,
            process=spec,
            errors=state.get("errors", []),
            warnings=list(dict.fromkeys(warnings)),
            validation=state.get("validation"),
            metrics=metrics,
            problems=build_problem_signals(spec, metrics) if spec is not None and metrics else [],
            patterns=[PatternRef(**{k: p[k] for k in ("pattern_id", "title", "source", "score", "matched_for")}) for p in state.get("patterns", [])],
            recommendations=recommendations,
            rejected_recommendations=state.get("rejected", []),
            llm_notes=state.get("llm_notes", []),
            approved=approved,
            simulation=simulation,
            tobe_metrics=state.get("tobe_metrics"),
            to_be=build_to_be(spec, approved) if outcome == "completed" and spec is not None else None,
            skill=SkillRef(**state["skill_ref"]) if state.get("skill_ref") else None,
            llm_model=deps.llm_model if state.get("raw_recommendations") else None,
            embedding_model=deps.embedding_model if state.get("patterns") else None,
        )
        return {"report": report.model_dump(mode="json")}

    # Сборка графа -------------------------------------------------------------------------------
    graph = StateGraph(AnalysisState)
    graph.add_node("validate_process", validate_process)
    graph.add_node("calculate_metrics", calculate_metrics)
    graph.add_node("retrieve_automation_patterns", retrieve_automation_patterns)
    graph.add_node("generate_recommendations", generate_recommendations_node)
    graph.add_node("validate_recommendations", validate_recommendations_node)
    graph.add_node("human_approval", human_approval)
    graph.add_node("simulate_automation", simulate_automation)
    graph.add_node("generate_report", generate_report)

    def stop_on_failure(next_node: str) -> Callable[[AnalysisState], str]:
        """Ошибка (invalid/failed) или пустой результат ведут сразу к отчёту, иначе — к следующему узлу."""
        return lambda state: "generate_report" if state.get("outcome") in ("invalid", "failed", "no_recommendations", "no_selection") else next_node

    graph.set_entry_point("validate_process")
    for source, target in (
        ("validate_process", "calculate_metrics"),
        ("calculate_metrics", "retrieve_automation_patterns"),
        ("generate_recommendations", "validate_recommendations"),
        ("validate_recommendations", "human_approval"),
        ("human_approval", "simulate_automation"),
    ):
        graph.add_conditional_edges(source, stop_on_failure(target), {target: target, "generate_report": "generate_report"})
    graph.add_edge("retrieve_automation_patterns", "generate_recommendations")
    graph.add_edge("simulate_automation", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


# ---------- запуск, пауза и продолжение ----------


class AnalysisRun(BaseModel):
    """Состояние запуска: либо ждёт решения пользователя (`awaiting_approval`), либо завершён с отчётом."""

    thread_id: str
    status: Literal["awaiting_approval", "finished"]
    approval_request: dict[str, Any] | None = None
    report: AnalysisReport | None = None


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _pending_approval(state) -> dict[str, Any] | None:
    """Payload ожидающего interrupt() или None. Признак паузы — interrupts задач, а не `state.next`:
    после повторного interrupt() в одном узле (перезапрос при некорректном решении) `next` пуст."""
    return next((i.value for task in state.tasks for i in task.interrupts), None)


def _snapshot(graph, thread_id: str) -> AnalysisRun:
    state = graph.get_state(_config(thread_id))
    payload = _pending_approval(state)
    if payload is not None:  # граф остановлен на interrupt()
        return AnalysisRun(thread_id=thread_id, status="awaiting_approval", approval_request=payload)
    return AnalysisRun(thread_id=thread_id, status="finished", report=AnalysisReport.model_validate(state.values["report"]))


def start_analysis(graph, spec: ProcessSpec | dict[str, Any], runs_per_month: int | None = None, thread_id: str | None = None) -> AnalysisRun:
    """Запускает анализ. Возвращает либо запрос подтверждения (граф на паузе), либо готовый отчёт."""
    thread_id = thread_id or uuid.uuid4().hex
    process = spec.model_dump(mode="json") if isinstance(spec, BaseModel) else spec
    graph.invoke({"process": process, "runs_per_month": runs_per_month, "errors": [], "warnings": []}, _config(thread_id))
    return _snapshot(graph, thread_id)


def resume_analysis(graph, thread_id: str, decision: dict[str, Any]) -> AnalysisRun:
    """Продолжает приостановленный анализ решением пользователя: {"selections": [{"rec_id": "R1", "new_duration_minutes": 5}]}.

    Некорректное решение не бросает исключение: запуск остаётся на паузе, а в `approval_request["error"]` —
    причина; нужно отправить исправленное решение.
    """
    if _pending_approval(graph.get_state(_config(thread_id))) is None:
        raise ApprovalStateError(f"Запуск {thread_id} не ожидает подтверждения")
    graph.invoke(Command(resume=decision), _config(thread_id))
    return _snapshot(graph, thread_id)


class ApprovalStateError(RuntimeError):
    """Запуск не находится на паузе (уже завершён или не существует)."""


def peek_state(graph, thread_id: str) -> dict[str, Any]:
    """Промежуточные результаты запуска (метрики, найденные паттерны и т.д.) — например, чтобы показать их во время паузы."""
    return dict(graph.get_state(_config(thread_id)).values)
