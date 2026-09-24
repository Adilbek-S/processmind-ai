"""Проверка РЕАЛЬНОЙ трассировки LangSmith: выполняет настоящий анализ и убеждается, что трейсы появились в проекте.

Ничего не имитируется: без LANGSMITH_API_KEY проверка завершается ошибкой и инструкцией по настройке.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from langsmith.run_helpers import trace as ls_trace

from processmind.observability import SETUP_INSTRUCTIONS, flush_tracing, tracing_status
from processmind.workflow.graph import NODE_NAMES, build_workflow, create_default_deps, resume_analysis, start_analysis

POLL_TIMEOUT_S = 120
POLL_INTERVAL_S = 5


class TracingUnavailable(RuntimeError):
    """LangSmith не настроен или недоступен (сообщение содержит инструкцию)."""


def _require_tracing() -> Any:
    status = tracing_status()
    if not status.enabled:
        raise TracingUnavailable(f"Трассировка LangSmith выключена: {status.reason}.\n\n{SETUP_INSTRUCTIONS}")
    return status


def run_traced_analysis(process: dict, runs_per_month: int = 120) -> dict[str, Any]:
    """Настоящий анализ (OpenAI, RAG, MCP) внутри корневого трейса; возвращает id трейса и краткий итог."""
    status = _require_tracing()
    deps = create_default_deps()
    try:
        graph = build_workflow(deps)
        with ls_trace("tracing_verification", run_type="chain", project_name=status.project, tags=["verification", "processmind"], metadata={"purpose": "проверка появления трейсов"}) as root:
            run = start_analysis(graph, process, runs_per_month)
            note = "пауза не наступила (нет рекомендаций или сбой)"
            if run.status == "awaiting_approval":
                rec = next((r for r in run.approval_request["recommendations"] if r["original_duration_minutes"] is not None), None)
                if rec is None:
                    run = resume_analysis(graph, run.thread_id, {"selections": []})
                    note = "выбор пуст: рекомендации без известной исходной длительности"
                else:
                    assumed = rec["new_duration_minutes"] if rec["new_duration_minutes"] is not None else rec["original_duration_minutes"] / 2
                    run = resume_analysis(graph, run.thread_id, {"selections": [{"rec_id": rec["rec_id"], "new_duration_minutes": assumed}]})
                    note = f"проверочный выбор {rec['rec_id']} с допущением {assumed:g} мин (только для проверки трейсов)"
            trace_id, root_id = str(root.trace_id), str(root.id)
    finally:
        deps.close()
    flush_tracing()
    return {"trace_id": trace_id, "root_id": root_id, "project": status.project, "status": run.report.status if run.report else run.status, "note": note}


def fetch_trace(trace_id: str, project: str, expected: set[str] = frozenset(NODE_NAMES)) -> list[Any]:
    """Ждёт появления runs трейса в LangSmith (приём асинхронный) и возвращает их."""
    from langsmith import Client

    client = Client()
    deadline, runs = time.time() + POLL_TIMEOUT_S, []
    while time.time() < deadline:
        runs = list(client.list_runs(project_name=project, trace_id=trace_id))
        if expected <= {r.name for r in runs}:
            return runs
        time.sleep(POLL_INTERVAL_S)
    return runs


def summarize_trace(runs: list[Any]) -> dict[str, Any]:
    names = {r.name for r in runs}
    types = Counter(r.run_type for r in runs)
    llm_runs = [r for r in runs if r.run_type == "llm"]
    tokens = {k: sum(getattr(r, k, 0) or 0 for r in llm_runs) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    root = next((r for r in runs if r.parent_run_id is None), None)
    duration = (root.end_time - root.start_time).total_seconds() if root and root.end_time else None
    checks = {
        "все 8 узлов LangGraph": set(NODE_NAMES) <= names,
        "LLM-вызов с токенами": bool(llm_runs) and tokens["total_tokens"] > 0,
        "RAG retrieval (retriever)": types.get("retriever", 0) > 0,
        "вызовы MCP (tool)": types.get("tool", 0) >= 2,
        "длительность корневого трейса": duration is not None,
    }
    return {
        "runs": len(runs), "by_type": dict(types), "tokens": tokens, "duration_s": duration,
        "error_runs": [r.name for r in runs if r.error], "names": sorted(names), "checks": checks, "ok": all(checks.values()),
    }


def verify(process: dict, runs_per_month: int = 120) -> dict[str, Any]:
    """Полная проверка: выполнить анализ, дождаться трейса в LangSmith, проверить состав."""
    started = run_traced_analysis(process, runs_per_month)
    runs = fetch_trace(started["trace_id"], started["project"])
    return {**started, **summarize_trace(runs)}
