"""Тесты трассировки LangSmith: конфигурация и инструментация (на локальном двойнике LangSmith API).

Настоящий SDK отправляет runs на двойник, поэтому проверяется то, что реально уйдёт в LangSmith:
имена и типы runs, вложенность, токены, длительности, ошибки. Облачный LangSmith это не проверяет —
для него есть `python -m evals.verify_tracing` (нужен LANGSMITH_API_KEY).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from openai import APIConnectionError

from processmind.analysis.recommendations import LLMRecommendationSet
from processmind.config import Settings
from processmind.mcp.client import McpToolError, SyncMCPClient
from processmind.observability import configure_tracing, resolve_tracing
from processmind.parsing.llm import LLMParams, get_client, request_structured_ex
from processmind.rag.embeddings import OpenAIEmbedder
from processmind.workflow.graph import NODE_NAMES, resume_analysis
from tests.helpers_kb import FakeEmbedder
from tests.helpers_langsmith import LangSmithDouble
from tests.helpers_workflow import FakeLLMClient
from tests.test_analysis_workflow import Harness, kb, mcp_client  # noqa: F401 (фикстуры)
from tests.test_embeddings_http import server as embeddings_server  # noqa: F401 (фикстура)

OFF = Settings(_env_file=None, langsmith_tracing=False)
_TRACING_ENV = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT")


@pytest.fixture(autouse=True)
def clean_tracing_env(monkeypatch):
    """configure_tracing экспортирует переменные окружения процесса; тесты не должны зависеть от порядка выполнения."""
    for name in _TRACING_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def ls():
    double = LangSmithDouble()
    double.start_tracing()
    yield double
    configure_tracing(OFF, force=True)  # не оставляем трассировку включённой для остальных тестов
    double.close()


# ---------- конфигурация ----------


@pytest.mark.parametrize(
    ("kwargs", "enabled", "reason"),
    [
        ({}, False, "выключена"),
        ({"langsmith_tracing": True}, False, "LANGSMITH_API_KEY"),
        ({"langsmith_tracing": True, "langsmith_api_key": "k"}, True, None),
        ({"langchain_tracing_v2": True, "langchain_api_key": "k"}, True, None),  # прежние имена переменных
        ({"langchain_tracing_v2": True, "langsmith_tracing": False, "langsmith_api_key": "k"}, False, "выключена"),  # новое имя главнее
        ({"langsmith_tracing": True, "langchain_api_key": "legacy-key"}, True, None),
    ],
)
def test_tracing_resolution_from_settings(kwargs, enabled, reason):
    status = resolve_tracing(Settings(_env_file=None, **kwargs))
    assert status.enabled is enabled
    assert (reason is None and status.reason is None) or reason in status.reason


def test_project_and_endpoint_come_from_settings():
    status = resolve_tracing(Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="k", langsmith_project="my-proj", langsmith_endpoint="https://eu.api.smith.langchain.com"))
    assert (status.project, status.endpoint) == ("my-proj", "https://eu.api.smith.langchain.com")
    assert resolve_tracing(Settings(_env_file=None)).project == "processmind-ai"  # значение по умолчанию


def test_configure_exports_environment_for_the_sdk(monkeypatch):
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(name, raising=False)
    try:
        status = configure_tracing(Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="secret", langsmith_project="p", langsmith_endpoint="http://127.0.0.1:9"), force=True)
        import os

        assert status.enabled and os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_PROJECT"] == "p"
        monkeypatch.delenv("LANGSMITH_API_KEY")  # ключ, экспортированный первым вызовом, убираем
        configure_tracing(Settings(_env_file=None, langsmith_tracing=True), force=True)  # ключа нет -> трассировка выключается
        assert os.environ["LANGSMITH_TRACING"] == "false"
    finally:
        configure_tracing(OFF, force=True)


def test_disabled_tracing_sends_nothing_and_app_still_works(mcp_client, kb):  # noqa: F811
    double = LangSmithDouble()
    try:
        configure_tracing(Settings(_env_file=None, langsmith_tracing=False, langsmith_endpoint=double.url), force=True)
        h = Harness(mcp_client, kb)
        run = h.start()
        assert run.status == "awaiting_approval" and double.runs() == []
    finally:
        double.close()


# ---------- инструментация workflow ----------


def test_workflow_trace_contains_graph_nodes_mcp_calls_and_retrieval(ls, mcp_client, kb):  # noqa: F811
    h = Harness(mcp_client, kb)
    run = h.start()
    resume_analysis(h.graph, run.thread_id, {"selections": [{"rec_id": "R1"}]})

    names = ls.names()
    assert set(NODE_NAMES) <= names  # все 8 этапов LangGraph
    assert {"process_analysis", "process_analysis.resume", "workflow_outcome"} <= names

    tools = {r["name"] for r in ls.runs() if r.get("run_type") == "tool"}
    assert tools == {"mcp.validate_process", "mcp.calculate_process_metrics", "mcp.simulate_automation"}
    for tool in (r for r in ls.runs() if r.get("run_type") == "tool"):
        assert {"validate_process", "calculate_metrics", "simulate_automation"} & set(ls.ancestors(tool))  # вложены в узел графа
        assert tool["inputs"]["tool"] == tool["name"].removeprefix("mcp.") and "start_time" in tool and "end_time" in tool

    retrievals = ls.by_name("rag.search_automation_patterns")
    assert len(retrievals) >= 2 and all(r["run_type"] == "retriever" for r in retrievals)
    assert all("retrieve_automation_patterns" in ls.ancestors(r) for r in retrievals)
    docs = retrievals[0]["outputs"]["documents"]
    assert docs and {"pattern_id", "source", "section", "chunk_id", "score"} <= set(docs[0]["metadata"])
    assert retrievals[0]["inputs"].keys() == {"query", "top_k"}  # без эмбеддера и путей


def test_root_runs_carry_thread_and_process_metadata(ls, mcp_client, kb):  # noqa: F811
    h = Harness(mcp_client, kb)
    run = h.start()
    root = ls.by_name("process_analysis")[0]
    metadata = root["extra"]["metadata"]
    assert metadata["thread_id"] == run.thread_id and metadata["process_id"] == "demo" and metadata["runs_per_month"] == 40
    assert "processmind" in root["tags"]
    assert root["session_name"] == "processmind-test"  # проект LangSmith берётся из настроек


def test_durations_are_recorded_for_every_finished_run(ls, mcp_client, kb):  # noqa: F811
    h = Harness(mcp_client, kb)
    run = h.start()
    resume_analysis(h.graph, run.thread_id, {"selections": []})
    finished = [r for r in ls.runs() if r["name"] in NODE_NAMES and r["name"] != "human_approval"]
    assert finished and all(r["start_time"] < r["end_time"] for r in finished)


# ---------- LLM, токены ----------


class _OpenAIDouble:
    """Минимальный /chat/completions с usage: проверяем, что токены попадают в трейс."""

    def __init__(self, monkeypatch):
        payload = json.dumps(
            {
                "id": "x", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({"recommendations": [], "notes": ["n"]}), "refusal": None}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
            }
        ).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{self.httpd.server_port}/v1")


def test_llm_call_is_traced_with_token_usage_and_measured(ls, monkeypatch):
    double = _OpenAIDouble(monkeypatch)
    try:
        client = get_client(Settings(_env_file=None, openai_api_key="sk-test"))
        result, usage = request_structured_ex(client, "gpt-4o-mini", [{"role": "user", "content": "привет"}], LLMRecommendationSet, params=LLMParams(temperature=0.0, top_p=1.0, max_tokens=500, seed=7))
    finally:
        double.httpd.shutdown()

    assert result.notes == ["n"] and (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (123, 45, 168) and usage.latency_s > 0
    llm_runs = [r for r in ls.runs() if r.get("run_type") == "llm"]
    assert len(llm_runs) == 1
    metadata = llm_runs[0]["extra"]["metadata"]
    # то, что LangSmith показывает в интерфейсе: модель, параметры генерации и токены
    assert metadata["ls_model_name"] == "gpt-4o-mini" and metadata["ls_temperature"] == 0.0 and metadata["ls_max_tokens"] == 500
    assert metadata["ls_invocation_params"]["top_p"] == 1.0 and metadata["ls_invocation_params"]["seed"] == 7
    tokens = metadata["usage_metadata"]
    assert (tokens["input_tokens"], tokens["output_tokens"], tokens["total_tokens"]) == (123, 45, 168)


# ---------- ошибки ----------


def test_mcp_tool_errors_are_recorded_as_error_runs(ls, mcp_client):  # noqa: F811
    with pytest.raises(McpToolError):
        mcp_client.call_tool("no_such_tool", {})
    (run,) = ls.by_name("mcp.no_such_tool")
    assert run["run_type"] == "tool" and run["error"] and "McpToolError" in run["error"]


def test_workflow_failure_is_recorded_as_error_run(ls, mcp_client, kb):  # noqa: F811
    error = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    h = Harness(mcp_client, kb, llm=FakeLLMClient(error=error))
    run = h.start()
    assert run.report.status == "failed"
    (outcome,) = ls.by_name("workflow_outcome")
    assert outcome["error"] and "подключиться" in outcome["error"] and "outcome:failed" in outcome["tags"]


def test_successful_and_invalid_outcomes_are_not_errors(ls, mcp_client, kb):  # noqa: F811
    h = Harness(mcp_client, kb)
    run = h.start()
    resume_analysis(h.graph, run.thread_id, {"selections": []})
    (outcome,) = ls.by_name("workflow_outcome")
    assert not outcome.get("error") and "outcome:no_selection" in outcome["tags"]


# ---------- эмбеддинги и токены ----------


def test_embeddings_run_reports_tokens(ls, embeddings_server):  # noqa: F811
    from openai import OpenAI

    embedder = OpenAIEmbedder(OpenAI(api_key="sk-test", max_retries=0), "text-embedding-3-small")
    embedder.embed(["один", "два"])
    assert embedder.tokens_used == 1  # usage.total_tokens из ответа API
    (run,) = ls.by_name("openai.embeddings")
    assert run["run_type"] == "embedding" and run["inputs"] == {"texts": 2}
    assert run["outputs"]["vectors"] == 2 and "tokens" in json.dumps(run)
