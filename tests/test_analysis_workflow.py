"""Тесты главного LangGraph workflow: успех, ошибки, пауза human-in-the-loop, продолжение.

MCP-сервер здесь настоящий (отдельный процесс, stdio); подменены только LLM и эмбеддинги (см. helpers_workflow).
"""

import copy

import pytest
from openai import APIConnectionError
import httpx

from processmind.analysis.recommendations import LLMRecommendationSet
from processmind.analysis.skill import load_skill
from processmind.mcp.client import SyncMCPClient
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import build_knowledge_base, get_pattern_by_id, search_automation_patterns
from processmind.workflow.graph import (
    NODE_NAMES,
    ApprovalStateError,
    WorkflowDeps,
    build_workflow,
    resume_analysis,
    start_analysis,
)
from tests.helpers_kb import FakeEmbedder
from tests.helpers_workflow import PROCESS, FakeLLMClient, RecordingMcp, context_of, honest_llm, rec

RUNS = 40


@pytest.fixture(scope="module")
def mcp_client():
    with SyncMCPClient() as client:
        yield client


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    store = tmp_path_factory.mktemp("chroma")
    embedder = FakeEmbedder()
    build_knowledge_base(embedder=embedder, persist_dir=store)
    return embedder, store


class Harness:
    """Собранный workflow + всё, что нужно тестам для проверки побочных эффектов."""

    def __init__(self, mcp_client, kb, llm=None, mcp_fail=None, retrieve_error=None, **deps_kw):
        embedder, store = kb
        self.mcp = RecordingMcp(mcp_client, mcp_fail)
        self.llm = llm or FakeLLMClient()
        self.queries: list[str] = []

        def retrieve(query, k):
            self.queries.append(query)
            if retrieve_error:
                raise retrieve_error
            return search_automation_patterns(query, top_k=k, embedder=embedder, persist_dir=store)

        deps = WorkflowDeps(
            mcp=self.mcp,
            retrieve=retrieve,
            get_pattern=lambda pid: get_pattern_by_id(pid, persist_dir=store, embedding_model=embedder.model),
            llm_model="gpt-4o-mini",
            llm_client=self.llm,
            embedding_model=embedder.model,
            **deps_kw,
        )
        self.graph = build_workflow(deps)

    def start(self, process=None, runs=RUNS, thread_id=None):
        return start_analysis(self.graph, process or PROCESS, runs, thread_id)

    def paused_at(self, thread_id):
        """Узел, на котором приостановлен запуск (по ожидающим interrupt), либо () если запуск не на паузе."""
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return tuple(task.name for task in state.tasks if task.interrupts)


@pytest.fixture
def harness(mcp_client, kb):
    return Harness(mcp_client, kb)


def by_step(request, step_id):
    return next(r for r in request["recommendations"] if r["step_id"] == step_id)


# ---------- структура ----------


def test_graph_contains_the_eight_required_nodes(harness):
    nodes = set(harness.graph.get_graph().nodes)
    assert set(NODE_NAMES) <= nodes and len(NODE_NAMES) == 8


# ---------- успешный сценарий: пауза -> решение -> продолжение ----------


def test_workflow_pauses_for_human_approval_before_simulation(harness):
    run = harness.start()

    assert run.status == "awaiting_approval" and run.report is None
    assert harness.paused_at(run.thread_id) == ("human_approval",)
    assert harness.graph.get_state({"configurable": {"thread_id": run.thread_id}}).next == ("human_approval",)
    # до решения пользователя выполнены только валидация и метрики; симуляции нет
    assert harness.mcp.calls == ["validate_process", "calculate_process_metrics"]
    assert len(harness.llm.calls) == 1

    request = run.approval_request
    assert request["type"] == "approval_request"
    assert [r["rec_id"] for r in request["recommendations"]] == ["R1", "R2", "R3", "R4"]
    assert {r["step_id"] for r in request["recommendations"]} == {"s1", "s2", "s4", "s5"}


def test_paused_workflow_does_not_advance_on_its_own(harness):
    run = harness.start()
    for _ in range(3):
        assert harness.paused_at(run.thread_id) == ("human_approval",)
    assert "simulate_automation" not in harness.mcp.calls


def test_resume_with_selection_runs_mcp_simulation_and_builds_report(harness):
    run = harness.start()
    req = run.approval_request
    r1, r2 = by_step(req, "s1")["rec_id"], by_step(req, "s2")["rec_id"]

    final = resume_analysis(harness.graph, run.thread_id, {"selections": [{"rec_id": r1}, {"rec_id": r2, "new_duration_minutes": 30}]})

    assert final.status == "finished" and final.approval_request is None
    # симуляция и метрики TO-BE — только после решения пользователя
    assert harness.mcp.calls == ["validate_process", "calculate_process_metrics", "simulate_automation", "calculate_process_metrics"]
    report = final.report
    assert report.status == "completed" and report.errors == []

    # s1: 30 -> 5 (значение LLM), s2: 120 -> 30 (значение пользователя), остальное как есть: 5 + 30 + 60 + 240 = 335
    sim = report.simulation
    assert sim["baseline_time_minutes"] == 450 and sim["proposed_time_minutes"] == 335
    assert sim["absolute_reduction_minutes"] == 115 and sim["reduction_percent"] == pytest.approx(25.5556, abs=1e-4)
    assert sim["baseline_manual_share"] == pytest.approx(0.6) and sim["proposed_manual_share"] == pytest.approx(0.2)
    assert sim["monthly_saving_minutes"] == 115 * RUNS == 4600
    assert sim["is_model_estimate"] is True
    assert report.metrics["total_time_minutes"] == 450 and report.metrics["monthly_effort_minutes"] == 450 * RUNS

    assert [(a.rec_id, a.new_duration_minutes, a.duration_source) for a in report.approved] == [(r1, 5, "llm"), (r2, 30, "user")]
    decisions = {r.rec_id: r.decision for r in report.recommendations}
    assert decisions[r1] == decisions[r2] == "approved" and set(decisions.values()) == {"approved", "not_selected"}
    shown = {r.rec_id: (r.approved_duration_minutes, r.duration_source) for r in report.recommendations if r.decision == "approved"}
    assert shown == {r1: (5, "llm"), r2: (30, "user")}  # в отчёте — то, что реально пошло в симуляцию
    tobe = {s.id: s for s in report.to_be.steps}
    assert tobe["s1"].is_manual is False and tobe["s1"].duration_minutes == 5 and tobe["s2"].duration_minutes == 30
    assert tobe["s4"].is_manual is True  # невыбранное не меняется
    assert "модельная оценка" in report.model_notice


def test_report_links_recommendations_to_found_patterns(harness):
    final = resume_analysis(harness.graph, (run := harness.start()).thread_id, {"selections": []})
    report = final.report
    found = {p.pattern_id for p in report.patterns}
    assert found and all(r.pattern_id in found for r in report.recommendations if r.kb_based)
    assert all(r.pattern_title and r.pattern_source for r in report.recommendations if r.kb_based)
    assert {"process", "s1"} <= {label for p in report.patterns for label in p.matched_for}
    assert harness.queries[0].startswith("Бизнес-процесс: Согласование закупки")


def test_empty_selection_finishes_without_simulation(harness):
    run = harness.start()
    final = resume_analysis(harness.graph, run.thread_id, {"selections": []})

    assert final.status == "finished" and final.report.status == "no_selection"
    assert "simulate_automation" not in harness.mcp.calls
    assert final.report.simulation is None and final.report.to_be is None
    assert {r.decision for r in final.report.recommendations} == {"not_selected"}


def test_parallel_runs_are_independent(harness):
    a, b = harness.start(), harness.start()
    assert a.thread_id != b.thread_id
    resume_analysis(harness.graph, a.thread_id, {"selections": []})
    assert harness.paused_at(b.thread_id) == ("human_approval",)  # решение по A не тронуло B
    assert resume_analysis(harness.graph, b.thread_id, {"selections": []}).report.status == "no_selection"


# ---------- некорректное решение пользователя ----------


def test_invalid_decision_keeps_workflow_paused_and_can_be_corrected(harness):
    run = harness.start()
    req = run.approval_request
    r1, r_s4 = by_step(req, "s1")["rec_id"], by_step(req, "s4")["rec_id"]

    bad_decisions = [
        ({"selections": [{"rec_id": "R99"}]}, "не было среди предложенных"),
        ({"selections": [{"rec_id": r1}, {"rec_id": r1}]}, "несколько раз"),
        ({"selections": [{"rec_id": r_s4, "new_duration_minutes": 5}]}, "не указана исходная длительность"),
        ({"selections": [{"rec_id": r1, "new_duration_minutes": -3}]}, "Некорректное решение"),
        ({"selections": "все"}, "Некорректное решение"),
    ]
    for decision, message in bad_decisions:
        again = resume_analysis(harness.graph, run.thread_id, decision)
        assert again.status == "awaiting_approval" and message in again.approval_request["error"]
        assert again.approval_request["recommendations"] == req["recommendations"]  # тот же запрос, плюс причина отказа
        assert harness.paused_at(run.thread_id) == ("human_approval",)
    assert "simulate_automation" not in harness.mcp.calls

    final = resume_analysis(harness.graph, run.thread_id, {"selections": [{"rec_id": r1}]})
    assert final.status == "finished" and final.report.status == "completed"
    assert harness.mcp.calls.count("simulate_automation") == 1 and final.report.approved[0].rec_id == r1


def test_resume_of_finished_or_unknown_run_is_rejected(harness):
    run = harness.start()
    resume_analysis(harness.graph, run.thread_id, {"selections": []})
    with pytest.raises(ApprovalStateError):
        resume_analysis(harness.graph, run.thread_id, {"selections": []})
    with pytest.raises(ApprovalStateError):
        resume_analysis(harness.graph, "no-such-thread", {"selections": []})


# ---------- ошибочные сценарии ----------


def test_invalid_process_returns_errors_and_stops_early(harness):
    bad = copy.deepcopy(PROCESS)
    bad["steps"][1]["id"] = "s1"  # дубликат ID
    bad["steps"][0]["duration_minutes"] = -5

    run = harness.start(bad)

    assert run.status == "finished" and run.report.status == "invalid"
    errors = " ".join(run.report.errors)
    assert "уже использован" in errors and "duration_minutes" in errors
    assert harness.mcp.calls == ["validate_process"]  # метрики, RAG, LLM и симуляция не запускались
    assert harness.queries == [] and harness.llm.calls == []
    assert run.report.recommendations == [] and run.report.simulation is None


def test_llm_recommendations_are_validated_and_rejected_ones_are_reported(mcp_client, kb):
    def responder(messages):
        pattern = context_of(messages)["knowledge_base_patterns"][0]["pattern_id"]
        return LLMRecommendationSet(
            recommendations=[
                rec("s1", pattern, 5),  # корректная
                rec("s99", pattern),  # несуществующая операция
                rec("s2", "AP-777"),  # паттерн, которого не было в результатах поиска
                rec("s5", pattern, expected_effect="Время сократится на 70%"),  # выдуманное количественное утверждение
                rec("s3", pattern),  # операция уже автоматизирована
                rec("s1", pattern),  # вторая рекомендация на ту же операцию
            ],
            notes=[],
        )

    h = Harness(mcp_client, kb, llm=FakeLLMClient(responder))
    run = h.start()

    assert [r["step_id"] for r in run.approval_request["recommendations"]] == ["s1"]
    final = resume_analysis(h.graph, run.thread_id, {"selections": [{"rec_id": "R1"}]})
    reasons = {r.recommendation.step_id + ":" + str(i): " ".join(r.reasons) for i, r in enumerate(final.report.rejected_recommendations)}
    joined = " | ".join(reasons.values())
    for expected in ("нет в процессе", "AP-777", "количественное утверждение", "уже автоматизирована", "уже есть рекомендация"):
        assert expected in joined
    assert len(final.report.rejected_recommendations) == 5


def test_all_recommendations_rejected_means_no_pause(mcp_client, kb):
    llm = FakeLLMClient(lambda m: LLMRecommendationSet(recommendations=[rec("nope", None), rec("s3", None)], notes=[]))
    h = Harness(mcp_client, kb, llm=llm)
    run = h.start()

    assert run.status == "finished" and run.report.status == "no_recommendations"
    assert len(run.report.rejected_recommendations) == 2
    assert "simulate_automation" not in h.mcp.calls
    assert any("не найдено обоснованных рекомендаций" in w for w in run.report.warnings)


def test_llm_returning_no_recommendations_is_a_valid_outcome(mcp_client, kb):
    llm = FakeLLMClient(lambda m: LLMRecommendationSet(recommendations=[], notes=["данных недостаточно"]))
    run = Harness(mcp_client, kb, llm=llm).start()
    assert run.report.status == "no_recommendations" and run.report.llm_notes == ["данных недостаточно"]


def test_llm_failure_ends_with_failed_report(mcp_client, kb):
    error = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    h = Harness(mcp_client, kb, llm=FakeLLMClient(error=error))
    run = h.start()
    assert run.status == "finished" and run.report.status == "failed"
    assert any("подключиться" in e for e in run.report.errors)
    assert h.mcp.calls == ["validate_process", "calculate_process_metrics"]


def test_missing_openai_key_ends_with_failed_report(mcp_client, monkeypatch):
    def no_key(**kwargs):
        raise kwargs["error_cls"]("OPENAI_API_KEY не задан")

    monkeypatch.setattr("processmind.workflow.graph.get_client", no_key)
    deps = WorkflowDeps(mcp=RecordingMcp(mcp_client), retrieve=lambda q, k: [], get_pattern=lambda p: None, llm_model="m", llm_client=None)
    run = start_analysis(build_workflow(deps), PROCESS, RUNS)
    assert run.report.status == "failed" and any("OPENAI_API_KEY" in e for e in run.report.errors)


@pytest.mark.parametrize("tool", ["validate_process", "calculate_process_metrics"])
def test_mcp_failure_ends_with_failed_report(mcp_client, kb, tool):
    h = Harness(mcp_client, kb, mcp_fail={tool})
    run = h.start()
    assert run.report.status == "failed" and any(tool in e for e in run.report.errors)
    assert h.llm.calls == []


def test_mcp_failure_during_simulation_is_reported(mcp_client, kb):
    h = Harness(mcp_client, kb, mcp_fail={"simulate_automation"})
    run = h.start()
    final = resume_analysis(h.graph, run.thread_id, {"selections": [{"rec_id": "R1"}]})
    assert final.report.status == "failed" and any("simulate_automation" in e for e in final.report.errors)
    assert final.report.simulation is None and final.report.to_be is None


def test_knowledge_base_unavailable_degrades_visibly(mcp_client, kb):
    def responder(messages):
        assert context_of(messages)["knowledge_base_patterns"] == []
        return LLMRecommendationSet(recommendations=[rec("s1", "AP-001", 5), rec("s2", None, 20)], notes=[])

    h = Harness(mcp_client, kb, llm=FakeLLMClient(responder), retrieve_error=KnowledgeBaseError("База знаний ещё не построена."))
    run = h.start()

    assert [r["step_id"] for r in run.approval_request["recommendations"]] == ["s2"]  # ссылка на паттерн без поиска — отклонена
    final = resume_analysis(h.graph, run.thread_id, {"selections": [{"rec_id": "R1"}]})
    assert any("База знаний недоступна" in w for w in final.report.warnings)
    assert final.report.recommendations[0].kb_based is False and final.report.patterns == []
    assert final.report.status == "completed"


# ---------- применение методики Skill ----------


def test_llm_system_prompt_is_the_skill_methodology(harness):
    harness.start()
    skill = load_skill()
    (call,) = harness.llm.calls
    system = call["messages"][0]
    assert system["role"] == "system" and system["content"] == skill.body
    assert "Запрет на выдумывание фактов" in system["content"] and "Правила выявления ручных операций" in system["content"]


def test_llm_receives_process_metrics_problems_and_patterns(harness):
    harness.start()
    context = context_of(harness.llm.calls[0]["messages"])

    assert [s["id"] for s in context["process"]["steps"]] == ["s1", "s2", "s3", "s4", "s5"]
    assert context["metrics"]["total_time_minutes"] == 450  # метрики рассчитал MCP-сервер, а не LLM
    kinds = {(s["step_id"], s["kind"]) for s in context["problem_signals"]}
    assert {("s1", "manual_step"), ("s4", "missing_duration"), ("s5", "unknown_type"), ("s2", "long_step")} <= kinds
    assert ("s3", "manual_step") not in kinds
    assert context["knowledge_base_patterns"] and all(p["sections"] for p in context["knowledge_base_patterns"])
    assert harness.llm.calls[0]["response_format"] is LLMRecommendationSet


def test_skill_is_read_from_disk_on_every_run_and_recorded_in_report(mcp_client, kb, tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: test-skill\ndescription: тест\n---\n# Методика A\nМАРКЕР-ПЕРВЫЙ\n", encoding="utf-8")
    h = Harness(mcp_client, kb, skill_path=skill_file)

    first = h.start()
    skill_file.write_text("---\nname: test-skill\ndescription: тест\n---\n# Методика B\nМАРКЕР-ВТОРОЙ\n", encoding="utf-8")
    second = h.start()

    assert "МАРКЕР-ПЕРВЫЙ" in h.llm.calls[0]["messages"][0]["content"]
    assert "МАРКЕР-ВТОРОЙ" in h.llm.calls[1]["messages"][0]["content"]  # правка методики меняет поведение без перезапуска
    r1 = resume_analysis(h.graph, first.thread_id, {"selections": []}).report
    r2 = resume_analysis(h.graph, second.thread_id, {"selections": []}).report
    assert r1.skill.name == "test-skill" and r1.skill.sha256 != r2.skill.sha256 == load_skill(skill_file).sha256


def test_broken_skill_file_fails_visibly_instead_of_running_without_methodology(mcp_client, kb, tmp_path):
    bad = tmp_path / "SKILL.md"
    bad.write_text("без frontmatter", encoding="utf-8")
    h = Harness(mcp_client, kb, skill_path=bad)
    run = h.start()
    assert run.report.status == "failed" and any("Skill" in e for e in run.report.errors) and h.llm.calls == []
