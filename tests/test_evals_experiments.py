"""Тесты исполнителя оценок (A/B), агрегации, правила выбора temperature, отчёта и защиты от «имитации».

Настоящими остаются MCP-сервер (stdio), поиск по ChromaDB, программная валидация и Skill; подменены LLM и эмбеддинги.
Тесты проверяют логику оценок, а не качество модели: реальные числа даёт `python -m evals.run_all`.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from evals import experiments, report, run_all
from evals.dataset import load_dataset
from evals.experiments import aggregate_config, case_outcome, run_temperature_experiment
from evals.runner import CaseResult, EvalCache, EvalContext, LLMStats, RecView, execute_case, kb_fingerprint
from processmind.analysis.recommendations import LLMRecommendationSet
from processmind.analysis.skill import load_skill
from processmind.config import Settings
from processmind.mcp.client import SyncMCPClient
from processmind.parsing.llm import LLMParams
from processmind.rag.knowledge_base import build_knowledge_base, get_pattern_by_id, search_automation_patterns
from tests.helpers_kb import FakeEmbedder
from tests.helpers_workflow import context_of, rec

DATASET = load_dataset()
PARAMS = LLMParams(temperature=0.0, top_p=1.0, max_tokens=4096, seed=42)


class CountingEmbedder(FakeEmbedder):
    """Лексический двойник эмбеддера, считающий «токены» как OpenAIEmbedder."""

    def __init__(self):
        super().__init__("fake-embedding-v1")
        self.tokens_used = 0

    def embed(self, texts):
        self.tokens_used += sum(len(t) // 4 + 1 for t in texts)
        return super().embed(texts)


class UsageLLM:
    """Двойник openai.OpenAI с usage; `responder(messages, kwargs)` строит ответ, вызовы запоминаются."""

    def __init__(self, responder):
        self.calls: list[dict] = []
        self._responder = responder
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self._responder(kwargs["messages"], kwargs), refusal=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=1000 + len(self.calls), completion_tokens=200, total_tokens=1200 + len(self.calls))
        return SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o-mini", system_fingerprint="fp_test")


def good_llm(messages, kwargs):
    """Добросовестная «модель»: рекомендует по AP-005 только если этот паттерн ей передали (B), иначе — без ссылки (A)."""
    context = context_of(messages)
    patterns = {p["pattern_id"] for p in context["knowledge_base_patterns"]}
    manual = [s["id"] for s in context["process"]["steps"] if s["is_manual"] is True]
    if not manual:
        return LLMRecommendationSet(recommendations=[], notes=[])
    step = manual[0]
    pattern = "AP-005" if "AP-005" in patterns else None
    return LLMRecommendationSet(recommendations=[rec(step, pattern, problem="Данные вводятся повторно", solution="Убрать повторный ввод данных из разных систем", rationale="Операция ручная, одни и те же данные вводятся повторно; это подтверждено описанием операции.")], notes=[])


@pytest.fixture(scope="module")
def mcp_client():
    with SyncMCPClient() as client:
        yield client


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("chroma")
    build_knowledge_base(embedder=FakeEmbedder("fake-embedding-v1"), persist_dir=path)
    return path


def make_ctx(mcp_client, store, llm, cache=None) -> EvalContext:
    embedder = CountingEmbedder()
    return EvalContext(
        mcp=mcp_client, client=llm, model="gpt-4o-mini", skill=load_skill(),
        retrieve=lambda q, k: search_automation_patterns(q, top_k=k, embedder=embedder, persist_dir=store),
        get_pattern=lambda pid: get_pattern_by_id(pid, persist_dir=store, embedding_model=embedder.model),
        resolve_text=lambda text: (search_automation_patterns(text, top_k=1, embedder=embedder, persist_dir=store) or [None])[0].pattern_id,
        embedder=embedder, kb_sha=kb_fingerprint(), dataset_sha=DATASET.sha256, cache=cache or EvalCache(None),
    )


# ---------- A и B отличаются ТОЛЬКО наличием паттернов ----------


def test_configs_differ_only_in_retrieved_patterns(mcp_client, store):
    llm = UsageLLM(good_llm)
    ctx = make_ctx(mcp_client, store, llm)
    example = DATASET.by_id("GD-013")
    execute_case(example, "A", ctx, PARAMS)
    execute_case(example, "B", ctx, PARAMS)

    a_call, b_call = llm.calls
    a_ctx, b_ctx = context_of(a_call["messages"]), context_of(b_call["messages"])
    assert a_ctx["knowledge_base_patterns"] == [] and b_ctx["knowledge_base_patterns"]  # единственное различие
    assert a_call["messages"][0] == b_call["messages"][0]  # одни и те же основные инструкции (Skill)
    for key in ("model", "temperature", "top_p", "max_completion_tokens", "seed"):
        assert a_call[key] == b_call[key]
    assert (a_call["temperature"], a_call["top_p"], a_call["max_completion_tokens"], a_call["seed"]) == (0.0, 1.0, 4096, 42)
    assert {k: v for k, v in a_ctx.items() if k != "knowledge_base_patterns"} == {k: v for k, v in b_ctx.items() if k != "knowledge_base_patterns"}
    assert example.question in a_call["messages"][1]["content"] and example.question in b_call["messages"][1]["content"]


def test_config_a_cannot_reference_patterns_and_validation_rejects_invented_ones(mcp_client, store):
    invented = UsageLLM(lambda m, k: LLMRecommendationSet(recommendations=[rec("s2", "AP-005")], notes=[]))
    ctx = make_ctx(mcp_client, store, invented)
    a = execute_case(DATASET.by_id("GD-013"), "A", ctx, PARAMS)
    assert a.raw_recs == [{"step_id": "s2", "pattern_id": "AP-005"}] and a.recs == [] and any("AP-005" in r for r in a.rejected_reasons)


def test_result_records_usage_latency_retrieval_and_resolution(mcp_client, store):
    ctx = make_ctx(mcp_client, store, UsageLLM(good_llm))
    b = execute_case(DATASET.by_id("GD-013"), "B", ctx, PARAMS)
    assert (b.llm.prompt_tokens, b.llm.completion_tokens, b.llm.total_tokens) == (1001, 200, 1201) and b.llm.latency_s > 0 and b.llm.finish_reason == "stop"
    assert b.retrieval_latency_s > 0 and b.embedding_tokens > 0 and b.total_latency_s == pytest.approx(b.retrieval_latency_s + b.llm.latency_s)
    assert b.retrieved_pool and b.recs[0].pattern_id == "AP-005" and b.recs[0].resolved_pattern.startswith("AP-")
    assert b.recs[0].rationale_present and b.recs[0].conditions_analyzed
    a = execute_case(DATASET.by_id("GD-013"), "A", make_ctx(mcp_client, store, UsageLLM(good_llm)), PARAMS)
    assert a.retrieved_pool == [] and a.embedding_tokens == 0 and a.retrieval_latency_s == 0 and a.recs[0].pattern_id is None


def test_llm_error_is_recorded_not_swallowed_and_not_cached(mcp_client, store, tmp_path):
    def boom(messages, kwargs):
        raise RuntimeError("не должно быть вызвано как исключение SDK")

    from openai import APIConnectionError
    import httpx

    class Failing(UsageLLM):
        def _parse(self, **kwargs):
            raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))

    cache = EvalCache(tmp_path / "cache.json")
    ctx = make_ctx(mcp_client, store, Failing(good_llm), cache)
    result = execute_case(DATASET.by_id("GD-013"), "B", ctx, PARAMS)
    assert result.error and "подключиться" in result.error and result.recs == []
    assert cache._data == {} or all("case" not in k for k in cache._data)  # ошибка не попала в кэш кейсов


def test_cache_avoids_repeated_llm_calls_and_respects_fresh(mcp_client, store, tmp_path):
    llm = UsageLLM(good_llm)
    cache = EvalCache(tmp_path / "cache.json")
    ctx = make_ctx(mcp_client, store, llm, cache)
    first = execute_case(DATASET.by_id("GD-013"), "B", ctx, PARAMS)
    second = execute_case(DATASET.by_id("GD-013"), "B", ctx, PARAMS)
    assert len(llm.calls) == 1 and first == second  # второй раз — из кэша
    execute_case(DATASET.by_id("GD-013"), "B", ctx, dataclasses.replace(PARAMS, temperature=0.7))
    assert len(llm.calls) == 2  # другие параметры — другой ключ кэша
    reloaded = make_ctx(mcp_client, store, UsageLLM(good_llm), EvalCache(tmp_path / "cache.json"))
    assert execute_case(DATASET.by_id("GD-013"), "B", reloaded, PARAMS) == first  # кэш переживает перезапуск
    fresh_llm = UsageLLM(good_llm)
    execute_case(DATASET.by_id("GD-013"), "B", make_ctx(mcp_client, store, fresh_llm, EvalCache(tmp_path / "cache.json", enabled=False)), PARAMS)
    assert len(fresh_llm.calls) == 1  # --fresh игнорирует кэш


# ---------- оценка результата ----------


def result(test_id, config="B", recs=(), raw=None, **llm):
    kind = DATASET.by_id(test_id).kind
    views = [RecView(step_id=s, pattern_id=p, resolved_pattern=r, problem="п", solution="р", rationale_present=True, conditions_analyzed=p is not None) for s, p, r in recs]
    return CaseResult(test_id=test_id, config=config, kind=kind, params={}, recs=views, raw_recs=raw if raw is not None else [{"step_id": s, "pattern_id": p} for s, p, _ in recs], llm=LLMStats(**{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "latency_s": 1.0, **llm}))


def test_case_outcome_scoring_by_kind():
    positive = DATASET.by_id("GD-013")  # ожидается AP-005; допустимы AP-006
    assert case_outcome(positive, result("GD-013", recs=[("s2", "AP-005", "AP-005")]))["f1"] == 1.0
    assert case_outcome(positive, result("GD-013", recs=[("s2", None, "AP-011")]))["f1"] == 0.0  # по смыслу — не тот паттерн
    assert case_outcome(positive, result("GD-013", recs=[("s2", "AP-005", "AP-011")]), "explicit")["f1"] == 1.0
    negative = DATASET.by_id("GD-024")
    assert case_outcome(negative, result("GD-024"))["correct"] is True
    assert case_outcome(negative, result("GD-024", recs=[("s2", None, "AP-001")]))["correct"] is False
    forbidden = DATASET.by_id("GD-028")  # для s1 недопустимы AP-007, AP-014, AP-012
    assert case_outcome(forbidden, result("GD-028", recs=[("s1", None, "AP-007")]))["violations"] == 1
    assert case_outcome(forbidden, result("GD-028"))["correct"] is True


def test_aggregate_config_hand_calculated():
    results = [
        result("GD-013", recs=[("s2", "AP-005", "AP-005")], latency_s=2.0, prompt_tokens=100),  # positive, F1=1
        result("GD-014", recs=[("s2", "AP-001", "AP-001")], latency_s=4.0, prompt_tokens=300),  # positive, F1=0 (AP-001 вне ожидаемых и допустимых)
        result("GD-024", raw=[{"step_id": "s2", "pattern_id": None}]),  # отрицательный: LLM сначала что-то предложила, валидация убрала
        result("GD-025", recs=[("s1", None, "AP-001")]),  # отрицательный: необоснованная рекомендация
    ]
    m = aggregate_config(DATASET, results)
    assert m["f1_resolved"]["n_positive"] == 2 and m["f1_resolved"]["macro_f1"] == pytest.approx(0.5)
    assert (m["f1_resolved"]["tp"], m["f1_resolved"]["fp"], m["f1_resolved"]["fn"]) == (1, 1, 1) and m["f1_resolved"]["micro_f1"] == pytest.approx(0.5)
    assert m["negatives"] == {"n": 2, "correct_after_validation": 1, "correct_raw_llm": 0, "false_recommendations_after_validation": 1, "false_recommendations_raw_llm": 2}
    assert m["llm_latency_s"]["mean"] == pytest.approx((2 + 4 + 1 + 1) / 4) and m["prompt_tokens"]["total"] == 100 + 300 + 10 + 10
    assert m["justification"]["recommendations"] == 3 and m["justification"]["kb_reference"] == pytest.approx(2 / 3)
    assert m["rejected_by_validation"] == 1  # сырых рекомендаций 4 (одна из них убрана валидацией у GD-024), принято 3


def test_errors_are_excluded_from_scores_but_reported():
    failed = CaseResult(test_id="GD-013", config="B", kind="positive", params={}, error="сбой API")
    m = aggregate_config(DATASET, [failed, result("GD-014", recs=[("s2", "AP-005", "AP-005")])])
    assert m["errors"] == ["GD-013"] and m["evaluated"] == 1 and m["f1_resolved"]["n_positive"] == 1


# ---------- эксперимент с temperature ----------


def test_temperature_experiment_uses_fixed_examples_seeds_and_only_changes_temperature(mcp_client, store):
    llm = UsageLLM(good_llm)
    ctx = make_ctx(mcp_client, store, llm)
    out = run_temperature_experiment(DATASET, ctx, PARAMS, progress=lambda _: None)

    assert len(out["examples"]) >= 5 and set(out["temperatures"]) == {0.0, 0.7}  # не менее 5 примеров и два значения
    assert len(llm.calls) == len(out["examples"]) * 2 * 3
    seen = {(c["temperature"], c["seed"]) for c in llm.calls}
    assert seen == {(t, s) for t in (0.0, 0.7) for s in (11, 22, 33)}
    assert {c["top_p"] for c in llm.calls} == {1.0} and {c["max_completion_tokens"] for c in llm.calls} == {4096}
    per_example_patterns = {tuple(json.dumps(c["messages"], sort_keys=True) for c in [llm.calls[0]])}
    assert per_example_patterns  # запросы к LLM для одного примера совпадают, кроме temperature/seed (retrieval считается один раз)


def test_temperature_rule_prefers_lower_value_unless_gain_is_material(mcp_client, store, monkeypatch):
    def run_with_f1(low_f1, high_f1, stability=(1.0, 1.0)):
        """Подменяем оценку F1 и устойчивость, чтобы проверить только правило выбора."""
        ctx = make_ctx(mcp_client, store, UsageLLM(good_llm))
        monkeypatch.setattr(experiments, "case_outcome", lambda ex, r, view="resolved": {"f1": high_f1 if r.params.get("temperature") == 0.7 else low_f1, "violations": 0, "recs": 0, "correct": True})
        monkeypatch.setattr(experiments, "mean_pairwise_jaccard", lambda sets: stability[1] if len(sets) and False else stability[0])
        return run_temperature_experiment(DATASET, ctx, PARAMS, progress=lambda _: None)

    assert run_with_f1(0.60, 0.62)["chosen_temperature"] == 0.0  # прирост 0.02 < 0.05 — воспроизводимость важнее
    assert run_with_f1(0.60, 0.70)["chosen_temperature"] == 0.7  # прирост 0.10 ≥ 0.05 и устойчивость не хуже
    assert run_with_f1(0.70, 0.60)["chosen_temperature"] == 0.0


# ---------- отчёт ----------


def _results(verification=None):
    ab = {"metrics": {}, "results": {}}
    for cfg, recs in (("A", [("s2", None, "AP-001")]), ("B", [("s2", "AP-005", "AP-005")])):
        rs = [result("GD-013", cfg, recs=recs, latency_s=3.0, prompt_tokens=2000), result("GD-024", cfg)]
        ab["results"][cfg] = [r.model_dump(mode="json") for r in rs]
        ab["metrics"][cfg] = aggregate_config(DATASET, rs)
    ab["metrics"]["B"]["embedding_tokens_total"] = 40
    rows = [{"test_id": "GD-013", "query": "q", "top3": ["AP-005"], "scores": [0.6], "pool": ["AP-005"], "latency_s": 0.4, "embedding_tokens": 10, "expected": ["AP-005"], "hit_at_3": True, "recall_at_3": 1.0, "pool_hit": True}]
    return {
        "meta": {"generated_at": "2026-01-01 00:00", "duration_s": 12.0, "git": "abc123", "model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small", "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096, "seed": 42},
                 "dataset_sha": "d" * 64, "kb_sha": "k" * 64, "skill_sha": "s" * 64, "versions": {"openai": "1"}, "partial": False,
                 "langsmith": {"enabled": verification is not None, "project": "p", "reason": None if verification else "не задан LANGSMITH_API_KEY", "verification": verification}},
        "dataset": {"n": 30, "by_category": {"routing": 3}, "by_kind": {"positive": 23, "negative": 4, "ambiguous": 3}},
        "retrieval": {"n_positive": 1, "hits": 1, "hit_at_3": 1.0, "mean_recall_at_3": 1.0, "pool_hit_rate": 1.0, "mean_pool_size": 3.0, "latency_s": {"mean": 0.4, "median": 0.4, "p95": 0.4}, "embedding_tokens_total": 10, "rows": rows},
        "ab": ab, "temperature": None,
        "chosen": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096, "observed_max_completion_tokens": 900, "default_matches": True},
        "appendix": [{"test_id": "GD-013", "category": "deduplication", "kind": "positive", "expected": ["AP-005"], "a": ["AP-005"], "b": ["AP-005"], "b_explicit": ["AP-005"]}],
    }


def test_report_contains_all_required_sections_and_actual_numbers():
    md = report.render(_results())
    for heading in ("## 1. Мониторинг: LangSmith", "## 2. Golden Dataset", "### 3.1 Retrieval Hit@3", "## 4. A/B-эксперимент", "## 5. Гиперпараметры", "## 6. Ограничения оценки"):
        assert heading in md, heading
    assert "**Retrieval Hit@3**" in md and "1/1 (100%)" in md
    assert "| **Recommendation F1, macro**" in md and "Latency сквозная" in md and "Токены prompt" in md
    assert "temperature | 0.0" in md and "max_tokens | 4096" in md and "900 токенов" in md


def test_report_states_plainly_when_langsmith_traces_were_not_verified():
    md = report.render(_results(verification=None))
    assert "реальные трейсы в этом прогоне НЕ проверены" in md and "не задан LANGSMITH_API_KEY" in md and "tests/test_tracing.py" in md
    verified = report.render(_results(verification={"project": "p", "trace_id": "t1", "status": "completed", "checks": {"все 8 узлов LangGraph": True}, "runs": 40, "by_type": {"chain": 30}, "tokens": {"total_tokens": 5}, "duration_s": 9.0, "error_runs": []}))
    assert "трейсы проверены на реальном прогоне" in verified and "`t1`" in verified


def test_report_conclusions_follow_the_numbers():
    md = report.render(_results())
    assert "A = 0.000, B = 1.000" in md and "B лучше A" in md  # A: смысловой паттерн не совпал (нет ожидаемого), B: совпал


def test_takeaway_states_comparable_quality_and_worse_negatives_from_the_numbers():
    ab = _results()["ab"]["metrics"]
    a, b = json.loads(json.dumps(ab["A"])), json.loads(json.dumps(ab["B"]))
    a["f1_resolved"]["macro_f1"], b["f1_resolved"]["macro_f1"] = 0.725, 0.710  # разница -0.015: в пределах шума
    a["negatives"].update(n=4, correct_after_validation=3)
    b["negatives"].update(n=4, correct_after_validation=1)
    text = report._takeaway(a, b)
    assert "не дал измеримого прироста" in text and "1/4 против 3/4" in text and "причинность этой выборкой не доказана" in text
    assert "ссылки на паттерны" in text  # преимущество B — прослеживаемость — тоже отмечено

    b["f1_resolved"]["macro_f1"] = 0.900  # значимое улучшение выше порога
    assert "повысил macro-F1 на 0.175" in report._takeaway(a, b)


def test_temperature_rule_check_lists_each_condition_with_actual_values():
    temp = {"temperatures": [0.0, 0.7], "min_f1_gain_rule": 0.05,
            "summary": {"0.0": {"mean_f1": 0.567, "stability_jaccard": 0.796, "violation_rate": 0.17}, "0.7": {"mean_f1": 0.689, "stability_jaccard": 0.778, "violation_rate": 0.17}}}
    text = report._rule_check(temp)
    assert "| прирост macro-F1 при temperature=0.7 не менее 0.05 | да | +0.122 |" in text
    assert "| устойчивость (Jaccard) не хуже | нет | 0.778 против 0.796 |" in text
    assert "не выполнено — устойчивость (Jaccard) не хуже" in text and "скорее тенденция" in text


def test_render_only_regenerates_report_from_saved_results_without_any_api_calls(monkeypatch, tmp_path):
    (tmp_path / "latest.json").write_text(json.dumps(_results()), encoding="utf-8")
    monkeypatch.setattr(run_all, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_all, "EVALS_MD", tmp_path / "EVALS.md")
    monkeypatch.setattr(run_all, "get_settings", lambda: (_ for _ in ()).throw(AssertionError("настройки/ключи не нужны")))
    assert run_all.main(["--render-only"]) == 0
    assert "## 4. A/B-эксперимент" in (tmp_path / "EVALS.md").read_text(encoding="utf-8")
    (tmp_path / "latest.json").unlink()
    assert run_all.main(["--render-only"]) == 2  # нет сохранённых результатов — ошибка, а не пустой отчёт


# ---------- запуск одной командой: без ключей — ошибка и инструкция, без имитации ----------


def test_run_all_refuses_without_openai_key(monkeypatch, capsys):
    monkeypatch.setattr(run_all, "get_settings", lambda: Settings(_env_file=None, openai_api_key=None))
    assert run_all.main(["--limit", "1"]) == 2
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out and ".env" in out


def test_run_all_refuses_without_langsmith_unless_explicitly_allowed(monkeypatch, capsys):
    from processmind.observability import TracingStatus

    monkeypatch.setattr(run_all, "get_settings", lambda: Settings(_env_file=None, openai_api_key="sk-test"))
    monkeypatch.setattr(run_all, "tracing_status", lambda: TracingStatus(False, "p", None, "не задан LANGSMITH_API_KEY"))
    assert run_all.main(["--limit", "1"]) == 2
    out = capsys.readouterr().out
    assert "LANGSMITH_API_KEY" in out and "--allow-no-tracing" in out and "https://smith.langchain.com" in out


def test_verify_tracing_command_fails_with_instructions_when_not_configured(monkeypatch, capsys):
    from evals import tracing_check, verify_tracing
    from processmind.observability import TracingStatus

    monkeypatch.setattr(tracing_check, "tracing_status", lambda: TracingStatus(False, "p", None, "не задан LANGSMITH_API_KEY"))
    assert verify_tracing.main() == 2
    out = capsys.readouterr().out
    assert "ОШИБКА" in out and "LANGSMITH_API_KEY" in out and "python -m evals.verify_tracing" in out
