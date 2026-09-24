"""Эксперименты: оценка поиска (Hit@3), A/B (LLM без RAG и с RAG), сравнение значений temperature."""

from __future__ import annotations

import dataclasses
import statistics
import time
from typing import Any, Callable

from evals.dataset import GoldenDataset, GoldenExample
from evals.runner import CaseResult, EvalContext, RetrievalCase, execute_case
from evals.scoring import (
    PredictedRecommendation,
    ambiguous_correct,
    find_violations,
    hit_at_k,
    macro_f1,
    mean_pairwise_jaccard,
    micro_f1,
    negative_correct,
    pattern_score,
    rate,
    recall_at_k,
    summarize,
)
from processmind.observability import traced
from processmind.parsing.llm import LLMParams
from processmind.rag.knowledge_base import build_query_from_process

View = str  # "resolved" — паттерн, ближайший по смыслу к тексту рекомендации; "explicit" — pattern_id, названный LLM

# Эксперимент с temperature: фиксированный заранее набор примеров (по одному из шести категорий, включая отрицательный).
TEMPERATURE_EXAMPLES = ("GD-001", "GD-007", "GD-010", "GD-013", "GD-016", "GD-025")
TEMPERATURE_VALUES = (0.0, 0.7)
TEMPERATURE_SEEDS = (11, 22, 33)
# Правило выбора (задано до эксперимента): большая temperature выбирается, только если даёт прирост macro-F1 не менее
# этой величины И не ухудшает устойчивость и долю нарушений; иначе — меньшая (воспроизводимость важнее).
TEMPERATURE_MIN_F1_GAIN = 0.05


def _pattern(rec: Any, view: View) -> str | None:
    return rec.pattern_id if view == "explicit" else rec.resolved_pattern


def predicted(result: CaseResult, view: View) -> list[PredictedRecommendation]:
    return [PredictedRecommendation(r.step_id, _pattern(r, view)) for r in result.recs]


def predicted_patterns(result: CaseResult, view: View) -> set[str]:
    return {p for r in result.recs if (p := _pattern(r, view))}


def case_outcome(example: GoldenExample, result: CaseResult, view: View = "resolved") -> dict[str, Any]:
    """Оценка одного результата относительно эталона примера."""
    preds = predicted(result, view)
    violations = find_violations(preds, example.unacceptable_recommendations)
    out: dict[str, Any] = {"violations": len(violations), "recs": len(preds)}
    if example.kind == "positive":
        score = pattern_score({r.pattern_id for r in preds if r.pattern_id}, example.expected_pattern_ids, example.acceptable_patterns)
        out.update(score=score, f1=score.f1)
    elif example.kind == "negative":
        out["correct"] = negative_correct(preds)
    else:
        out["correct"] = ambiguous_correct(preds, violations, set(example.expected_pattern_ids) | example.acceptable_patterns)
    return out


def _summary_dict(values: list[float]) -> dict[str, float]:
    return dataclasses.asdict(summarize(values))


def aggregate_config(dataset: GoldenDataset, results: list[CaseResult]) -> dict[str, Any]:
    """Метрики одной конфигурации по всем её результатам."""
    ok = {r.test_id: r for r in results if not r.error}
    errors = sorted(r.test_id for r in results if r.error)
    pairs = [(dataset.by_id(tid), r) for tid, r in ok.items()]
    out: dict[str, Any] = {"n": len(results), "evaluated": len(ok), "errors": errors}

    for view in ("resolved", "explicit"):
        scores = [case_outcome(ex, r, view)["score"] for ex, r in pairs if ex.kind == "positive"]
        out[f"f1_{view}"] = {
            "n_positive": len(scores),
            "macro_f1": macro_f1(scores),
            "micro_f1": micro_f1(scores),
            "mean_precision": statistics.fmean(s.precision for s in scores) if scores else 0.0,
            "mean_recall": statistics.fmean(s.recall for s in scores) if scores else 0.0,
            "tp": sum(s.tp for s in scores), "fp": sum(s.fp for s in scores), "fn": sum(s.fn for s in scores),
        }
        violations = [case_outcome(ex, r, view)["violations"] for ex, r in pairs]
        out[f"violations_{view}"] = {"total": sum(violations), "examples_with_violation": sum(1 for v in violations if v), "n": len(violations)}

    neg = [(ex, r) for ex, r in pairs if ex.kind == "negative"]
    out["negatives"] = {
        "n": len(neg),
        "correct_after_validation": sum(1 for ex, r in neg if not r.recs),
        "correct_raw_llm": sum(1 for ex, r in neg if not r.raw_recs),
        "false_recommendations_after_validation": sum(len(r.recs) for ex, r in neg),
        "false_recommendations_raw_llm": sum(len(r.raw_recs) for ex, r in neg),
    }
    amb = [(ex, r) for ex, r in pairs if ex.kind == "ambiguous"]
    out["ambiguous"] = {"n": len(amb), "correct": sum(1 for ex, r in amb if case_outcome(ex, r)["correct"])}

    recs = [rec for r in ok.values() for rec in r.recs]
    out["justification"] = {
        "recommendations": len(recs),
        "per_example": len(recs) / len(ok) if ok else 0.0,
        "rationale_present": rate([r.rationale_present for r in recs]),
        "kb_reference": rate([r.pattern_id is not None for r in recs]),
        "conditions_analyzed": rate([r.conditions_analyzed for r in recs]),
    }
    out["raw_recommendations"] = sum(len(r.raw_recs) for r in ok.values())
    out["rejected_by_validation"] = sum(len(r.raw_recs) - len(r.recs) for r in ok.values())
    out["llm_latency_s"] = _summary_dict([r.llm.latency_s for r in ok.values()])
    out["retrieval_latency_s"] = _summary_dict([r.retrieval_latency_s for r in ok.values()])
    out["total_latency_s"] = _summary_dict([r.total_latency_s for r in ok.values()])
    for kind in ("prompt_tokens", "completion_tokens", "total_tokens"):
        out[kind] = _summary_dict([float(getattr(r.llm, kind)) for r in ok.values()])
    out["embedding_tokens_total"] = sum(r.embedding_tokens for r in ok.values())
    out["finish_reasons"] = sorted({r.llm.finish_reason or "?" for r in ok.values()})
    out["system_fingerprints"] = sorted({r.llm.system_fingerprint or "?" for r in ok.values()})
    return out


# ---------- оценка поиска ----------


@traced(name="eval_retrieval_case", run_type="retriever", process_inputs=lambda i: {"test_id": i["example"].test_id}, process_outputs=lambda r: {"top3": r.top3})
def retrieval_case(example: GoldenExample, ctx: EvalContext) -> RetrievalCase:
    spec = ctx.prepare(example)[0]
    query = build_query_from_process(spec, example.target_step_id)  # запрос для целевой операции — как в узле workflow
    tokens_before, started = ctx.embedder.tokens_used, time.perf_counter()
    matches = ctx.retrieve(query, 3)
    latency = time.perf_counter() - started
    pool = [p["pattern_id"] for p in ctx.retrieval(example)["patterns"]]
    return RetrievalCase(
        test_id=example.test_id, query=query, top3=[m.pattern_id for m in matches], scores=[m.score for m in matches],
        pool=pool, latency_s=latency, embedding_tokens=ctx.embedder.tokens_used - tokens_before,
    )


def retrieval_eval(dataset: GoldenDataset, ctx: EvalContext, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Retrieval Hit@3 на positive-примерах: найден ли ожидаемый паттерн в Top-3 поиска для целевой операции."""
    cases = []
    for ex in dataset.positives:
        cases.append(retrieval_case(ex, ctx))
        progress(f"  retrieval {ex.test_id}: top3={cases[-1].top3} ожидалось={ex.expected_pattern_ids}")
    rows = []
    for case in cases:
        expected = dataset.by_id(case.test_id).expected_pattern_ids
        rows.append({**case.model_dump(), "expected": expected, "hit_at_3": hit_at_k(case.top3, expected, 3), "recall_at_3": recall_at_k(case.top3, expected, 3), "pool_hit": bool(set(case.pool) & set(expected))})
    return {
        "n_positive": len(rows),
        "hit_at_3": rate([r["hit_at_3"] for r in rows]),
        "hits": sum(1 for r in rows if r["hit_at_3"]),
        "mean_recall_at_3": statistics.fmean(r["recall_at_3"] for r in rows) if rows else 0.0,
        "pool_hit_rate": rate([r["pool_hit"] for r in rows]),
        "mean_pool_size": statistics.fmean(len(r["pool"]) for r in rows) if rows else 0.0,
        "latency_s": _summary_dict([r["latency_s"] for r in rows]),
        "embedding_tokens_total": sum(r["embedding_tokens"] for r in rows),
        "rows": rows,
    }


# ---------- A/B ----------


def run_ab(dataset: GoldenDataset, ctx: EvalContext, params: LLMParams, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """A — LLM без RAG, B — LLM с RAG: одна модель, одни данные, одни инструкции и параметры генерации."""
    results: dict[str, list[CaseResult]] = {"A": [], "B": []}
    for ex in dataset.examples:
        for config in ("A", "B"):
            r = execute_case(ex, config, ctx, params, langsmith_extra={"tags": [f"config:{config}", f"test:{ex.test_id}", "experiment:ab"], "metadata": {"test_id": ex.test_id, "config": config, "category": ex.category}})
            results[config].append(r)
            progress(f"  {ex.test_id} [{config}] рекомендаций={len(r.recs)} (сырых {len(r.raw_recs)}) паттерны={sorted(predicted_patterns(r, 'resolved'))} llm={r.llm.latency_s:.1f}с" + (f" ОШИБКА: {r.error}" if r.error else ""))
    return {"results": {c: [r.model_dump(mode="json") for r in rs] for c, rs in results.items()}, "metrics": {c: aggregate_config(dataset, rs) for c, rs in results.items()}}


# ---------- temperature ----------


def run_temperature_experiment(dataset: GoldenDataset, ctx: EvalContext, base: LLMParams, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Сравнение двух значений temperature на 6 примерах (конфигурация B), 3 повтора с фиксированными seed.

    Меняется только temperature; retrieval вычисляется один раз на пример и одинаков для всех запусков.
    """
    examples = [dataset.by_id(t) for t in TEMPERATURE_EXAMPLES]
    runs: dict[float, dict[str, list[CaseResult]]] = {t: {ex.test_id: [] for ex in examples} for t in TEMPERATURE_VALUES}
    for temperature in TEMPERATURE_VALUES:
        for ex in examples:
            for seed in TEMPERATURE_SEEDS:
                params = dataclasses.replace(base, temperature=temperature, seed=seed)
                r = execute_case(ex, "B", ctx, params, langsmith_extra={"tags": ["experiment:temperature", f"temperature:{temperature}"], "metadata": {"test_id": ex.test_id, "temperature": temperature, "seed": seed}})
                runs[temperature][ex.test_id].append(r)
                progress(f"  T={temperature} {ex.test_id} seed={seed}: паттерны={sorted(predicted_patterns(r, 'resolved'))}" + (f" ОШИБКА: {r.error}" if r.error else ""))

    summary: dict[str, Any] = {}
    for temperature, by_example in runs.items():
        flat = [r for rs in by_example.values() for r in rs if not r.error]
        outcomes = [(dataset.by_id(r.test_id), r) for r in flat]
        f1s = [case_outcome(ex, r)["f1"] for ex, r in outcomes if ex.kind == "positive"]
        neg = [r for ex, r in outcomes if ex.kind == "negative"]
        stability = [mean_pairwise_jaccard([predicted_patterns(r, "resolved") for r in rs if not r.error]) for rs in by_example.values()]
        summary[str(temperature)] = {
            "runs": len(flat), "errors": sum(1 for rs in by_example.values() for r in rs if r.error),
            "mean_f1": statistics.fmean(f1s) if f1s else 0.0,
            "stability_jaccard": statistics.fmean(stability) if stability else 1.0,
            "violation_rate": rate([case_outcome(ex, r)["violations"] > 0 for ex, r in outcomes]),
            "negative_correct_rate": rate([not r.recs for r in neg]),
            "mean_llm_latency_s": statistics.fmean(r.llm.latency_s for r in flat) if flat else 0.0,
            "mean_total_tokens": statistics.fmean(r.llm.total_tokens for r in flat) if flat else 0.0,
            "max_completion_tokens": max((r.llm.completion_tokens for r in flat), default=0),
            "per_example": {tid: {"seeds": list(TEMPERATURE_SEEDS), "patterns": [sorted(predicted_patterns(r, "resolved")) for r in rs]} for tid, rs in by_example.items()},
        }
    low, high = (str(t) for t in sorted(TEMPERATURE_VALUES))
    gain = summary[high]["mean_f1"] - summary[low]["mean_f1"]
    high_ok = gain >= TEMPERATURE_MIN_F1_GAIN and summary[high]["stability_jaccard"] >= summary[low]["stability_jaccard"] and summary[high]["violation_rate"] <= summary[low]["violation_rate"]
    return {"examples": list(TEMPERATURE_EXAMPLES), "temperatures": list(TEMPERATURE_VALUES), "seeds": list(TEMPERATURE_SEEDS), "summary": summary, "f1_gain_high_vs_low": gain, "min_f1_gain_rule": TEMPERATURE_MIN_F1_GAIN, "chosen_temperature": float(high if high_ok else low)}
