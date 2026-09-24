"""Запуск всех оценок одной командой: `python -m evals.run_all`.

Что делает: проверяет доступность OpenAI и LangSmith (иначе — ошибка и инструкция, без имитации), затем считает
Retrieval Hit@3, A/B (LLM без RAG и с RAG) на всех 30 примерах, эксперимент с temperature и проверяет реальные трейсы
LangSmith; результаты пишет в `evals/results/latest.json`, отчёт — в `EVALS.md`.

Опции:
  --limit N                 быстрый прогон на первых N примерах (пишет evals/results/smoke.json, EVALS.md не трогает)
  --fresh                   игнорировать кэш результатов (по умолчанию кэш позволяет возобновить прерванный прогон)
  --allow-no-tracing        продолжить без LangSmith; в EVALS.md будет явно указано, что трейсы НЕ проверены
  --skip-temperature        пропустить эксперимент со temperature
  --render-only             пересоздать EVALS.md из evals/results/latest.json без новых вызовов LLM
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata as metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from langsmith import tracing_context

from evals import report
from evals.dataset import GoldenDataset, load_dataset
from evals.experiments import TEMPERATURE_EXAMPLES, predicted_patterns, retrieval_eval, run_ab, run_temperature_experiment
from evals.runner import CaseResult, EvalCache, build_context
from processmind.config import get_settings
from processmind.observability import SETUP_INSTRUCTIONS, flush_tracing, tracing_status
from processmind.parsing.llm import LLMParams
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import build_knowledge_base

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evals" / "results"
EVALS_MD = ROOT / "EVALS.md"
EVAL_SEED = 42  # seed воспроизводимости для A/B (OpenAI обеспечивает его по возможности)

OPENAI_INSTRUCTIONS = """\
OpenAI недоступен. Проверьте:
  1. В файле .env задан OPENAI_API_KEY=<ваш ключ> (см. .env.example) и он действителен;
  2. есть сетевое соединение и на аккаунте OpenAI доступна квота;
  3. заданы модели: OPENAI_MODEL=gpt-4o-mini, EMBEDDING_MODEL=text-embedding-3-small.
Оценки требуют реальных вызовов LLM и эмбеддингов, поэтому без OpenAI они не запускаются."""


def _log(message: str) -> None:
    print(message, flush=True)


def _git() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
        return rev + (" (есть незакоммиченные изменения)" if dirty else "")
    except Exception:  # noqa: BLE001
        return "не определена"


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for name in ("openai", "langgraph", "langsmith", "chromadb", "mcp", "pydantic"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "?"
    return out


def _subset(dataset: GoldenDataset, limit: int | None) -> GoldenDataset:
    return dataset if not limit else GoldenDataset(examples=dataset.examples[:limit], sha256=dataset.sha256)


def _appendix(dataset: GoldenDataset, ab: dict) -> list[dict]:
    by_cfg = {c: {r["test_id"]: CaseResult.model_validate(r) for r in ab["results"][c]} for c in ("A", "B")}
    rows = []
    for ex in dataset.examples:
        a, b = by_cfg["A"][ex.test_id], by_cfg["B"][ex.test_id]
        rows.append({
            "test_id": ex.test_id, "category": ex.category, "kind": ex.kind, "expected": ex.expected_pattern_ids,
            "a": sorted(predicted_patterns(a, "resolved")) if not a.error else ["ОШИБКА"],
            "b": sorted(predicted_patterns(b, "resolved")) if not b.error else ["ОШИБКА"],
            "b_explicit": sorted(predicted_patterns(b, "explicit")) if not b.error else ["ОШИБКА"],
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description="Запуск всех автоматизированных оценок ProcessMind AI")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--allow-no-tracing", action="store_true")
    parser.add_argument("--skip-temperature", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    if args.render_only:  # только форматирование уже полученных результатов: никаких новых вызовов и чисел
        source = RESULTS_DIR / "latest.json"
        if not source.exists():
            _log(f"ОШИБКА: нет {source}. Сначала выполните `python -m evals.run_all`.")
            return 2
        EVALS_MD.write_text(report.render(json.loads(source.read_text(encoding="utf-8"))), encoding="utf-8")
        _log(f"Отчёт пересоздан из {source}: {EVALS_MD}")
        return 0
    settings = get_settings()

    # ---------- проверка окружения: ошибка + инструкция вместо имитации ----------
    if not settings.openai_api_key:
        _log("ОШИБКА: не задан OPENAI_API_KEY.\n\n" + OPENAI_INSTRUCTIONS)
        return 2
    status = tracing_status()
    if not status.enabled and not args.allow_no_tracing:
        _log(f"ОШИБКА: LangSmith не настроен ({status.reason}).\n\n{SETUP_INSTRUCTIONS}\n\n"
             "Чтобы всё же выполнить оценки без трейсов (в EVALS.md будет явно указано, что трейсы не проверены), добавьте флаг --allow-no-tracing.")
        return 2

    dataset = _subset(load_dataset(), args.limit)
    partial = args.limit is not None
    cache = EvalCache(RESULTS_DIR / "cache.json", enabled=not args.fresh)
    try:
        ctx = build_context(dataset.sha256 if not partial else load_dataset().sha256, cache, settings)
        ctx.embedder.embed(["проверка соединения с OpenAI"])  # реальный вызов: убеждаемся, что API доступен
        if build_knowledge_base(embedder=ctx.embedder).embedded_chunks:
            _log("База знаний (пере)индексирована.")
    except (KnowledgeBaseError, Exception) as exc:  # noqa: BLE001
        _log(f"ОШИБКА: {exc}\n\n{OPENAI_INSTRUCTIONS}")
        return 2

    ctx.client.chat.completions.create(model=ctx.model, messages=[{"role": "user", "content": "ok"}], max_completion_tokens=5)  # прогрев соединения: первый вызов не должен искажать latency
    params = LLMParams(temperature=settings.llm_temperature, top_p=settings.llm_top_p, max_tokens=settings.llm_max_tokens, seed=EVAL_SEED)
    eval_project = f"{status.project}-evals"
    _log(f"Оценки: {len(dataset.examples)} примеров, модель {ctx.model}, параметры {params}. Трассировка: "
         + (f"LangSmith, проект «{eval_project}»" if status.enabled else f"ВЫКЛЮЧЕНА ({status.reason})"))

    try:
        with tracing_context(project_name=eval_project, enabled=True if status.enabled else None):
            _log("\n[1/3] Retrieval Hit@3")
            retrieval = retrieval_eval(dataset, ctx, _log)
            _log(f"  Hit@3 = {retrieval['hits']}/{retrieval['n_positive']}")
            _log("\n[2/3] A/B: LLM без RAG (A) и с RAG (B)")
            ab = run_ab(dataset, ctx, params, _log)
            temperature = None
            if not args.skip_temperature and all(t in {e.test_id for e in dataset.examples} for t in TEMPERATURE_EXAMPLES):
                _log("\n[3/3] Эксперимент со temperature")
                temperature = run_temperature_experiment(dataset, ctx, params, _log)
            else:
                _log("\n[3/3] Эксперимент со temperature пропущен")
    finally:
        flush_tracing()
        ctx.close()

    # ---------- проверка реальных трейсов ----------
    verification, verification_error = None, None
    if status.enabled and not partial:
        _log("\nПроверка трейсов LangSmith (реальный анализ демо-процесса)…")
        from evals.tracing_check import verify
        from scripts.demo_mcp import demo_process

        try:
            verification = verify(demo_process())
            _log(f"  трейс {verification['trace_id']}: {verification['runs']} runs, проверки: {verification['checks']}")
        except Exception as exc:  # noqa: BLE001
            verification_error = str(exc)
            _log(f"  ОШИБКА проверки трейсов: {exc}")

    # ---------- результаты ----------
    observed_max = max([r["llm"]["completion_tokens"] for c in ("A", "B") for r in ab["results"][c]] + ([m["max_completion_tokens"] for m in temperature["summary"].values()] if temperature else []))
    chosen_t = temperature["chosen_temperature"] if temperature else params.temperature
    results = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "duration_s": time.time() - started, "git": _git(),
            "model": ctx.model, "embedding_model": settings.embedding_model, "params": {"temperature": params.temperature, "top_p": params.top_p, "max_tokens": params.max_tokens, "seed": params.seed},
            "dataset_sha": ctx.dataset_sha, "kb_sha": ctx.kb_sha, "skill_sha": ctx.skill.sha256, "versions": _versions(), "partial": partial,
            "langsmith": {"enabled": status.enabled, "project": status.project, "reason": status.reason, "verification": verification, "verification_error": verification_error},
        },
        "dataset": {"n": len(dataset.examples), "by_category": dict(collections.Counter(e.category for e in dataset.examples)), "by_kind": dict(collections.Counter(e.kind for e in dataset.examples))},
        "retrieval": retrieval, "ab": ab, "temperature": temperature,
        "chosen": {"temperature": chosen_t, "top_p": params.top_p, "max_tokens": params.max_tokens, "observed_max_completion_tokens": observed_max, "default_matches": settings.llm_temperature == chosen_t},
        "appendix": _appendix(dataset, ab),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULTS_DIR / ("smoke.json" if partial else "latest.json")
    target.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"\nРезультаты: {target}")
    if partial:
        _log("Частичный прогон (--limit): EVALS.md не обновлялся.")
    else:
        EVALS_MD.write_text(report.render(results), encoding="utf-8")
        _log(f"Отчёт: {EVALS_MD}")

    errors = ab["metrics"]["A"]["errors"] + ab["metrics"]["B"]["errors"]
    if errors:
        _log(f"ВНИМАНИЕ: ошибки LLM в примерах: {errors}")
    if status.enabled and not partial and (verification is None or not verification["ok"]):
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
