"""Исполнитель оценок: один пример -> результат конфигурации A (LLM без RAG) или B (LLM с RAG).

Никакой параллельной реализации нет: метрики процесса считает MCP-сервер, паттерны ищет
`retrieve_patterns_for_process` (тот же код, что в workflow), контекст и валидацию рекомендаций дают функции
`processmind.analysis.recommendations`. Конфигурации отличаются ТОЛЬКО блоком `knowledge_base_patterns`
в данных для LLM: у A он пуст, у B содержит найденные паттерны. Модель, параметры генерации, Skill (системный
промпт), вопрос и данные процесса — одинаковые.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from evals.dataset import GoldenExample
from processmind.analysis.recommendations import (
    RecommendationError,
    build_llm_context,
    generate_recommendations_ex,
    validate_recommendations,
)
from processmind.analysis.retrieval import retrieve_patterns_for_process
from processmind.analysis.skill import Skill, load_skill
from processmind.config import Settings, get_settings
from processmind.mcp.client import SyncMCPClient
from processmind.models import ProcessSpec
from processmind.observability import traced
from processmind.parsing.llm import LLMParams, get_client
from processmind.rag.embeddings import OpenAIEmbedder, get_embedder
from processmind.rag.knowledge_base import PatternMatch, get_pattern_by_id, search_automation_patterns

Config = Literal["A", "B"]
CACHE_VERSION = "v1"  # менять при изменении логики, влияющей на результаты
JUSTIFICATION_MIN_CHARS = 40


# ---------- результаты ----------


class RecView(BaseModel):
    """Рекомендация, прошедшая программную валидацию (то, что увидел бы пользователь)."""

    step_id: str
    pattern_id: str | None
    resolved_pattern: str | None = Field(description="Паттерн, ближайший по смыслу к тексту рекомендации (одинаковая процедура для A и B)")
    problem: str
    solution: str
    rationale_present: bool
    conditions_analyzed: bool


class LLMStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None
    system_fingerprint: str | None = None


class CaseResult(BaseModel):
    test_id: str
    config: Config
    kind: str
    params: dict[str, Any]
    error: str | None = None
    raw_recs: list[dict[str, Any]] = Field(default_factory=list)  # до программной валидации
    recs: list[RecView] = Field(default_factory=list)
    rejected_reasons: list[str] = Field(default_factory=list)
    retrieved_pool: list[str] = Field(default_factory=list)  # только B
    retrieval_latency_s: float = 0.0
    embedding_tokens: int = 0  # токены эмбеддингов запросов поиска (только B)
    llm: LLMStats = Field(default_factory=LLMStats)

    @property
    def total_latency_s(self) -> float:
        return self.retrieval_latency_s + self.llm.latency_s


class RetrievalCase(BaseModel):
    test_id: str
    query: str
    top3: list[str]
    scores: list[float]
    pool: list[str]
    latency_s: float
    embedding_tokens: int


# ---------- кэш ----------


class EvalCache:
    """Кэш результатов на диске: позволяет возобновить длинный прогон; `--fresh` его игнорирует."""

    def __init__(self, path: Path | None, *, enabled: bool = True) -> None:
        self._path, self._enabled = path, enabled and path is not None
        self._data: dict[str, Any] = {}
        if self._enabled and path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha1(json.dumps([CACHE_VERSION, *parts], ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        return self._data.get(key) if self._enabled else None

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")


# ---------- контекст ----------


@dataclass
class EvalContext:
    mcp: Any
    client: Any
    model: str
    skill: Skill
    retrieve: Callable[[str, int], list[PatternMatch]]
    get_pattern: Callable[[str], Any]
    resolve_text: Callable[[str], str | None]
    embedder: Any
    kb_sha: str
    dataset_sha: str
    cache: EvalCache
    owned_mcp: SyncMCPClient | None = None
    _prepared: dict[str, Any] = field(default_factory=dict)
    _retrieval: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        if self.owned_mcp is not None:
            self.owned_mcp.close()

    # -- подготовка примера: метрики процесса считает MCP-сервер, как в workflow --
    def prepare(self, example: GoldenExample) -> tuple[ProcessSpec, dict, dict]:
        if example.test_id not in self._prepared:
            validation = self.mcp.validate_process(example.process)
            metrics = self.mcp.calculate_process_metrics(example.process, None)
            self._prepared[example.test_id] = (ProcessSpec.model_validate(example.process), validation, metrics)
        return self._prepared[example.test_id]

    # -- поиск паттернов: тот же код, что в узле retrieve_automation_patterns --
    def retrieval(self, example: GoldenExample) -> dict[str, Any]:
        if example.test_id not in self._retrieval:
            key = self.cache.key("retrieval-pool", example.test_id, self.kb_sha, self.embedder.model)
            cached = self.cache.get(key)
            if cached is None:
                spec = self.prepare(example)[0]
                tokens_before, started = self.embedder.tokens_used, time.perf_counter()
                patterns, warning = retrieve_patterns_for_process(spec, self.retrieve, self.get_pattern)
                cached = {"patterns": patterns, "warning": warning, "latency_s": time.perf_counter() - started, "embedding_tokens": self.embedder.tokens_used - tokens_before}
                self.cache.put(key, cached)
            self._retrieval[example.test_id] = cached
        return self._retrieval[example.test_id]


def kb_fingerprint() -> str:
    from processmind.rag.knowledge_base import DEFAULT_KB_DIR
    from processmind.rag.patterns import load_patterns

    return hashlib.sha256("".join(sorted(p.content_hash for p in load_patterns(DEFAULT_KB_DIR))).encode()).hexdigest()


def build_context(dataset_sha: str, cache: EvalCache, settings: Settings | None = None) -> EvalContext:
    """Настоящие зависимости: OpenAI, эмбеддинги, ChromaDB (индекс приложения), MCP-сервер. Без ключа — понятная ошибка."""
    settings = settings or get_settings()
    client = get_client(settings, error_cls=RecommendationError)
    embedder: OpenAIEmbedder = get_embedder(settings)
    mcp = SyncMCPClient().start()

    def retrieve(query: str, k: int) -> list[PatternMatch]:
        return search_automation_patterns(query, top_k=k, embedder=embedder)

    def resolve_text(text: str) -> str | None:
        matches = search_automation_patterns(text, top_k=1, embedder=embedder)
        return matches[0].pattern_id if matches else None

    return EvalContext(
        mcp=mcp, client=client, model=settings.openai_model, skill=load_skill(), retrieve=retrieve, get_pattern=get_pattern_by_id,
        resolve_text=resolve_text, embedder=embedder, kb_sha=kb_fingerprint(), dataset_sha=dataset_sha, cache=cache, owned_mcp=mcp,
    )


# ---------- выполнение примера ----------


def _case_key(ctx: EvalContext, example: GoldenExample, config: str, params: LLMParams) -> str:
    return ctx.cache.key("case", example.test_id, config, ctx.model, dataclasses.asdict(params), ctx.skill.sha256, ctx.dataset_sha, ctx.kb_sha)


@traced(
    name="eval_case",
    run_type="chain",
    process_inputs=lambda inputs: {"test_id": inputs["example"].test_id, "config": inputs["config"], "question": inputs["example"].question},
    process_outputs=lambda result: {"recommendations": len(result.recs), "error": result.error, "llm_tokens": result.llm.total_tokens},
)
def execute_case(example: GoldenExample, config: Config, ctx: EvalContext, params: LLMParams) -> CaseResult:
    """Один пример в конфигурации A или B. Ошибка LLM фиксируется в результате (пример не выбрасывается молча)."""
    key = _case_key(ctx, example, config, params)
    cached = ctx.cache.get(key)
    if cached is not None:
        return CaseResult.model_validate(cached)

    spec, validation, metrics = ctx.prepare(example)
    result = CaseResult(test_id=example.test_id, config=config, kind=example.kind, params={"temperature": params.temperature, "top_p": params.top_p, "max_tokens": params.max_tokens, "seed": params.seed})

    patterns: list[dict[str, Any]] = []
    if config == "B":
        retrieval = ctx.retrieval(example)
        patterns = retrieval["patterns"]
        result.retrieved_pool = [p["pattern_id"] for p in patterns]
        result.retrieval_latency_s, result.embedding_tokens = retrieval["latency_s"], retrieval["embedding_tokens"]

    context = build_llm_context(spec, validation, metrics, patterns)
    try:
        raw, usage = generate_recommendations_ex(ctx.client, ctx.model, ctx.skill, context, question=example.question, params=params)
    except RecommendationError as exc:
        result.error = str(exc)
        return result  # ошибки не кэшируем: повторный запуск попробует снова

    result.llm = LLMStats(**{k: getattr(usage, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_s", "finish_reason", "system_fingerprint")})
    result.raw_recs = [{"step_id": r.step_id, "pattern_id": r.pattern_id} for r in raw.recommendations]

    outcome = validate_recommendations(raw, spec, {p["pattern_id"]: p for p in patterns})
    result.rejected_reasons = [reason for rej in outcome.rejected for reason in rej.reasons]
    for rec in outcome.accepted:
        result.recs.append(
            RecView(
                step_id=rec.step_id,
                pattern_id=rec.pattern_id,
                resolved_pattern=ctx.resolve_text(f"{rec.problem}. {rec.solution}"),
                problem=rec.problem,
                solution=rec.solution,
                rationale_present=len(rec.rationale.strip()) >= JUSTIFICATION_MIN_CHARS,
                conditions_analyzed=bool(rec.conditions_confirmed or rec.conditions_unverified),
            )
        )
    ctx.cache.put(key, result.model_dump(mode="json"))
    return result
