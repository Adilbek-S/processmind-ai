"""Automation Knowledge Base: индексация паттернов автоматизации в ChromaDB и семантический поиск.

Пайплайн: Markdown-документы -> chunking по разделам -> эмбеддинги OpenAI -> ChromaDB (косинусная
метрика) -> поиск по вектору запроса -> Top-N паттернов с подтверждающими фрагментами.

Коллекция называется по модели эмбеддингов, поэтому векторы разных моделей никогда не смешиваются.
Повторная индексация идемпотентна: неизменившиеся документы не пересчитываются, изменившиеся
заменяются целиком, удалённые из каталога — убираются из индекса.
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from pydantic import BaseModel, Field

from processmind.config import Settings, get_settings
from processmind.models import ProcessSpec
from processmind.rag.embeddings import Embedder, get_embedder
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.patterns import AutomationPattern, PatternChunk, chunk_pattern, load_patterns

DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base" / "automation_patterns"
COLLECTION_PREFIX = "automation_patterns"
MAX_CANDIDATE_CHUNKS = 200


class IndexReport(BaseModel):
    """Итог индексации: что изменилось в индексе."""

    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    total_patterns: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0  # сколько chunk'ей реально отправлено на эмбеддинг в этом запуске


class ChunkHit(BaseModel):
    """Найденный фрагмент. Поля идентификации сохраняются вместе с фрагментом при индексации."""

    chunk_id: str
    pattern_id: str
    title: str
    source: str
    section: str
    text: str
    score: float  # косинусное сходство с запросом (1 — совпадение направлений)


class PatternMatch(BaseModel):
    """Найденный паттерн: оценка = сходство лучшего фрагмента; chunks — по убыванию сходства."""

    pattern_id: str
    title: str
    source: str
    score: float
    chunks: list[ChunkHit]


# ---------- ChromaDB ----------


def _collection_name(model: str) -> str:
    return f"{COLLECTION_PREFIX}__{re.sub(r'[^A-Za-z0-9._-]', '-', model)}"


def _client(persist_dir: str | Path | None, settings: Settings) -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(path=str(persist_dir or settings.chroma_persist_dir))


def _open_collection(client, model: str, *, create: bool):
    name = _collection_name(model)
    if create:
        # embedding_function=None: векторы считаем сами (OpenAI) и передаём явно — Chroma ничего не подставит.
        return client.get_or_create_collection(name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
    existing = {getattr(c, "name", c) for c in client.list_collections()}
    return client.get_collection(name, embedding_function=None) if name in existing else None


def _chunk_metadata(chunk: PatternChunk, content_hash: str, model: str) -> dict:
    return {
        "pattern_id": chunk.pattern_id,
        "title": chunk.title,
        "source": chunk.source,
        "section": chunk.section,
        "section_order": chunk.section_order,
        "chunk_id": chunk.chunk_id,
        "content_hash": content_hash,
        "embedding_model": model,
    }


# ---------- публичные функции ----------


def build_knowledge_base(
    kb_dir: str | Path | None = None,
    *,
    embedder: Embedder | None = None,
    persist_dir: str | Path | None = None,
    settings: Settings | None = None,
    prune: bool = True,
) -> IndexReport:
    """Индексирует Markdown-паттерны из `kb_dir`. Безопасно вызывать повторно.

    Эмбеддинги считаются ДО изменения индекса: если API недоступен, индекс остаётся как был.
    """
    settings = settings or get_settings()
    patterns = load_patterns(kb_dir or settings.knowledge_base_dir or DEFAULT_KB_DIR)
    embedder = embedder or get_embedder(settings)

    collection = _open_collection(_client(persist_dir, settings), embedder.model, create=True)
    indexed = collection.get(include=["metadatas"])
    hashes_by_pattern: dict[str, set[str]] = {}
    ids_by_pattern: dict[str, set[str]] = {}
    for chunk_id, meta in zip(indexed["ids"], indexed["metadatas"]):
        hashes_by_pattern.setdefault(meta["pattern_id"], set()).add(meta["content_hash"])
        ids_by_pattern.setdefault(meta["pattern_id"], set()).add(chunk_id)

    report = IndexReport(total_patterns=len(patterns))
    pending: list[tuple[AutomationPattern, list[PatternChunk]]] = []
    for pattern in patterns:
        chunks = chunk_pattern(pattern)
        report.total_chunks += len(chunks)
        is_current = hashes_by_pattern.get(pattern.pattern_id) == {pattern.content_hash} and ids_by_pattern.get(
            pattern.pattern_id
        ) == {c.chunk_id for c in chunks}
        if is_current:
            report.unchanged.append(pattern.pattern_id)
            continue
        (report.updated if pattern.pattern_id in ids_by_pattern else report.added).append(pattern.pattern_id)
        pending.append((pattern, chunks))

    all_chunks = [c for _, chunks in pending for c in chunks]
    vectors = embedder.embed([c.embedding_text for c in all_chunks]) if all_chunks else []
    report.embedded_chunks = len(all_chunks)

    for pattern, _ in pending:
        if pattern.pattern_id in ids_by_pattern:  # старые chunk'и заменяемого паттерна не должны остаться
            collection.delete(ids=sorted(ids_by_pattern[pattern.pattern_id]))
    if all_chunks:
        hash_by_pattern = {p.pattern_id: p.content_hash for p, _ in pending}
        collection.upsert(
            ids=[c.chunk_id for c in all_chunks],
            documents=[c.text for c in all_chunks],
            embeddings=vectors,
            metadatas=[_chunk_metadata(c, hash_by_pattern[c.pattern_id], embedder.model) for c in all_chunks],
        )

    if prune:
        current_ids = {p.pattern_id for p in patterns}
        for stale in sorted(set(ids_by_pattern) - current_ids):
            collection.delete(ids=sorted(ids_by_pattern[stale]))
            report.removed.append(stale)
    return report


def search_automation_patterns(
    query: str,
    top_k: int = 3,
    *,
    embedder: Embedder | None = None,
    persist_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> list[PatternMatch]:
    """Семантический поиск: возвращает до `top_k` наиболее подходящих паттернов (по умолчанию Top-3)."""
    query = query.strip()
    if not query:
        raise KnowledgeBaseError("Поисковый запрос пуст.")
    if top_k < 1:
        raise KnowledgeBaseError("top_k должен быть не меньше 1.")

    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    collection = _open_collection(_client(persist_dir, settings), embedder.model, create=False)
    if collection is None or collection.count() == 0:
        raise KnowledgeBaseError("База знаний ещё не построена. Сначала выполните индексацию (build_knowledge_base).")

    query_vector = embedder.embed([query])[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(collection.count(), MAX_CANDIDATE_CHUNKS),
        include=["documents", "metadatas", "distances"],
    )

    by_pattern: dict[str, list[ChunkHit]] = {}
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        by_pattern.setdefault(meta["pattern_id"], []).append(
            ChunkHit(
                chunk_id=chunk_id,
                pattern_id=meta["pattern_id"],
                title=meta["title"],
                source=meta["source"],
                section=meta["section"],
                text=text,
                score=round(1.0 - distance, 4),
            )
        )

    matches = [
        PatternMatch(
            pattern_id=pid,
            title=hits[0].title,
            source=hits[0].source,
            score=max(h.score for h in hits),
            chunks=sorted(hits, key=lambda h: h.score, reverse=True),
        )
        for pid, hits in by_pattern.items()
    ]
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:top_k]


def get_pattern_by_id(
    pattern_id: str,
    *,
    persist_dir: str | Path | None = None,
    embedding_model: str | None = None,
    settings: Settings | None = None,
) -> AutomationPattern | None:
    """Собирает паттерн целиком из индекса (эмбеддинги и ключ API не нужны). None — паттерна нет."""
    settings = settings or get_settings()
    collection = _open_collection(
        _client(persist_dir, settings), embedding_model or settings.embedding_model, create=False
    )
    if collection is None:
        return None
    found = collection.get(where={"pattern_id": pattern_id}, include=["documents", "metadatas"])
    if not found["ids"]:
        return None

    rows = sorted(zip(found["ids"], found["documents"], found["metadatas"]), key=lambda r: (r[2]["section_order"], r[0]))
    sections: dict[str, list[str]] = {}
    for _, text, meta in rows:
        sections.setdefault(meta["section"], []).append(text)
    first = rows[0][2]
    return AutomationPattern(
        pattern_id=pattern_id,
        title=first["title"],
        source=first["source"],
        sections={name: "\n\n".join(parts) for name, parts in sections.items()},
        content_hash=first["content_hash"],
    )


def list_indexed_patterns(
    *,
    persist_dir: str | Path | None = None,
    embedding_model: str | None = None,
    settings: Settings | None = None,
) -> list[dict]:
    """Паттерны, реально лежащие в индексе: [{pattern_id, title, source}] по возрастанию ID."""
    settings = settings or get_settings()
    collection = _open_collection(
        _client(persist_dir, settings), embedding_model or settings.embedding_model, create=False
    )
    if collection is None:
        return []
    metas = collection.get(include=["metadatas"])["metadatas"]
    unique = {m["pattern_id"]: {"pattern_id": m["pattern_id"], "title": m["title"], "source": m["source"]} for m in metas}
    return [unique[k] for k in sorted(unique)]


def build_query_from_process(spec: ProcessSpec, step_id: str | None = None) -> str:
    """Формирует поисковый запрос из ProcessSpec (только из известных данных — без домыслов).

    Запрос для операции содержит только факты об операции: название процесса и цель общие для всех
    операций и «перетягивают» эмбеддинг запроса, из-за чего разные операции находят одни и те же паттерны.
    """
    lines: list[str] = []
    if step_id is None:
        lines.append(f"Бизнес-процесс: {spec.name}.")
        if spec.goal:
            lines.append(f"Цель: {spec.goal}.")

    if step_id is not None:
        step = next((s for s in spec.steps if s.id == step_id), None)
        if step is None:
            raise KnowledgeBaseError(f"В процессе нет операции {step_id}.")
        lines.append(f"Операция: {step.name}.")
        if step.description:
            lines.append(step.description)
        if step.actor:
            lines.append(f"Исполнитель: {step.actor}.")
        if step.is_manual is True:
            lines.append("Выполняется вручную.")
        elif step.is_manual is False:
            lines.append("Уже автоматизирована.")
        if step.duration_minutes is not None:
            lines.append(f"Длительность: {step.duration_minutes:g} мин.")
    else:
        descriptions = []
        for i, step in enumerate(spec.steps, start=1):
            note = " (вручную)" if step.is_manual is True else ""
            descriptions.append(f"{i}) {step.name}{note}")
        lines.append("Операции: " + "; ".join(descriptions) + ".")
    return " ".join(lines)
