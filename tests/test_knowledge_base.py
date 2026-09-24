"""Тесты индексации и поиска Automation Knowledge Base (ChromaDB).

Эмбеддер подменён лексическим двойником (см. helpers_kb) — тесты проверяют обвязку RAG, а не
семантическое качество. Качество: `python -m scripts.verify_kb_live` (настоящие эмбеддинги).
"""

import shutil

import chromadb
import pytest

from processmind.config import Settings
from processmind.models import ProcessSpec, ProcessStep
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import (
    DEFAULT_KB_DIR,
    ChunkHit,
    build_knowledge_base,
    build_query_from_process,
    get_pattern_by_id,
    list_indexed_patterns,
    search_automation_patterns,
)
from processmind.rag.patterns import chunk_pattern, load_patterns
from tests.helpers_kb import FakeEmbedder, pattern_markdown

N_PATTERNS = len(list(DEFAULT_KB_DIR.glob("*.md")))
N_CHUNKS = sum(len(chunk_pattern(p)) for p in load_patterns(DEFAULT_KB_DIR))
NO_KEY = Settings(_env_file=None, openai_api_key=None)


@pytest.fixture
def kb_dir(tmp_path):
    """Копия реальной базы знаний, которую тесты могут менять."""
    target = tmp_path / "kb"
    shutil.copytree(DEFAULT_KB_DIR, target)
    return target


@pytest.fixture
def store(tmp_path):
    return tmp_path / "chroma"


def stored_chunk_count(store, model="fake-embedding-v1") -> int:
    client = chromadb.PersistentClient(path=str(store))
    return client.get_collection(f"automation_patterns__{model}", embedding_function=None).count()


# ---------- индексация ----------


def test_build_indexes_every_chunk_with_metadata(kb_dir, store):
    emb = FakeEmbedder()
    report = build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)

    assert report.total_patterns == report.added.__len__() == N_PATTERNS >= 12
    assert report.total_chunks == report.embedded_chunks == N_CHUNKS
    assert stored_chunk_count(store) == N_CHUNKS

    client = chromadb.PersistentClient(path=str(store))
    col = client.get_collection("automation_patterns__fake-embedding-v1", embedding_function=None)
    got = col.get(include=["metadatas", "documents", "embeddings"])
    for chunk_id, meta, doc, vec in zip(got["ids"], got["metadatas"], got["documents"], got["embeddings"]):
        assert meta["chunk_id"] == chunk_id
        assert all(meta[k] for k in ("pattern_id", "title", "source", "section", "chunk_id"))
        assert doc and len(vec) > 0


def test_embeddings_are_computed_for_title_and_section_context(kb_dir, store):
    emb = FakeEmbedder()
    build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    texts = [t for batch in emb.calls for t in batch]
    assert any(t.startswith("Автоматическая маршрутизация заявок. Описание проблемы.") for t in texts)


def test_reindexing_unchanged_kb_creates_no_duplicates_and_no_embedding_calls(kb_dir, store):
    emb = FakeEmbedder()
    build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    calls_before = len(emb.calls)

    second = build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    third = build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)

    assert len(second.unchanged) == N_PATTERNS and not second.added and not second.updated
    assert second.embedded_chunks == third.embedded_chunks == 0
    assert len(emb.calls) == calls_before
    assert stored_chunk_count(store) == N_CHUNKS


def test_changed_document_is_replaced_not_duplicated(kb_dir, store):
    emb = FakeEmbedder()
    build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)

    path = next(kb_dir.glob("AP-003-*.md"))
    path.write_text(path.read_text(encoding="utf-8") + "\n## Примечание\nДополнительный раздел.\n", encoding="utf-8")
    report = build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)

    assert report.updated == ["AP-003"] and len(report.unchanged) == N_PATTERNS - 1
    assert report.embedded_chunks == 6  # только изменённый паттерн: 5 разделов + новый
    assert stored_chunk_count(store) == N_CHUNKS + 1
    assert "Дополнительный раздел" in get_pattern_by_id("AP-003", persist_dir=store, embedding_model="fake-embedding-v1").sections["Примечание"]


def test_removed_section_leaves_no_stale_chunks(kb_dir, store):
    path = next(kb_dir.glob("AP-003-*.md"))
    original = path.read_text(encoding="utf-8")
    path.write_text(original + "\n## Примечание\nВременный раздел.\n", encoding="utf-8")
    emb = FakeEmbedder()
    build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)

    path.write_text(original, encoding="utf-8")
    build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    assert stored_chunk_count(store) == N_CHUNKS
    assert "Примечание" not in get_pattern_by_id("AP-003", persist_dir=store, embedding_model=emb.model).sections


def test_deleted_document_is_pruned_unless_disabled(kb_dir, store):
    emb = FakeEmbedder()
    build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    next(kb_dir.glob("AP-014-*.md")).unlink()

    kept = build_knowledge_base(kb_dir, embedder=emb, persist_dir=store, prune=False)
    assert kept.removed == [] and get_pattern_by_id("AP-014", persist_dir=store, embedding_model=emb.model)

    pruned = build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    assert pruned.removed == ["AP-014"]
    assert get_pattern_by_id("AP-014", persist_dir=store, embedding_model=emb.model) is None


def test_embedding_failure_leaves_existing_index_untouched(kb_dir, store):
    emb = FakeEmbedder()
    build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    path = next(kb_dir.glob("AP-001-*.md"))
    path.write_text(path.read_text(encoding="utf-8").replace("заявк", "обращени"), encoding="utf-8")

    with pytest.raises(RuntimeError):
        build_knowledge_base(kb_dir, embedder=FakeEmbedder(fail_after_calls=0), persist_dir=store)
    assert stored_chunk_count(store) == N_CHUNKS
    assert "заявк" in get_pattern_by_id("AP-001", persist_dir=store, embedding_model=emb.model).sections["Описание проблемы"]


def test_invalid_document_aborts_before_touching_index(kb_dir, store):
    (kb_dir / "AP-099-broken.md").write_text("# Без ID\n", encoding="utf-8")
    emb = FakeEmbedder()
    with pytest.raises(KnowledgeBaseError, match="AP-099-broken.md"):
        build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    assert emb.calls == []


def test_different_embedding_models_use_separate_collections(kb_dir, store):
    build_knowledge_base(kb_dir, embedder=FakeEmbedder("model-a"), persist_dir=store)
    build_knowledge_base(kb_dir, embedder=FakeEmbedder("model-b"), persist_dir=store)
    assert stored_chunk_count(store, "model-a") == stored_chunk_count(store, "model-b") == N_CHUNKS


def test_build_without_api_key_fails_loudly_instead_of_faking_embeddings(kb_dir, store):
    with pytest.raises(KnowledgeBaseError, match="OPENAI_API_KEY"):
        build_knowledge_base(kb_dir, persist_dir=store, settings=NO_KEY)


# ---------- поиск ----------


@pytest.fixture
def indexed(kb_dir, store):
    emb = FakeEmbedder()
    build_knowledge_base(kb_dir, embedder=emb, persist_dir=store)
    return emb, store


def test_search_returns_top3_distinct_patterns_sorted_by_score(indexed):
    emb, store = indexed
    matches = search_automation_patterns("сотрудники вручную пересылают заявки исполнителям", embedder=emb, persist_dir=store)

    assert len(matches) == 3
    assert len({m.pattern_id for m in matches}) == 3
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)


def test_every_hit_carries_pattern_id_title_source_section_chunk_id(indexed):
    emb, store = indexed
    for match in search_automation_patterns("распознавание текста на сканах документов", embedder=emb, persist_dir=store):
        assert match.pattern_id and match.title and match.source
        assert match.score == match.chunks[0].score == max(c.score for c in match.chunks)
        for hit in match.chunks:
            assert isinstance(hit, ChunkHit)
            assert hit.pattern_id == match.pattern_id and hit.source == match.source and hit.title == match.title
            assert hit.section and hit.chunk_id.startswith(match.pattern_id + ":") and hit.text
        assert [c.score for c in match.chunks] == sorted((c.score for c in match.chunks), reverse=True)


def test_search_uses_vector_similarity_query_equal_to_chunk_scores_top(indexed, kb_dir):
    emb, store = indexed
    pattern = next(p for p in load_patterns(kb_dir) if p.pattern_id == "AP-009")
    chunk = chunk_pattern(pattern)[1]

    (top, *_) = search_automation_patterns(chunk.embedding_text, embedder=emb, persist_dir=store)
    assert top.pattern_id == "AP-009"
    assert top.chunks[0].chunk_id == chunk.chunk_id
    assert top.chunks[0].score == pytest.approx(1.0, abs=1e-3)  # косинусная метрика: тот же вектор = 1


def test_search_embeds_only_the_query(indexed):
    emb, store = indexed
    calls_before = len(emb.calls)
    search_automation_patterns("контроль сроков", embedder=emb, persist_dir=store)
    assert emb.calls[calls_before:] == [["контроль сроков"]]


def test_top_k_is_respected(indexed):
    emb, store = indexed
    assert len(search_automation_patterns("заявка", top_k=1, embedder=emb, persist_dir=store)) == 1
    assert len(search_automation_patterns("заявка", top_k=5, embedder=emb, persist_dir=store)) == 5


def test_search_errors(indexed, tmp_path):
    emb, store = indexed
    with pytest.raises(KnowledgeBaseError, match="пуст"):
        search_automation_patterns("   ", embedder=emb, persist_dir=store)
    with pytest.raises(KnowledgeBaseError, match="top_k"):
        search_automation_patterns("x", top_k=0, embedder=emb, persist_dir=store)
    with pytest.raises(KnowledgeBaseError, match="не построена"):
        search_automation_patterns("x", embedder=emb, persist_dir=tmp_path / "empty_store")
    with pytest.raises(KnowledgeBaseError, match="не построена"):  # индекс есть, но для другой модели
        search_automation_patterns("x", embedder=FakeEmbedder("other-model"), persist_dir=store)
    with pytest.raises(KnowledgeBaseError, match="OPENAI_API_KEY"):
        search_automation_patterns("x", persist_dir=store, settings=NO_KEY)


# ---------- get_pattern_by_id / список ----------


def test_get_pattern_by_id_returns_full_pattern_from_index(indexed, kb_dir):
    emb, store = indexed
    expected = next(p for p in load_patterns(kb_dir) if p.pattern_id == "AP-007")
    got = get_pattern_by_id("AP-007", persist_dir=store, embedding_model=emb.model)

    assert got is not None
    assert (got.pattern_id, got.title, got.source) == (expected.pattern_id, expected.title, expected.source)
    assert got.sections == expected.sections
    assert list(got.sections) == list(expected.sections)  # порядок разделов сохранён


def test_get_pattern_by_id_unknown_or_not_built_returns_none(indexed, tmp_path):
    emb, store = indexed
    assert get_pattern_by_id("AP-999", persist_dir=store, embedding_model=emb.model) is None
    assert get_pattern_by_id("AP-001", persist_dir=tmp_path / "nothing", embedding_model=emb.model) is None


def test_get_pattern_reassembles_split_sections(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    long_body = "\n\n".join(f"Абзац {i}. " + "слово " * 200 for i in range(4))
    (kb / "AP-500-long.md").write_text(pattern_markdown("AP-500", "Длинный", problem=long_body), encoding="utf-8")
    emb = FakeEmbedder()
    build_knowledge_base(kb, embedder=emb, persist_dir=tmp_path / "s")

    got = get_pattern_by_id("AP-500", persist_dir=tmp_path / "s", embedding_model=emb.model)
    assert got.sections["Описание проблемы"] == "\n\n".join(p.strip() for p in long_body.split("\n\n"))


def test_list_indexed_patterns(indexed):
    emb, store = indexed
    listing = list_indexed_patterns(persist_dir=store, embedding_model=emb.model)
    assert len(listing) == N_PATTERNS
    assert listing[0]["pattern_id"] == "AP-001" and listing[0]["title"] and listing[0]["source"]
    assert list_indexed_patterns(persist_dir=store, embedding_model="unknown") == []


# ---------- запрос из ProcessSpec ----------


def make_spec() -> ProcessSpec:
    return ProcessSpec(
        process_id="p",
        name="Обработка заявки",
        goal="Ответить клиенту",
        steps=[
            ProcessStep(id="step-1", name="Проверить данные", actor="Менеджер", duration_minutes=30, is_manual=True),
            ProcessStep(id="step-2", name="Выставить счёт"),
        ],
    )


def test_query_for_step_uses_only_known_facts():
    q = build_query_from_process(make_spec(), "step-1")
    assert "Проверить данные" in q and "Менеджер" in q and "вручную" in q and "30 мин" in q
    bare = build_query_from_process(make_spec(), "step-2")
    assert "Выставить счёт" in bare
    assert not any(word in bare for word in ("Исполнитель", "вручную", "Длительность"))  # неизвестное не домысливается


def test_step_query_has_no_shared_process_context():
    q = build_query_from_process(make_spec(), "step-1")
    assert "Обработка заявки" not in q and "Ответить клиенту" not in q  # общий контекст стирает различия между операциями


def test_query_for_whole_process_lists_operations():
    q = build_query_from_process(make_spec())
    assert "Обработка заявки" in q and "1) Проверить данные (вручную)" in q and "2) Выставить счёт" in q


def test_query_for_unknown_step_rejected():
    with pytest.raises(KnowledgeBaseError):
        build_query_from_process(make_spec(), "step-99")
