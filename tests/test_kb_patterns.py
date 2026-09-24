"""Тесты корпуса паттернов, парсинга Markdown и chunking."""

import re

import pytest

from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import DEFAULT_KB_DIR
from processmind.rag.patterns import REQUIRED_SECTIONS, chunk_pattern, load_patterns, parse_pattern
from tests.helpers_kb import pattern_markdown

REQUIRED_TOPICS = {  # обязательные темы из постановки -> фрагмент названия паттерна
    "маршрутизац": "автоматическая маршрутизация заявок",
    "согласован": "электронное согласование",
    "обязательных полей": "проверка обязательных полей",
    "уведомлен": "автоматические уведомления",
    "повторного ввода": "устранение повторного ввода данных",
    "интеграция информационных систем": "интеграция информационных систем",
    "ocr": "OCR",
    "регистрация обращений": "автоматическая регистрация обращений",
    "сроков": "контроль сроков",
    "шаблонов": "шаблоны документов",
    "классификация обращений": "автоматическая классификация обращений",
    "отчётности": "формирование отчётности",
}


# ---------- корпус ----------


@pytest.fixture(scope="module")
def corpus():
    return load_patterns(DEFAULT_KB_DIR)


def test_corpus_has_at_least_12_patterns_with_unique_ids(corpus):
    assert len(corpus) >= 12
    assert len({p.pattern_id for p in corpus}) == len(corpus)


def test_every_pattern_has_all_required_sections_non_empty(corpus):
    for p in corpus:
        assert set(REQUIRED_SECTIONS) <= set(p.sections), p.pattern_id
        assert all(body.strip() for body in p.sections.values()), p.pattern_id


@pytest.mark.parametrize("fragment", list(REQUIRED_TOPICS))
def test_required_topics_are_covered(corpus, fragment):
    assert any(fragment.casefold() in p.title.casefold() for p in corpus), REQUIRED_TOPICS[fragment]


def test_each_pattern_is_its_own_markdown_file(corpus):
    files = sorted(DEFAULT_KB_DIR.glob("*.md"))
    assert len(files) == len(corpus)
    assert {f.name for f in files} == {p.source for p in corpus}


def test_corpus_has_no_invented_numbers_or_normative_references(corpus):
    """Эффекты качественные; ссылок на (вымышленные) нормативные документы нет."""
    forbidden = re.compile(r"\d+\s*%|\bГОСТ\b|\bФЗ\b|\bСНиП\b|\bISO\b|№\s*\d|\bприказ|\bпостановлен|\bзакон", re.IGNORECASE)
    for p in corpus:
        for section, body in p.sections.items():
            assert not forbidden.search(body), f"{p.pattern_id} / {section}: {forbidden.search(body).group()}"


# ---------- парсинг ----------


def test_parse_valid_pattern():
    p = parse_pattern(pattern_markdown("AP-777", "Тестовый паттерн"), source="AP-777-test.md")
    assert (p.pattern_id, p.title, p.source) == ("AP-777", "Тестовый паттерн", "AP-777-test.md")
    assert list(p.sections) == list(REQUIRED_SECTIONS)


def test_parse_handles_crlf_and_bom():
    md = "﻿" + pattern_markdown("AP-777", "Т").replace("\n", "\r\n")
    assert parse_pattern(md, "AP-777-t.md").pattern_id == "AP-777"


def test_content_hash_changes_with_content():
    a = parse_pattern(pattern_markdown("AP-777", "Т"), "AP-777-t.md")
    b = parse_pattern(pattern_markdown("AP-777", "Т", problem="Другой текст."), "AP-777-t.md")
    assert a.content_hash != b.content_hash


@pytest.mark.parametrize(
    ("markdown", "source", "message"),
    [
        (pattern_markdown("AP-777", "Т").replace("**ID:** AP-777\n", ""), "AP-777-t.md", "идентификатор"),
        (pattern_markdown("AP-777", "Т").replace("# Т\n", ""), "AP-777-t.md", "заголовок"),
        (pattern_markdown("AP-777", "Т"), "AP-999-t.md", "имя файла"),
        (pattern_markdown("AP-777", "Т").replace("## Ограничения применения", "## Другое"), "AP-777-t.md", "Ограничения применения"),
        (pattern_markdown("AP-777", "Т", limitations=" "), "AP-777-t.md", "пуст"),
        (pattern_markdown("AP-777", "Т") + "\n## Описание проблемы\nЕщё раз\n", "AP-777-t.md", "дважды"),
    ],
)
def test_invalid_patterns_rejected_with_source_in_message(markdown, source, message):
    with pytest.raises(KnowledgeBaseError, match=message) as exc:
        parse_pattern(markdown, source)
    assert source in str(exc.value)


def test_load_rejects_duplicate_ids(tmp_path):
    (tmp_path / "AP-001-a.md").write_text(pattern_markdown("AP-001", "A"), encoding="utf-8")
    (tmp_path / "AP-001-b.md").write_text(pattern_markdown("AP-001", "B"), encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="Дублирующийся ID"):
        load_patterns(tmp_path)


def test_load_rejects_missing_or_empty_directory(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="не найден"):
        load_patterns(tmp_path / "nope")
    with pytest.raises(KnowledgeBaseError, match="нет Markdown"):
        load_patterns(tmp_path)


# ---------- chunking ----------


def test_one_chunk_per_section_with_full_metadata(corpus):
    for p in corpus:
        chunks = chunk_pattern(p)
        assert [c.section for c in chunks] == list(p.sections)
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        for c in chunks:
            assert (c.pattern_id, c.title, c.source) == (p.pattern_id, p.title, p.source)
            assert c.chunk_id.startswith(p.pattern_id + ":")
            assert c.text == p.sections[c.section]


def test_chunk_ids_are_stable_between_runs(corpus):
    assert [c.chunk_id for c in chunk_pattern(corpus[0])] == [c.chunk_id for c in chunk_pattern(corpus[0])]


def test_embedding_text_carries_title_and_section(corpus):
    chunk = chunk_pattern(corpus[0])[2]
    assert corpus[0].title in chunk.embedding_text and chunk.section in chunk.embedding_text


def test_long_section_split_by_paragraphs_without_losing_text():
    long_body = "\n\n".join(f"Абзац номер {i}. " + "слово " * 40 for i in range(10))
    p = parse_pattern(pattern_markdown("AP-777", "Т", problem=long_body), "AP-777-t.md")
    chunks = [c for c in chunk_pattern(p, max_chars=600) if c.section == "Описание проблемы"]
    assert len(chunks) > 1
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(len(c.text) <= 600 for c in chunks)
    expected = "\n\n".join(paragraph.strip() for paragraph in long_body.split("\n\n"))
    assert "\n\n".join(c.text for c in chunks) == expected


def test_extra_sections_are_indexed_too():
    md = pattern_markdown("AP-777", "Т") + "\n## Примечание\nДополнительный раздел.\n"
    chunks = chunk_pattern(parse_pattern(md, "AP-777-t.md"))
    assert chunks[-1].section == "Примечание" and chunks[-1].chunk_id == "AP-777:extra1"
