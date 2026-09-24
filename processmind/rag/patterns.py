"""Паттерны автоматизации: загрузка Markdown-документов и chunking по смысловым разделам.

Формат документа:

    # Название паттерна

    **ID:** AP-001

    ## Описание проблемы
    ...
    ## Условия применимости
    ...
    ## Предлагаемый подход
    ...
    ## Ожидаемый качественный эффект
    ...
    ## Ограничения применения
    ...

Один раздел = один chunk (слишком длинный раздел делится по абзацам).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel

from processmind.rag.errors import KnowledgeBaseError

# Обязательные разделы и стабильные ключи для chunk_id.
REQUIRED_SECTIONS: dict[str, str] = {
    "Описание проблемы": "problem",
    "Условия применимости": "applicability",
    "Предлагаемый подход": "approach",
    "Ожидаемый качественный эффект": "effect",
    "Ограничения применения": "limitations",
}
MAX_CHUNK_CHARS = 1500

_ID_RE = re.compile(r"^\*\*ID:\*\*\s*([A-Z]{2,}-\d{3,})\s*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class AutomationPattern(BaseModel):
    """Паттерн автоматизации целиком (разделы в порядке документа)."""

    pattern_id: str
    title: str
    source: str
    sections: dict[str, str]
    content_hash: str


class PatternChunk(BaseModel):
    """Фрагмент паттерна — единица индексации и поиска."""

    chunk_id: str
    pattern_id: str
    title: str
    source: str
    section: str
    section_order: int
    text: str

    @property
    def embedding_text(self) -> str:
        """Текст для эмбеддинга: раздел без названия паттерна теряет контекст."""
        return f"{self.title}. {self.section}.\n{self.text}"


def parse_pattern(markdown: str, source: str) -> AutomationPattern:
    markdown = markdown.replace("\r\n", "\n").lstrip("﻿")

    title_match = _TITLE_RE.search(markdown)
    if not title_match:
        raise KnowledgeBaseError(f"{source}: не найден заголовок паттерна (строка вида «# Название»).")
    id_match = _ID_RE.search(markdown)
    if not id_match:
        raise KnowledgeBaseError(f"{source}: не найден идентификатор (строка вида «**ID:** AP-001»).")
    pattern_id = id_match.group(1)
    if not Path(source).name.startswith(pattern_id):
        raise KnowledgeBaseError(f"{source}: имя файла должно начинаться с ID паттерна «{pattern_id}».")

    headings = list(_SECTION_RE.finditer(markdown))
    sections: dict[str, str] = {}
    for i, heading in enumerate(headings):
        name = heading.group(1)
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        body = markdown[heading.end() : end].strip()
        if name in sections:
            raise KnowledgeBaseError(f"{source}: раздел «{name}» встречается дважды.")
        if not body:
            raise KnowledgeBaseError(f"{source}: раздел «{name}» пуст.")
        sections[name] = body

    missing = [name for name in REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise KnowledgeBaseError(f"{source}: нет обязательных разделов: {', '.join(missing)}.")

    return AutomationPattern(
        pattern_id=pattern_id,
        title=title_match.group(1),
        source=source,
        sections=sections,
        content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
    )


def load_patterns(directory: str | Path) -> list[AutomationPattern]:
    """Загружает все `*.md` из каталога. Любая ошибка формата останавливает загрузку целиком."""
    directory = Path(directory)
    if not directory.is_dir():
        raise KnowledgeBaseError(f"Каталог базы знаний не найден: {directory}")
    files = sorted(directory.glob("*.md"))
    if not files:
        raise KnowledgeBaseError(f"В каталоге {directory} нет Markdown-документов.")

    patterns: dict[str, AutomationPattern] = {}
    for path in files:
        pattern = parse_pattern(path.read_text(encoding="utf-8"), source=path.name)
        if pattern.pattern_id in patterns:
            raise KnowledgeBaseError(
                f"Дублирующийся ID {pattern.pattern_id}: {patterns[pattern.pattern_id].source} и {path.name}."
            )
        patterns[pattern.pattern_id] = pattern
    return list(patterns.values())


def chunk_pattern(pattern: AutomationPattern, max_chars: int = MAX_CHUNK_CHARS) -> list[PatternChunk]:
    """Делит паттерн на chunk'и по разделам; длинный раздел — по границам абзацев."""
    chunks: list[PatternChunk] = []
    extra = 0
    for order, (section, body) in enumerate(pattern.sections.items()):
        key = REQUIRED_SECTIONS.get(section)
        if key is None:
            extra += 1
            key = f"extra{extra}"
        parts = _split_paragraphs(body, max_chars)
        for n, part in enumerate(parts, start=1):
            suffix = "" if len(parts) == 1 else f":{n}"
            chunks.append(
                PatternChunk(
                    chunk_id=f"{pattern.pattern_id}:{key}{suffix}",
                    pattern_id=pattern.pattern_id,
                    title=pattern.title,
                    source=pattern.source,
                    section=section,
                    section_order=order,
                    text=part,
                )
            )
    return chunks


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.split("\n\n") if p.strip()):
        # Абзац длиннее лимита остаётся целым: резать посреди предложения хуже, чем превысить лимит.
        if current and len(current) + len(paragraph) + 2 > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts
