"""Загрузка Skill `process-analysis`: методика анализа, которая реально попадает в промпт LLM.

Файл `.claude/skills/process-analysis/SKILL.md` — одновременно Skill Claude Code и источник системного
промпта узла `generate_recommendations`. Workflow читает его при каждом запуске, поэтому правка
методики немедленно меняет поведение LLM-узла, а в отчёт попадает хэш применённой версии.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

SKILL_PATH = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "process-analysis" / "SKILL.md"


class SkillError(Exception):
    """Файл Skill отсутствует или имеет неверный формат."""


class Skill(BaseModel):
    name: str
    description: str
    body: str  # методика без frontmatter — используется как системный промпт
    sha256: str  # хэш всего файла: фиксирует, какая версия методики применена
    path: str


def load_skill(path: str | Path | None = None) -> Skill:
    path = Path(path or SKILL_PATH)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillError(f"Не удалось прочитать Skill {path}: {exc}") from exc

    normalized = text.replace("\r\n", "\n").lstrip("﻿")
    if not normalized.startswith("---\n"):
        raise SkillError(f"{path}: нет frontmatter (файл должен начинаться со строки ---)")
    header, sep, body = normalized[4:].partition("\n---\n")
    if not sep:
        raise SkillError(f"{path}: frontmatter не закрыт строкой ---")

    fields: dict[str, str] = {}
    for line in header.splitlines():
        key, colon, value = line.partition(":")
        if colon:
            fields[key.strip()] = value.strip()
    for required in ("name", "description"):
        if not fields.get(required):
            raise SkillError(f"{path}: в frontmatter нет обязательного поля «{required}»")
    if not body.strip():
        raise SkillError(f"{path}: пустая методика (нет текста после frontmatter)")

    return Skill(
        name=fields["name"],
        description=fields["description"],
        body=body.strip(),
        sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        path=str(path),
    )
