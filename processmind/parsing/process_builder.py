"""Нормализация сырого вывода LLM в ProcessSpec.

Здесь происходит всё, что нельзя отдавать на откуп модели:
- перевод длительности в минуты;
- отбрасывание длительностей и участников, которых нет в исходном тексте (защита от галлюцинаций);
- проверка связей между операциями;
- итоговая валидация через Pydantic (ProcessSpec).
"""

from __future__ import annotations

import hashlib
import re

from pydantic import ValidationError

from processmind.models import ProcessSpec, ProcessStep
from processmind.parsing.schemas import ExtractionError, ExtractionResult, RawProcess, RawStep

UNNAMED_PROCESS = "Название не указано"

_MINUTES_PER_UNIT = {"seconds": 1 / 60, "minutes": 1.0, "hours": 60.0, "days": 1440.0}
_MIN_STEM_LEN = 4
_STEM_LEN = 5


def _norm(text: str) -> str:
    """Регистр и пробельные символы не важны при сверке цитат с источником."""
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip().casefold()


def _actor_in_text(actor: str, source_text_norm: str) -> bool:
    """Русская морфология: сверяем по основам слов («менеджер» ~ «менеджером»)."""
    words = [w for w in re.findall(r"\w+", actor.casefold()) if len(w) >= _MIN_STEM_LEN]
    if not words:  # короткое обозначение (например «ИТ») — только точное вхождение
        return actor.casefold().strip() in source_text_norm
    return any(w[:_STEM_LEN] in source_text_norm for w in words)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def build_process_spec(
    raw: RawProcess,
    *,
    source_name: str,
    source_type: str,
    source_bytes: bytes,
    model: str,
    source_text: str | None = None,
) -> ExtractionResult:
    """Собирает и валидирует ProcessSpec.

    `source_text` передаётся для PDF: по нему сверяются цитаты длительностей и участники.
    Для изображений исходного текста нет — сверка невозможна, поэтому такие значения
    остаются, но пользователь видит пометку и проверяет их в редакторе.
    """
    warnings: list[str] = []
    text_norm = _norm(source_text) if source_text is not None else None

    if not raw.steps:
        raise ExtractionError("Модель не нашла в документе ни одной операции процесса.")

    steps = sorted(raw.steps, key=lambda s: s.order)
    orders = [s.order for s in steps]
    if len(set(orders)) != len(orders):
        raise ExtractionError("Модель вернула операции с повторяющимися номерами — результат ненадёжен.")

    id_by_order = {s.order: f"step-{i + 1}" for i, s in enumerate(steps)}
    linear_fallback = not any(s.next_orders for s in steps)

    built: list[ProcessStep] = []
    for i, raw_step in enumerate(steps):
        name = _clean(raw_step.name)
        if not name:
            raise ExtractionError(f"У операции №{raw_step.order} нет названия.")

        actor = _clean(raw_step.actor)
        if actor and text_norm is not None and not _actor_in_text(actor, text_norm):
            warnings.append(f"«{name}»: участник «{actor}» не найден в тексте документа — отброшен.")
            actor = None

        duration = _resolve_duration(raw_step, name, text_norm, warnings)

        if linear_fallback:
            next_ids = [id_by_order[steps[i + 1].order]] if i + 1 < len(steps) else []
        else:
            next_ids = []
            for n in raw_step.next_orders:
                if n in id_by_order and n != raw_step.order:
                    if id_by_order[n] not in next_ids:
                        next_ids.append(id_by_order[n])
                else:
                    warnings.append(f"«{name}»: связь с несуществующей операцией №{n} отброшена.")

        built.append(
            ProcessStep(
                id=id_by_order[raw_step.order],
                name=name,
                description=_clean(raw_step.description),
                actor=actor,
                duration_minutes=duration,
                is_manual={"manual": True, "automated": False}.get(raw_step.execution_type),
                next_steps=next_ids,
            )
        )

    if linear_fallback and len(built) > 1:
        warnings.append("Связи между операциями не указаны явно — принят порядок следования в документе.")

    participants = [p for p in (_clean(p) for p in raw.participants) if p]
    if text_norm is not None:
        kept = [p for p in participants if _actor_in_text(p, text_norm)]
        for dropped in dict.fromkeys(p for p in participants if p not in kept):
            warnings.append(f"Участник «{dropped}» не найден в тексте документа — отброшен.")
        participants = kept
    actors = list(dict.fromkeys([s.actor for s in built if s.actor] + participants))

    process_name = _clean(raw.name)
    if process_name is None:
        warnings.append("Название процесса в документе не указано.")
    if _clean(raw.goal) is None:
        warnings.append("Цель процесса в документе не указана.")
    for note in raw.uncertainties:
        if _clean(note):
            warnings.append(f"Неуверенное распознавание: {note.strip()}")

    try:
        spec = ProcessSpec(
            process_id="proc-" + hashlib.sha1(source_bytes).hexdigest()[:8],
            name=process_name or UNNAMED_PROCESS,
            goal=_clean(raw.goal),
            steps=built,
            actors=actors,
            metadata={
                "source_file": source_name,
                "source_type": source_type,
                "model": model,
                "warnings": warnings,
            },
        )
    except ValidationError as exc:
        raise ExtractionError(f"Результат не прошёл проверку ProcessSpec: {exc}") from exc

    return ExtractionResult(spec=spec, warnings=warnings)


def _resolve_duration(
    step: RawStep,
    name: str,
    text_norm: str | None,
    warnings: list[str],
) -> float | None:
    """Длительность в минутах или None. `text_norm is None` — источник-изображение."""
    quote = _clean(step.duration_quote)
    if step.duration_value is None and step.duration_unit is None and quote is None:
        return None

    if step.duration_value is None or step.duration_unit is None or step.duration_value < 0 or not quote:
        warnings.append(f"«{name}»: длительность указана неполно или без цитаты из источника — отброшена.")
        return None
    if text_norm is not None and _norm(quote) not in text_norm:
        warnings.append(
            f"«{name}»: цитата длительности «{quote}» не найдена в тексте документа — длительность отброшена."
        )
        return None
    if text_norm is None:
        warnings.append(f"«{name}»: длительность «{quote}» распознана с изображения — проверьте её в таблице.")
    return round(step.duration_value * _MINUTES_PER_UNIT[step.duration_unit], 4)
