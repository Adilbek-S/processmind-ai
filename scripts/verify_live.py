"""Живая проверка распознавания на демо-файлах (РЕАЛЬНЫЕ вызовы OpenAI, нужен OPENAI_API_KEY).

Запуск: `python -m scripts.verify_live [--regen]`

Сравнивает результат с эталоном `samples/demo_process_truth.json`:
- полнота: найдены все операции, порядок совпадает;
- честность: нет длительностей/участников там, где в эталоне их нет;
- точность: длительности и участники совпадают там, где они указаны.
Код возврата 0 — все проверки обоих сценариев пройдены.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

from processmind.models import ProcessSpec
from processmind.parsing.schemas import ExtractionError
from processmind.parsing.service import extract_process
from scripts import demo_data

SAMPLES = demo_data.SAMPLES_DIR
NAME_MATCH_THRESHOLD = 0.6


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio()


def compare(spec: ProcessSpec, truth: dict) -> list[str]:
    """Возвращает список найденных расхождений (пусто — всё совпало)."""
    problems: list[str] = []
    expected = truth["steps"]
    if len(spec.steps) != len(expected):
        problems.append(f"число операций: получено {len(spec.steps)}, ожидалось {len(expected)}")

    for i, exp in enumerate(expected):
        if i >= len(spec.steps):
            problems.append(f"#{i + 1} «{exp['name']}»: операция не найдена")
            continue
        got = spec.steps[i]
        if _sim(got.name, exp["name"]) < NAME_MATCH_THRESHOLD:
            problems.append(f"#{i + 1} название: «{got.name}» ≠ «{exp['name']}»")
        if exp["duration_minutes"] is None and got.duration_minutes is not None:
            problems.append(f"#{i + 1} ВЫДУМАНА длительность {got.duration_minutes} мин")
        elif exp["duration_minutes"] is not None and got.duration_minutes != exp["duration_minutes"]:
            problems.append(f"#{i + 1} длительность: {got.duration_minutes} ≠ {exp['duration_minutes']}")
        if exp["actor"] is None and got.actor is not None:
            problems.append(f"#{i + 1} ВЫДУМАН участник «{got.actor}»")
        elif exp["actor"] is not None and (got.actor is None or _sim(got.actor, exp["actor"]) < NAME_MATCH_THRESHOLD):
            problems.append(f"#{i + 1} участник: «{got.actor}» ≠ «{exp['actor']}»")
        if exp["is_manual"] is None and got.is_manual is not None:
            problems.append(f"#{i + 1} ВЫДУМАН тип выполнения ({'ручная' if got.is_manual else 'автомат.'})")
        elif exp["is_manual"] is not None and got.is_manual != exp["is_manual"]:
            problems.append(f"#{i + 1} тип выполнения: {got.is_manual} ≠ {exp['is_manual']}")

    ids = [s.id for s in spec.steps]
    chain = all(s.next_steps == ([ids[i + 1]] if i + 1 < len(ids) else []) for i, s in enumerate(spec.steps))
    if not chain:
        problems.append("связи между операциями не образуют ожидаемую последовательность 1→2→…→N")
    return problems


def main() -> int:
    if "--regen" in sys.argv or not (SAMPLES / "demo_process.pdf").exists():
        demo_data.main()
    truth = json.loads((SAMPLES / "demo_process_truth.json").read_text(encoding="utf-8"))

    failed = False
    for filename in ("demo_process.pdf", "demo_process.png"):
        path: Path = SAMPLES / filename
        print(f"\n=== {filename} ===")
        try:
            result = extract_process(filename, path.read_bytes())
        except ExtractionError as exc:
            print(f"ОШИБКА РАСПОЗНАВАНИЯ: {exc}")
            failed = True
            continue

        for i, s in enumerate(result.spec.steps, start=1):
            links = ",".join(n.removeprefix("step-") for n in s.next_steps) or "—"
            print(f"{i}. {s.name} | {s.actor or '—'} | {s.duration_minutes if s.duration_minutes is not None else '—'} мин | {s.is_manual} | далее: {links}")
        for w in result.warnings:
            print(f"  ! {w}")
        problems = compare(result.spec, truth)
        print("РЕЗУЛЬТАТ:", "OK, все проверки пройдены" if not problems else "РАСХОЖДЕНИЯ")
        for p in problems:
            print("  -", p)
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
