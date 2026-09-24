"""Проверка трассировки LangSmith: `python -m evals.verify_tracing`.

Выполняет реальный анализ демо-процесса и проверяет, что трейс появился в вашем проекте LangSmith. Если LangSmith не
настроен — печатает ошибку и инструкцию (код возврата 2), успех не имитируется.
"""

from __future__ import annotations

import sys

from evals.tracing_check import TracingUnavailable, verify
from scripts.demo_mcp import demo_process


def main() -> int:
    try:
        result = verify(demo_process())
    except TracingUnavailable as exc:
        print(f"ОШИБКА: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - показываем причину, а не падаем трассировкой
        print(f"ОШИБКА при выполнении проверки: {exc}")
        return 1

    print(f"Проект LangSmith: {result['project']}\nTrace ID: {result['trace_id']}\nСтатус анализа: {result['status']} ({result['note']})")
    print(f"Runs в трейсе: {result['runs']} {result['by_type']}; токены LLM: {result['tokens']}; длительность: {result['duration_s']}")
    print(f"Runs с ошибкой: {result['error_runs'] or 'нет'}")
    for name, passed in result["checks"].items():
        print(f"  [{'OK' if passed else 'НЕТ'}] {name}")
    print("\nОткройте https://smith.langchain.com -> Tracing -> проект " + result["project"] + " и найдите трейс tracing_verification.")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
