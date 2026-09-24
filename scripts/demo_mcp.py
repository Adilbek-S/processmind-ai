"""Демонстрация: три MCP-инструмента ProcessMind, вызванные через стандартный MCP-клиент по stdio.

Запуск: `python -m scripts.demo_mcp`

Сервер запускается отдельным подпроцессом (`python -m processmind.mcp.server`); клиент общается с ним
по JSON-RPC. Демо-процесс — эталон из scripts/demo_data.py (у части операций длительность не указана).
Предположения симуляции (длительности ПОСЛЕ автоматизации, число запусков) заданы здесь явно и
условно — это не данные о реальном процессе.
"""

from __future__ import annotations

import json
import sys

from processmind.mcp.client import McpToolError, SyncMCPClient
from scripts import demo_data

RUNS_PER_MONTH = 120  # условное число запусков в месяц (предположение демо)
# Предположения: какие операции автоматизируем и за сколько минут они будут выполняться после этого.
AUTOMATION_ASSUMPTIONS = [
    {"step_id": "step-2", "new_duration_minutes": 2},  # «Проверить комплектность заявки»: 30 -> 2
    {"step_id": "step-3", "new_duration_minutes": 20},  # «Согласовать бюджет»: 120 -> 20
]


def demo_process() -> dict:
    steps = []
    for i, s in enumerate(demo_data.TRUTH_STEPS, start=1):
        steps.append(
            {
                "id": f"step-{i}",
                "name": s["name"],
                "actor": s["actor"],
                "duration_minutes": s["duration_minutes"],
                "is_manual": s["is_manual"],
                "next_steps": [f"step-{i + 1}"] if i < len(demo_data.TRUTH_STEPS) else [],
            }
        )
    return {"process_id": "demo", "name": demo_data.PROCESS_NAME, "goal": demo_data.PROCESS_GOAL, "steps": steps}


def show(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    process = demo_process()
    with SyncMCPClient() as mcp:
        print(f"Подключено к MCP-серверу «{mcp.server_name}» (stdio, отдельный процесс)")
        print("Инструменты:", ", ".join(mcp.list_tools()))

        report = mcp.validate_process(process)
        print(f"\n=== validate_process === valid={report['valid']}, ошибок {len(report['errors'])}, предупреждений {len(report['warnings'])}")
        for w in report["warnings"]:
            print(f"  ! [{w['code']}] {w['path']}: {w['message']}")

        show(f"calculate_process_metrics (runs_per_month={RUNS_PER_MONTH})", mcp.calculate_process_metrics(process, RUNS_PER_MONTH))
        show("simulate_automation", mcp.simulate_automation(process, AUTOMATION_ASSUMPTIONS, RUNS_PER_MONTH))

        print("\n=== некорректные данные ===")
        broken = {**process, "steps": process["steps"] + [dict(process["steps"][0])]}  # дубликат ID
        print("validate_process ->", [(e["code"], e["path"]) for e in mcp.validate_process(broken)["errors"]])
        try:
            mcp.calculate_process_metrics(broken)
        except McpToolError as exc:
            print("calculate_process_metrics -> McpToolError:", exc)
        try:  # step-4 без длительности: сравнение до/после невозможно
            mcp.simulate_automation(process, [{"step_id": "step-4", "new_duration_minutes": 1}])
        except McpToolError as exc:
            print("simulate_automation ->", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
