"""Сквозной анализ демо-процесса главным LangGraph workflow (РЕАЛЬНЫЕ OpenAI, RAG и MCP-сервер).

Запуск: `python -m scripts.run_analysis [--runs 120] [--select all|none|R1,R3]`

Шаги: validate_process -> calculate_metrics -> retrieve_automation_patterns -> generate_recommendations
-> validate_recommendations -> [ПАУЗА human_approval] -> simulate_automation -> generate_report.

Пауза настоящая: скрипт получает запрос подтверждения, печатает рекомендации и только затем «принимает
решение пользователя» — в демо это правило `--select` (по умолчанию: все рекомендации, которые можно
смоделировать, с длительностями, предложенными LLM). Число запусков `--runs` — условное допущение демо.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from processmind.analysis.report import report_to_markdown
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import build_knowledge_base
from processmind.workflow.graph import build_workflow, create_default_deps, resume_analysis, start_analysis
from scripts.demo_mcp import demo_process

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=120, help="условное число запусков процесса в месяц (допущение)")
    parser.add_argument("--select", default="all", help="all | none | список rec_id через запятую, например R1,R3")
    parser.add_argument(
        "--assume",
        default="",
        help="допущения пользователя о длительности ПОСЛЕ автоматизации в минутах: R1=5,R3=20 (LLM их обычно не даёт)",
    )
    args = parser.parse_args()

    try:
        report = build_knowledge_base()
        print(f"База знаний: паттернов {report.total_patterns}, отправлено на эмбеддинг {report.embedded_chunks} (0 — индекс уже актуален)")
    except KnowledgeBaseError as exc:
        print(f"База знаний недоступна: {exc}")

    deps = create_default_deps()
    try:
        graph = build_workflow(deps)
        run = start_analysis(graph, demo_process(), args.runs)

        if run.status == "finished":
            print(f"\nWorkflow завершился без паузы: статус {run.report.status}")
            for error in run.report.errors:
                print(" -", error)
            return 1

        print("\n=== ПАУЗА: human_approval — workflow ждёт решения пользователя ===")
        for r in run.approval_request["recommendations"]:
            print(f"\n{r['rec_id']}  {r['step_name']} ({r['step_id']})  [паттерн: {r['pattern_id'] or 'нет — не опирается на БЗ'}]")
            print(f"   проблема:  {r['problem']}")
            print(f"   решение:   {r['solution']}")
            print(f"   обоснование: {r['rationale']}")
            print(f"   эффект:    {r['expected_effect']}")
            print(f"   факты: {r['process_facts']}")
            print(f"   длительность: {r['original_duration_minutes']} -> {r['new_duration_minutes']} мин  (допущение LLM: {r['duration_basis']})")
            print(f"   можно смоделировать: {r['simulatable']}" + (f"   предупреждения: {r['warnings']}" if r["warnings"] else ""))

        assumptions = {k.strip(): float(v) for k, _, v in (p.partition("=") for p in args.assume.split(",") if p.strip())}
        recs = {r["rec_id"]: r for r in run.approval_request["recommendations"]}
        if args.select == "all":  # все, что можно смоделировать: есть допущение LLM или пользователя
            chosen = [rid for rid, r in recs.items() if r["simulatable"] or (rid in assumptions and r["original_duration_minutes"] is not None)]
        elif args.select == "none":
            chosen = []
        else:
            chosen = [x.strip() for x in args.select.split(",") if x.strip()]
        selections = [{"rec_id": rid, **({"new_duration_minutes": assumptions[rid]} if rid in assumptions else {})} for rid in chosen]
        print(f"\n>>> Решение пользователя (демо-правило --select={args.select}): {chosen}")
        for rid, minutes in assumptions.items():
            if rid in recs:
                print(f"    допущение пользователя для {rid}: {recs[rid]['original_duration_minutes']} -> {minutes:g} мин")

        final = resume_analysis(graph, run.thread_id, {"selections": selections})
        if final.status != "finished":
            print("Решение отклонено:", final.approval_request.get("error"))
            return 1

        report = final.report
        print("\n" + "=" * 70 + "\n" + report_to_markdown(report))
        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "analysis_report_demo.json").write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT_DIR / "analysis_report_demo.md").write_text(report_to_markdown(report), encoding="utf-8")
        print(f"Отчёт сохранён: {OUT_DIR / 'analysis_report_demo.json'}")
        return 0 if report.status in ("completed", "no_selection") else 1
    finally:
        deps.close()


if __name__ == "__main__":
    sys.exit(main())
