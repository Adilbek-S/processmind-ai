"""Собственный MCP-сервер ProcessMind (Python MCP SDK, высокоуровневый API `MCPServer` — бывший FastMCP).

Запускается отдельным процессом с транспортом stdio:

    python -m processmind.mcp.server

Инструменты:
- validate_process          — проверка структуры и полноты процесса (ошибки и предупреждения);
- calculate_process_metrics — детерминированный расчёт метрик процесса (обычный Python, без LLM);
- simulate_automation       — модельная оценка эффекта автоматизации при ЗАДАННЫХ предположениях;
- search_knowledge_base     — семантический поиск паттернов автоматизации (RAG);
- get_process_spec_schema   — JSON-схема ProcessSpec.

stdout — канал протокола: в сервере нельзя использовать print(); диагностика уходит в stderr.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from processmind.config import get_settings
from processmind.mcp import process_tools
from processmind.mcp.process_tools import AutomationChange, ProcessMetrics, SimulationResult, ValidationReport
from processmind.models import ProcessSpec

settings = get_settings()
mcp_server = MCPServer(settings.mcp_server_name)

_PROCESS_ARG = "Бизнес-процесс в формате ProcessSpec (см. инструмент get_process_spec_schema)"


@mcp_server.tool()
def validate_process(process: dict[str, Any]) -> ValidationReport:
    """Проверяет структуру бизнес-процесса.

    Проверки: соответствие схеме ProcessSpec, обязательные поля, уникальность ID операций,
    корректность длительностей (отрицательные и нечисловые — ошибки; неуказанные и нулевые — предупреждения),
    целостность связей между операциями, циклы и недостижимые операции.
    Некорректные данные не приводят к сбою: они возвращаются списком errors/warnings.
    """
    return process_tools.validate_process_data(process)


@mcp_server.tool()
def calculate_process_metrics(process: dict[str, Any], runs_per_month: int | None = None) -> ProcessMetrics:
    """Считает метрики процесса обычным Python-кодом (без LLM).

    Возвращает число операций (всего/ручных/автоматизированных), долю ручных, суммарное время одного
    экземпляра процесса, время ручных операций и условную месячную трудоёмкость.
    Неуказанные длительности не заменяются нулями и не выдумываются: они перечисляются в
    steps_without_duration, а time_is_complete = false. Месячные показатели считаются только если
    задано runs_per_month (число запусков процесса в месяц).
    """
    try:
        spec, _ = process_tools.load_valid_spec(process)
        return process_tools.compute_metrics(spec, runs_per_month)
    except process_tools.ProcessToolError as exc:
        raise ToolError(str(exc)) from exc


@mcp_server.tool()
def simulate_automation(
    process: dict[str, Any],
    automation: list[AutomationChange],
    runs_per_month: int | None = None,
) -> SimulationResult:
    """Модельная оценка эффекта автоматизации выбранных операций при заданных предположениях.

    Для каждой автоматизируемой операции вызывающий ОБЯЗАН указать предполагаемую длительность после
    автоматизации (new_duration_minutes): никакой процент сокращения не подставляется автоматически.
    Возвращает исходное и предлагаемое время процесса, абсолютное и процентное сокращение, исходную и
    предлагаемую долю ручных операций и условную экономию за месяц (если задано runs_per_month).
    Результат помечен is_model_estimate = true и содержит перечень принятых предположений.
    """
    try:
        spec, _ = process_tools.load_valid_spec(process)
        return process_tools.simulate(spec, automation, runs_per_month)
    except process_tools.ProcessToolError as exc:
        raise ToolError(str(exc)) from exc


@mcp_server.tool()
def search_knowledge_base(query: str, n_results: int = 3) -> list[dict]:
    """Семантический поиск применимых паттернов автоматизации (эмбеддинги OpenAI + ChromaDB).

    Возвращает паттерны с pattern_id, title, source, score и подтверждающими фрагментами.
    Требует построенной базы знаний и OPENAI_API_KEY.
    """
    from processmind.rag.errors import KnowledgeBaseError
    from processmind.rag.knowledge_base import search_automation_patterns  # тяжёлый импорт (ChromaDB) — только по требованию

    try:
        return [m.model_dump() for m in search_automation_patterns(query, top_k=n_results)]
    except KnowledgeBaseError as exc:
        raise ToolError(str(exc)) from exc


@mcp_server.tool()
def get_process_spec_schema() -> dict:
    """Возвращает JSON-схему единой модели процесса ProcessSpec."""
    return ProcessSpec.model_json_schema()


if __name__ == "__main__":
    mcp_server.run(transport="stdio")
