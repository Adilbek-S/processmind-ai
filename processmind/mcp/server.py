"""Собственный MCP-сервер ProcessMind, реализованный через MCPServer (Python MCP SDK).

Экспортирует инструменты, которые внешние MCP-клиенты (или другие агенты) могут
использовать для работы с ProcessMind: поиск способов автоматизации в базе знаний (RAG) и
получение JSON-схемы модели ProcessSpec.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from processmind.config import get_settings
from processmind.models import ProcessSpec
from processmind.rag.knowledge_base import search_automation_patterns

settings = get_settings()
mcp_server = MCPServer(settings.mcp_server_name)


@mcp_server.tool()
def search_knowledge_base(query: str, n_results: int = 3) -> list[dict]:
    """Семантический поиск применимых паттернов автоматизации (эмбеддинги OpenAI + ChromaDB).

    Возвращает паттерны с pattern_id, title, source, score и подтверждающими фрагментами.
    Требует построенной базы знаний и OPENAI_API_KEY.
    """
    return [m.model_dump() for m in search_automation_patterns(query, top_k=n_results)]


@mcp_server.tool()
def get_process_spec_schema() -> dict:
    """Возвращает JSON-схему единой модели процесса ProcessSpec."""
    return ProcessSpec.model_json_schema()


if __name__ == "__main__":
    mcp_server.run()
