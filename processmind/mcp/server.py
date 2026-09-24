"""Собственный MCP-сервер ProcessMind, реализованный через MCPServer (Python MCP SDK).

Экспортирует инструменты, которые внешние MCP-клиенты (или другие агенты) могут
использовать для работы с ProcessMind: поиск по базе знаний (RAG) и получение
JSON-схемы модели ProcessSpec. Реальная LLM-логика анализа процесса пока не
подключена — это осознанное ограничение MVP.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from processmind.config import get_settings
from processmind.models import ProcessSpec
from processmind.rag.engine import RAGEngine

settings = get_settings()
mcp_server = MCPServer(settings.mcp_server_name)


@mcp_server.tool()
def search_knowledge_base(query: str, n_results: int = 3) -> list[str]:
    """Семантический поиск по базе знаний ProcessMind (ChromaDB)."""
    engine = RAGEngine()
    return engine.query(query, n_results=n_results)


@mcp_server.tool()
def get_process_spec_schema() -> dict:
    """Возвращает JSON-схему единой модели процесса ProcessSpec."""
    return ProcessSpec.model_json_schema()


if __name__ == "__main__":
    mcp_server.run()
