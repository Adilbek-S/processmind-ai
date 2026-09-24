"""Централизованная конфигурация приложения.

Все настройки загружаются из переменных окружения (или файла .env).
Секреты никогда не хранятся в коде — см. .env.example для списка ожидаемых переменных.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI / LLM
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # Automation Knowledge Base: каталог Markdown-паттернов (None — каталог по умолчанию в репозитории)
    knowledge_base_dir: str | None = None

    # LangSmith / LangChain tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "processmind-ai"

    # MCP
    mcp_server_name: str = "processmind-mcp"


@lru_cache
def get_settings() -> Settings:
    return Settings()
