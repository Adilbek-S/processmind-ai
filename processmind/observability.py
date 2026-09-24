"""Наблюдаемость: конфигурация LangSmith и декораторы трассировки.

LangSmith настраивается только переменными окружения (см. `.env.example`):

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=<ключ>
    LANGSMITH_PROJECT=processmind-ai        # необязательно
    LANGSMITH_ENDPOINT=<url>                # необязательно (EU-регион или self-hosted)

Прежние имена LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT тоже работают.
Если трейсинг выключен или нет ключа, все декораторы — no-op: приложение работает как обычно, а
`tracing_status()` объясняет, почему трейсов нет. Никакого «имитированного» трейсинга не существует.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Callable

import warnings

from langsmith import traceable

from processmind.config import Settings, get_settings

SETUP_INSTRUCTIONS = """\
LangSmith не настроен. Чтобы включить трассировку:
  1. Зарегистрируйтесь на https://smith.langchain.com и создайте API-ключ (Settings -> API Keys).
  2. Добавьте в файл .env (см. .env.example):
       LANGSMITH_TRACING=true
       LANGSMITH_API_KEY=<ваш ключ>
       LANGSMITH_PROJECT=processmind-ai
     (для EU-региона добавьте LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com)
  3. Перезапустите приложение / команду и выполните анализ, затем откройте проект в LangSmith.
Проверка: python -m evals.verify_tracing"""


@dataclass(frozen=True)
class TracingStatus:
    enabled: bool
    project: str
    endpoint: str | None
    reason: str | None  # почему трейсинг выключен (None, если включён)


_configured: TracingStatus | None = None

# Безобидные предупреждения: устаревший внутренний импорт langsmith и сериализация parsed-объекта обёрткой OpenAI.
warnings.filterwarnings("ignore", message="langsmith.wrappers._openai_agents is deprecated", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)


def resolve_tracing(settings: Settings | None = None) -> TracingStatus:
    """Определяет, включена ли трассировка, по настройкам (без побочных эффектов)."""
    s = settings or get_settings()
    flag = s.langsmith_tracing if s.langsmith_tracing is not None else s.langchain_tracing_v2
    key = s.langsmith_api_key or s.langchain_api_key
    project = s.langsmith_project or s.langchain_project
    if not flag:
        return TracingStatus(False, project, s.langsmith_endpoint, "трассировка выключена (LANGSMITH_TRACING не равно true)")
    if not key:
        return TracingStatus(False, project, s.langsmith_endpoint, "не задан LANGSMITH_API_KEY")
    return TracingStatus(True, project, s.langsmith_endpoint, None)


def _reset_langsmith_env_cache() -> None:
    """LangSmith кэширует чтение переменных окружения; после их изменения кэш нужно сбросить."""
    try:
        from langsmith import utils

        utils.get_env_var.cache_clear()
        utils.get_tracer_project.cache_clear() if hasattr(utils, "get_tracer_project") else None
    except Exception:  # noqa: BLE001 - отсутствие кэша в другой версии SDK не критично
        pass


def _reset_tracing_clients() -> None:
    """Сбрасывает закэшированные клиенты LangSmith (нужно, если изменились ключ или endpoint)."""
    try:
        import langchain_core.tracers.langchain as lc_tracer

        lc_tracer._CLIENT = None
    except Exception:  # noqa: BLE001
        pass
    try:
        from langsmith import run_trees

        run_trees._CLIENT = None  # get_cached_client() хранит клиента в модульной переменной, а не в lru_cache
    except Exception:  # noqa: BLE001
        pass


def flush_tracing() -> None:
    """Дожидается отправки накопленных трейсов (SDK отправляет их фоновым потоком; перед выходом процесса нужен flush)."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:  # noqa: BLE001
        pass
    try:
        from langsmith import run_trees

        run_trees.get_cached_client().flush()
    except Exception:  # noqa: BLE001
        pass


def configure_tracing(settings: Settings | None = None, *, force: bool = False) -> TracingStatus:
    """Переносит настройки из .env в переменные окружения, которые читает LangSmith SDK. Идемпотентно."""
    global _configured
    if _configured is not None and not force and settings is None:
        return _configured
    s = settings or get_settings()
    status = resolve_tracing(s)
    key = s.langsmith_api_key or s.langchain_api_key
    flag = "true" if status.enabled else "false"
    os.environ["LANGSMITH_TRACING"] = flag
    os.environ["LANGCHAIN_TRACING_V2"] = flag
    if status.enabled:
        os.environ["LANGSMITH_API_KEY"] = key or ""
        os.environ["LANGSMITH_PROJECT"] = status.project
        os.environ["LANGCHAIN_PROJECT"] = status.project
        if status.endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = status.endpoint
    _reset_langsmith_env_cache()
    if force:
        _reset_tracing_clients()
    _configured = status
    return status


def tracing_status() -> TracingStatus:
    return configure_tracing()


def traced(name: str | None = None, run_type: str = "chain", **kwargs: Any) -> Callable:
    """`langsmith.traceable`, который предварительно применяет конфигурацию из .env. Без трассировки — no-op."""

    def decorator(fn: Callable) -> Callable:
        wrapped = traceable(name=name or fn.__name__, run_type=run_type, **kwargs)(fn)

        @functools.wraps(fn)
        def inner(*args: Any, **kw: Any) -> Any:
            configure_tracing()
            return wrapped(*args, **kw)

        return inner

    return decorator
