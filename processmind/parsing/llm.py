"""Общий вызов OpenAI (официальный SDK) со структурированным выводом.

Используется и извлечением процесса (RawProcess), и генерацией рекомендаций: клиент, параметры генерации,
обработка ошибок API, учёт токенов/задержки и трассировка LangSmith живут в одном месте.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel

from processmind.config import Settings, get_settings
from processmind.observability import configure_tracing
from processmind.parsing.schemas import ExtractionError, RawProcess

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMParams:
    """Параметры генерации. Значения по умолчанию берутся из настроек (выбраны по EVALS.md)."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    seed: int | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LLMParams:
        s = settings or get_settings()
        return cls(temperature=s.llm_temperature, top_p=s.llm_top_p, max_tokens=s.llm_max_tokens, seed=s.llm_seed)


@dataclass(frozen=True)
class LLMUsage:
    """Фактические затраты одного вызова: токены из ответа API и измеренная задержка."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    finish_reason: str | None
    model: str | None
    system_fingerprint: str | None


def get_client(settings: Settings | None = None, error_cls: type[Exception] = ExtractionError) -> OpenAI:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise error_cls(
            "OPENAI_API_KEY не задан: для этой операции нужен вызов OpenAI. "
            "Укажите ключ в файле .env и перезапустите приложение."
        )
    client = OpenAI(api_key=settings.openai_api_key)
    configure_tracing()
    try:  # LLM-вызовы попадают в LangSmith как runs типа llm с токенами (если трассировка включена)
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception:  # noqa: BLE001 - несовместимая версия обёртки не должна ломать приложение
        return client


def request_structured_ex(
    client: OpenAI,
    model: str,
    messages: list[dict],
    response_format: type[T],
    error_cls: type[Exception] = ExtractionError,
    params: LLMParams | None = None,
) -> tuple[T, LLMUsage]:
    """Один запрос со структурированным выводом; возвращает результат и фактическое использование токенов."""
    params = params or LLMParams.from_settings()
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_completion_tokens": params.max_tokens,
    }
    if params.seed is not None:
        kwargs["seed"] = params.seed

    started = time.perf_counter()
    try:
        completion = client.chat.completions.parse(**kwargs)
    except openai.AuthenticationError as exc:
        raise error_cls("OpenAI отклонил ключ API (AuthenticationError). Проверьте OPENAI_API_KEY.") from exc
    except openai.RateLimitError as exc:
        raise error_cls("Превышен лимит запросов или квота OpenAI (RateLimitError).") from exc
    except openai.APIConnectionError as exc:
        raise error_cls("Не удалось подключиться к OpenAI. Проверьте сетевое соединение.") from exc
    except openai.OpenAIError as exc:
        raise error_cls(f"Ошибка OpenAI: {exc}") from exc
    latency = time.perf_counter() - started

    choice = completion.choices[0]
    message = choice.message
    if getattr(message, "refusal", None):
        raise error_cls(f"Модель отказалась обрабатывать запрос: {message.refusal}")
    if getattr(choice, "finish_reason", None) == "length":
        raise error_cls(f"Ответ модели обрезан по лимиту max_tokens={params.max_tokens}. Увеличьте LLM_MAX_TOKENS.")
    if message.parsed is None:
        raise error_cls("Модель вернула ответ, не соответствующий схеме. Повторите попытку.")

    usage = getattr(completion, "usage", None)
    return message.parsed, LLMUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        latency_s=latency,
        finish_reason=getattr(choice, "finish_reason", None),
        model=getattr(completion, "model", None),
        system_fingerprint=getattr(completion, "system_fingerprint", None),
    )


def request_structured(
    client: OpenAI,
    model: str,
    messages: list[dict],
    response_format: type[T],
    error_cls: type[Exception] = ExtractionError,
    params: LLMParams | None = None,
) -> T:
    return request_structured_ex(client, model, messages, response_format, error_cls, params)[0]


def request_raw_process(client: OpenAI, model: str, messages: list[dict]) -> RawProcess:
    return request_structured(client, model, messages, RawProcess, ExtractionError)
