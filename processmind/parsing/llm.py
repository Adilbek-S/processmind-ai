"""Общий вызов OpenAI (официальный SDK) со структурированным выводом.

Используется и извлечением процесса (RawProcess), и генерацией рекомендаций: клиент, обработка
ошибок API и разбор отказов/невалидных ответов живут в одном месте.
"""

from __future__ import annotations

from typing import TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel

from processmind.config import Settings, get_settings
from processmind.parsing.schemas import ExtractionError, RawProcess

T = TypeVar("T", bound=BaseModel)


def get_client(settings: Settings | None = None, error_cls: type[Exception] = ExtractionError) -> OpenAI:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise error_cls(
            "OPENAI_API_KEY не задан: для этой операции нужен вызов OpenAI. "
            "Укажите ключ в файле .env и перезапустите приложение."
        )
    return OpenAI(api_key=settings.openai_api_key)


def request_structured(
    client: OpenAI,
    model: str,
    messages: list[dict],
    response_format: type[T],
    error_cls: type[Exception] = ExtractionError,
) -> T:
    """Один запрос к модели со структурированным выводом; любой сбой превращается в error_cls."""
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=0,
        )
    except openai.AuthenticationError as exc:
        raise error_cls("OpenAI отклонил ключ API (AuthenticationError). Проверьте OPENAI_API_KEY.") from exc
    except openai.RateLimitError as exc:
        raise error_cls("Превышен лимит запросов или квота OpenAI (RateLimitError).") from exc
    except openai.APIConnectionError as exc:
        raise error_cls("Не удалось подключиться к OpenAI. Проверьте сетевое соединение.") from exc
    except openai.OpenAIError as exc:
        raise error_cls(f"Ошибка OpenAI: {exc}") from exc

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise error_cls(f"Модель отказалась обрабатывать запрос: {message.refusal}")
    if message.parsed is None:
        raise error_cls("Модель вернула ответ, не соответствующий схеме. Повторите попытку.")
    return message.parsed


def request_raw_process(client: OpenAI, model: str, messages: list[dict]) -> RawProcess:
    return request_structured(client, model, messages, RawProcess, ExtractionError)
