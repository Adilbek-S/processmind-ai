"""Общий вызов OpenAI (официальный SDK) со структурированным выводом RawProcess."""

from __future__ import annotations

import openai
from openai import OpenAI

from processmind.config import Settings, get_settings
from processmind.parsing.schemas import ExtractionError, RawProcess


def get_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise ExtractionError(
            "OPENAI_API_KEY не задан: распознавание процесса требует вызова OpenAI. "
            "Укажите ключ в файле .env и перезапустите приложение."
        )
    return OpenAI(api_key=settings.openai_api_key)


def request_raw_process(client: OpenAI, model: str, messages: list[dict]) -> RawProcess:
    """Один запрос к модели; любой сбой превращается в ExtractionError с понятным текстом."""
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=RawProcess,
            temperature=0,
        )
    except openai.AuthenticationError as exc:
        raise ExtractionError("OpenAI отклонил ключ API (AuthenticationError). Проверьте OPENAI_API_KEY.") from exc
    except openai.RateLimitError as exc:
        raise ExtractionError("Превышен лимит запросов или квота OpenAI (RateLimitError).") from exc
    except openai.APIConnectionError as exc:
        raise ExtractionError("Не удалось подключиться к OpenAI. Проверьте сетевое соединение.") from exc
    except openai.OpenAIError as exc:
        raise ExtractionError(f"Ошибка OpenAI: {exc}") from exc

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise ExtractionError(f"Модель отказалась обрабатывать документ: {message.refusal}")
    if message.parsed is None:
        raise ExtractionError("Модель вернула ответ, не соответствующий схеме ProcessSpec. Повторите попытку.")
    return message.parsed
