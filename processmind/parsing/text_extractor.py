"""Сценарий А: извлечение процесса из текста PDF через LLM."""

from __future__ import annotations

from openai import OpenAI

from processmind.parsing.llm import request_raw_process
from processmind.parsing.prompts import TEXT_SYSTEM_PROMPT, TEXT_USER_PROMPT_PREFIX
from processmind.parsing.schemas import ExtractionError, RawProcess

# gpt-4o-mini: контекст 128k токенов. Молча обрезать документ нельзя — потеряются операции.
MAX_TEXT_CHARS = 100_000


class TextProcessExtractor:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def extract(self, text: str) -> RawProcess:
        if len(text) > MAX_TEXT_CHARS:
            raise ExtractionError(
                f"Документ слишком большой ({len(text)} символов, максимум {MAX_TEXT_CHARS}). "
                "Загрузите только раздел с описанием процесса."
            )
        messages = [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": TEXT_USER_PROMPT_PREFIX + text},
        ]
        return request_raw_process(self._client, self._model, messages)
