"""Эмбеддинги через OpenAI (text-embedding-3-small).

Намеренно без офлайн-подстановки: если ключа нет или API недоступно, индексация и поиск
завершаются понятной ошибкой. Хэширование слов вместо эмбеддингов — это поиск по ключевым
словам под видом семантического.
"""

from __future__ import annotations

from typing import Protocol

import openai
from openai import OpenAI

from processmind.config import Settings, get_settings
from processmind.rag.errors import KnowledgeBaseError

EMBED_BATCH_SIZE = 64


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str, batch_size: int = EMBED_BATCH_SIZE) -> None:
        self._client = client
        self.model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
            except openai.AuthenticationError as exc:
                raise KnowledgeBaseError("OpenAI отклонил ключ API (AuthenticationError). Проверьте OPENAI_API_KEY.") from exc
            except openai.RateLimitError as exc:
                raise KnowledgeBaseError("Превышен лимит запросов или квота OpenAI (RateLimitError).") from exc
            except openai.APIConnectionError as exc:
                raise KnowledgeBaseError("Не удалось подключиться к OpenAI. Проверьте сетевое соединение.") from exc
            except openai.OpenAIError as exc:
                raise KnowledgeBaseError(f"Ошибка OpenAI при получении эмбеддингов: {exc}") from exc

            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise KnowledgeBaseError(
                    f"OpenAI вернул {len(data)} эмбеддингов на {len(batch)} текстов — индекс не обновлён."
                )
            vectors.extend(item.embedding for item in data)
        return vectors


def get_embedder(settings: Settings | None = None) -> OpenAIEmbedder:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise KnowledgeBaseError(
            "OPENAI_API_KEY не задан: для векторного поиска нужны эмбеддинги OpenAI. "
            "Укажите ключ в файле .env и перезапустите приложение."
        )
    return OpenAIEmbedder(OpenAI(api_key=settings.openai_api_key), settings.embedding_model)
