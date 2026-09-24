"""RAG Engine: хранение и семантический поиск документов базы знаний в ChromaDB.

Если задан OPENAI_API_KEY, используются OpenAI Embeddings (text-embedding-3-small).
Если ключ не задан, используется детерминированный офлайн-fallback на основе
хэширования токенов — это позволяет демонстрировать работу базы знаний без
реальных вызовов API (все демо-данные синтетические).
"""

from __future__ import annotations

import math
import uuid

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

from processmind.config import get_settings

DEFAULT_COLLECTION_NAME = "processmind_kb"
FALLBACK_EMBEDDING_DIM = 384


class SimpleHashEmbeddingFunction(EmbeddingFunction):
    """Детерминированный офлайн-эмбеддинг для демо/тестов без API-ключа."""

    def __init__(self, dim: int = FALLBACK_EMBEDDING_DIM) -> None:
        self._dim = dim

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - имя параметра задано chromadb
        return [self._embed(text) for text in input]

    @staticmethod
    def name() -> str:
        return "processmind-simple-hash-embedding"

    def get_config(self) -> dict:
        return {"dim": self._dim}

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in text.lower().split():
            idx = hash(token) % self._dim
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class RAGEngine:
    """Обёртка над ChromaDB для добавления и поиска документов базы знаний."""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME, persist_dir: str | None = None) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=persist_dir or settings.chroma_persist_dir)

        if settings.openai_api_key:
            embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name=settings.embedding_model,
            )
        else:
            embed_fn = SimpleHashEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )

    def add_documents(self, documents: list[str], ids: list[str] | None = None, metadatas: list[dict] | None = None) -> list[str]:
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)
        return ids

    def query(self, query_text: str, n_results: int = 3) -> list[str]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(query_texts=[query_text], n_results=min(n_results, self._collection.count()))
        return results.get("documents", [[]])[0]
