"""OpenAIEmbedder и весь RAG через НАСТОЯЩИЙ OpenAI SDK по HTTP на локальный сервер-двойник.

Проверяется формат запросов к /v1/embeddings (модель, батчи, порядок), обработка ошибок и работа
build/search с эмбеддером из настроек. Векторы здесь заготовленные (лексические), т.е. это тест
обвязки, а не качества эмбеддингов text-embedding-3-small.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from openai import OpenAI

from processmind.config import Settings
from processmind.rag.embeddings import OpenAIEmbedder, get_embedder
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import DEFAULT_KB_DIR, build_knowledge_base, search_automation_patterns
from tests.helpers_kb import fake_vector


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    status = 200
    reverse = False
    drop_last = False

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        cls.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if cls.status != 200:
            payload = json.dumps({"error": {"message": "nope", "type": "invalid_request_error"}}).encode()
        else:
            inputs = body["input"]
            data = [{"object": "embedding", "index": i, "embedding": fake_vector(t)} for i, t in enumerate(inputs)]
            if cls.reverse:
                data.reverse()  # SDK-клиент обязан восстановить порядок по index
            if cls.drop_last:
                data = data[:-1]
            payload = json.dumps(
                {"object": "list", "data": data, "model": body["model"], "usage": {"prompt_tokens": 1, "total_tokens": 1}}
            ).encode()
        self.send_response(cls.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    _Handler.requests, _Handler.status, _Handler.reverse, _Handler.drop_last = [], 200, False, False
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{httpd.server_port}/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    _Handler.base_url = base_url
    yield _Handler
    httpd.shutdown()


def embedder(batch_size: int = 64) -> OpenAIEmbedder:
    return OpenAIEmbedder(OpenAI(api_key="sk-test", max_retries=0), "text-embedding-3-small", batch_size=batch_size)


def test_request_format_model_and_auth(server):
    vectors = embedder().embed(["один", "два"])

    (req,) = server.requests
    assert req["path"].endswith("/embeddings") and req["auth"] == "Bearer sk-test"
    assert req["body"]["model"] == "text-embedding-3-small"
    assert req["body"]["input"] == ["один", "два"]
    assert vectors == [fake_vector("один"), fake_vector("два")]


def test_texts_are_sent_in_batches(server):
    texts = [f"текст {i}" for i in range(5)]
    vectors = embedder(batch_size=2).embed(texts)
    assert [len(r["body"]["input"]) for r in server.requests] == [2, 2, 1]
    assert vectors == [fake_vector(t) for t in texts]


def test_response_order_restored_by_index(server):
    server.reverse = True
    assert embedder().embed(["альфа", "бета"]) == [fake_vector("альфа"), fake_vector("бета")]


def test_count_mismatch_is_an_error(server):
    server.drop_last = True
    with pytest.raises(KnowledgeBaseError, match="эмбеддингов"):
        embedder().embed(["а", "б"])


def test_api_error_is_wrapped(server):
    server.status = 401
    with pytest.raises(KnowledgeBaseError, match="ключ"):
        embedder().embed(["а"])


def test_get_embedder_requires_key_and_uses_configured_model():
    with pytest.raises(KnowledgeBaseError, match="OPENAI_API_KEY"):
        get_embedder(Settings(_env_file=None, openai_api_key=None))
    assert get_embedder(Settings(_env_file=None, openai_api_key="sk-x")).model == "text-embedding-3-small"


def test_full_pipeline_over_http_with_settings_embedder(server, tmp_path):
    settings = Settings(_env_file=None, openai_api_key="sk-test", chroma_persist_dir=str(tmp_path / "chroma"))

    report = build_knowledge_base(settings=settings)
    assert report.total_patterns >= 12 and report.embedded_chunks == report.total_chunks
    assert {m for r in server.requests for m in [r["body"]["model"]]} == {"text-embedding-3-small"}
    embedded_inputs = sum(len(r["body"]["input"]) for r in server.requests)
    assert embedded_inputs == report.total_chunks

    again = build_knowledge_base(settings=settings)  # повторная индексация: ни одного нового запроса
    assert again.embedded_chunks == 0 and sum(len(r["body"]["input"]) for r in server.requests) == embedded_inputs

    matches = search_automation_patterns("Распознавание текста на сканах документов", settings=settings)
    assert len(matches) == 3 and server.requests[-1]["body"]["input"] == ["Распознавание текста на сканах документов"]
    assert DEFAULT_KB_DIR.is_dir()
