"""Прогон через НАСТОЯЩИЙ OpenAI SDK по HTTP на локальный сервер-двойник.

Проверяет то, что подмена клиента не покрывает: реальный формат запроса (Structured Outputs,
image_url) и разбор ответа официальным SDK. Ответ модели здесь заготовлен — это тест обвязки,
а не качества распознавания.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from processmind.config import Settings
from processmind.parsing.service import extract_process
from tests.test_extraction_service import demo_pdf, demo_png, perfect_raw_process  # noqa: F401 (фикстуры)


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    reply: dict = {}

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        payload = json.dumps(type(self).reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # тишина в выводе pytest
        pass


@pytest.fixture
def fake_openai(monkeypatch):
    _Handler.requests = []
    _Handler.reply = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": perfect_raw_process().model_dump_json(), "refusal": None},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    yield _Handler
    server.shutdown()


SETTINGS = Settings(_env_file=None, openai_api_key="sk-test", openai_model="gpt-4o-mini")


def test_image_request_uses_structured_outputs_and_image_url(fake_openai, demo_png):  # noqa: F811
    result = extract_process("scheme.png", demo_png, settings=SETTINGS)

    (req,) = fake_openai.requests
    assert req["path"].endswith("/chat/completions")
    assert req["auth"] == "Bearer sk-test"
    body = req["body"]
    assert body["model"] == "gpt-4o-mini"
    schema = body["response_format"]
    assert schema["type"] == "json_schema" and schema["json_schema"]["strict"] is True
    content = body["messages"][1]["content"]
    image = next(p for p in content if p["type"] == "image_url")["image_url"]
    assert image["url"].startswith("data:image/png;base64,") and image["detail"] == "high"
    assert len(result.spec.steps) == 8


def test_pdf_request_carries_extracted_text(fake_openai, demo_pdf):  # noqa: F811
    result = extract_process("demo.pdf", demo_pdf, settings=SETTINGS)

    (req,) = fake_openai.requests
    assert "Согласование заявки на закупку оборудования" in req["body"]["messages"][1]["content"]
    assert len(result.spec.steps) == 8
