"""Локальный двойник LangSmith API: принимает `POST /runs/multipart` от настоящего SDK и собирает runs.

Проверяет НАШУ инструментацию (какие runs, какого типа, с какими родителями, токенами и ошибками отправляются).
Это не подтверждение работы облачного LangSmith: реальные трейсы проверяет `python -m evals.verify_tracing`
(нужен LANGSMITH_API_KEY).
"""

from __future__ import annotations

import json
import threading
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from processmind.config import Settings
from processmind.observability import configure_tracing, flush_tracing


class LangSmithDouble:
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        double = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, body: bytes = b"{}") -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - /info и т.п.
                self._reply()

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path.endswith("/runs/multipart"):
                    double._ingest(self.headers.get("Content-Type", ""), body)
                self._reply()

            do_PATCH = do_POST  # noqa: N815

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def _ingest(self, content_type: str, body: bytes) -> None:
        message = BytesParser().parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
        for part in message.get_payload():
            name = part.get_param("name", header="content-disposition") or ""
            payload = part.get_payload(decode=True)
            if not name or not payload or not part.get_content_type().endswith("json"):
                continue
            kind, _, rest = name.partition(".")  # post.<id>[.<field>] | patch.<id>[.<field>]
            run_id, _, field = rest.partition(".")
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            with self._lock:
                run = self._runs.setdefault(run_id, {"id": run_id})
                if field:
                    run[field] = data
                elif isinstance(data, dict):
                    run.update({k: v for k, v in data.items() if v is not None})

    def start_tracing(self, project: str = "processmind-test") -> None:
        """Направляет трассировку SDK на этот двойник (через те же переменные окружения, что и в приложении)."""
        settings = Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="ls-test-key", langsmith_project=project, langsmith_endpoint=self.url)
        configure_tracing(settings, force=True)

    def runs(self) -> list[dict[str, Any]]:
        flush_tracing()
        with self._lock:
            return [dict(r) for r in self._runs.values()]

    def by_name(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.runs() if r.get("name") == name]

    def names(self) -> set[str]:
        return {r["name"] for r in self.runs() if r.get("name")}

    def ancestors(self, run: dict[str, Any]) -> list[str]:
        """Имена предков run (от родителя к корню) — для проверки вложенности."""
        by_id = {r["id"]: r for r in self.runs()}
        names, parent = [], run.get("parent_run_id")
        while parent and parent in by_id:
            names.append(by_id[parent].get("name", "?"))
            parent = by_id[parent].get("parent_run_id")
        return names

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
