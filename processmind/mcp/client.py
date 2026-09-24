"""MCP-клиент ProcessMind: вызов инструментов сервера отдельным процессом по stdio.

Стандартный клиент Python MCP SDK (`stdio_client` + `ClientSession`): при входе в контекст клиент
запускает `python -m processmind.mcp.server` подпроцессом, выполняет initialize и общается по
JSON-RPC через stdin/stdout. Инструменты НЕ вызываются как локальные функции.

Использование в LangGraph:

    # асинхронный узел (graph.ainvoke)
    async def metrics_node(state):
        async with MCPClient() as mcp:
            return {"metrics": await mcp.calculate_process_metrics(state["process_spec"], runs_per_month=200)}

    # синхронный узел (graph.invoke): одна сессия и один процесс сервера на все вызовы
    mcp = SyncMCPClient().start()          # один раз при сборке графа
    def metrics_node(state):
        return {"metrics": mcp.calculate_process_metrics(state["process_spec"], runs_per_month=200)}
    ...
    mcp.close()                            # при завершении приложения
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, TextIO

from mcp import ClientSession, MCPError, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from processmind.observability import traced

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_SECONDS = 30.0
CLEANUP_ALLOWANCE_SECONDS = 15.0


class McpClientError(Exception):
    """Не удалось запустить сервер, установить сессию или дождаться ответа."""


class McpToolError(Exception):
    """Инструмент вернул ошибку (isError) либо запрос отклонён на уровне протокола."""


def default_server_parameters(*, env: dict[str, str] | None = None) -> StdioServerParameters:
    """Запуск собственного сервера тем же интерпретатором; cwd = корень проекта (для .env и импортов)."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "processmind.mcp.server"],
        cwd=PROJECT_ROOT,
        env=env,
    )


def _to_payload(process: BaseModel | dict[str, Any]) -> dict[str, Any]:
    return process.model_dump(mode="json") if isinstance(process, BaseModel) else process


class MCPClient:
    """Асинхронный клиент: `async with MCPClient() as mcp: await mcp.call_tool(...)`."""

    def __init__(
        self,
        server: StdioServerParameters | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        errlog: TextIO | None = None,
    ) -> None:
        self._server = server or default_server_parameters()
        self._timeout = timeout
        self._errlog = errlog
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._devnull: TextIO | None = None
        self.server_name: str | None = None

    async def __aenter__(self) -> MCPClient:
        self._stack = AsyncExitStack()
        try:
            errlog = self._errlog or self._default_errlog()
            read, write = await self._stack.enter_async_context(stdio_client(self._server, errlog=errlog))
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=self._timeout)
            )
            init = await asyncio.wait_for(self._session.initialize(), timeout=self._timeout)
            self.server_name = init.server_info.name
        except BaseException as exc:
            await self._cleanup()
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                raise
            raise McpClientError(f"Не удалось запустить MCP-сервер или установить сессию: {exc!r}") from exc
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._cleanup()

    async def _cleanup(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        try:
            if stack is not None:
                await stack.aclose()
        finally:
            if self._devnull is not None:
                self._devnull.close()
                self._devnull = None

    def _default_errlog(self) -> TextIO:
        """stderr сервера: в консоль, а если у stderr нет файлового дескриптора (pytest) — в никуда."""
        try:
            sys.stderr.fileno()
            return sys.stderr
        except (AttributeError, OSError, ValueError):
            self._devnull = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - закрывается в _cleanup
            return self._devnull

    # ---------- вызовы ----------

    async def list_tools(self) -> list[str]:
        return [tool.name for tool in (await self._require_session().list_tools()).tools]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Вызывает инструмент сервера и возвращает structuredContent (словарь)."""
        session = self._require_session()
        try:
            result = await session.call_tool(name, arguments or {})
        except MCPError as exc:
            raise McpToolError(f"MCP-сервер отклонил вызов «{name}»: {exc}") from exc
        except TimeoutError as exc:
            raise McpClientError(f"MCP-сервер не ответил на «{name}» за {self._timeout:g} с") from exc

        text = " ".join(getattr(block, "text", "") for block in result.content).strip()
        if result.is_error:
            text = re.sub(r"^Error executing tool \S+: ", "", text)  # служебный префикс SDK не нужен вызывающему
            raise McpToolError(text or f"Инструмент «{name}» вернул ошибку")
        if result.structured_content is not None:
            return result.structured_content
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise McpToolError(f"Инструмент «{name}» вернул ответ без структурированных данных") from exc

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise McpClientError("MCP-клиент не запущен (используйте `async with MCPClient()`)")
        return self._session

    # ---------- типизированные обёртки ----------

    async def validate_process(self, process: BaseModel | dict[str, Any]) -> dict[str, Any]:
        return await self.call_tool("validate_process", {"process": _to_payload(process)})

    async def calculate_process_metrics(
        self, process: BaseModel | dict[str, Any], runs_per_month: int | None = None
    ) -> dict[str, Any]:
        return await self.call_tool(
            "calculate_process_metrics", {"process": _to_payload(process), "runs_per_month": runs_per_month}
        )

    async def simulate_automation(
        self,
        process: BaseModel | dict[str, Any],
        automation: list[dict[str, Any]],
        runs_per_month: int | None = None,
    ) -> dict[str, Any]:
        """`automation`: [{"step_id": "...", "new_duration_minutes": 5}, ...] — длительность после автоматизации."""
        return await self.call_tool(
            "simulate_automation",
            {"process": _to_payload(process), "automation": automation, "runs_per_month": runs_per_month},
        )


class SyncMCPClient:
    """Синхронный фасад для обычных (sync) узлов LangGraph.

    Держит одну сессию и один процесс сервера: сессия живёт в отдельном потоке со своим event loop,
    поэтому её можно вызывать из любого синхронного кода (в том числе внутри работающего loop'а).
    """

    def __init__(self, server: StdioServerParameters | None = None, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 errlog: TextIO | None = None) -> None:
        self._client = MCPClient(server, timeout=timeout, errlog=errlog)
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._start_error: BaseException | None = None

    def start(self) -> SyncMCPClient:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._thread_main, name="mcp-client", daemon=True)
        self._thread.start()
        # таймаут инициализации + время на завершение зависшего процесса сервера (см. stdio_client)
        if not self._ready.wait(timeout=self._timeout + CLEANUP_ALLOWANCE_SECONDS):
            self.close()
            raise McpClientError("MCP-сервер не запустился вовремя")
        if self._start_error is not None:
            error, self._start_error = self._start_error, None
            self._thread.join(timeout=5)
            self._thread = None
            raise error if isinstance(error, McpClientError) else McpClientError(str(error))
        return self

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            async with self._client:  # вход и выход из контекста — в одной задаче (требование anyio)
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:
            self._start_error = exc
        finally:
            self._ready.set()

    def close(self) -> None:
        thread, self._thread = self._thread, None
        if thread is None:
            return
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        thread.join(timeout=10)

    def __enter__(self) -> SyncMCPClient:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _run_coro(self, coro):
        if self._thread is None or self._loop is None:
            coro.close()
            raise McpClientError("MCP-клиент не запущен (вызовите start() или используйте `with`)")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=self._timeout + 5)
        except TimeoutError as exc:
            future.cancel()
            raise McpClientError("MCP-сервер не ответил вовремя") from exc

    def list_tools(self) -> list[str]:
        return self._run_coro(self._client.list_tools())

    def _traced_tool(self, name: str, arguments: Any, call) -> dict[str, Any]:
        """Вызов инструмента как run типа tool: в трейсе видны инструмент, аргументы, результат, длительность и ошибка.

        Трассируется в потоке вызывающего (а не в фоновом потоке клиента), поэтому вызов вкладывается в трейс узла графа.
        """

        @traced(
            name=f"mcp.{name}",
            run_type="tool",
            metadata={"mcp_server": self.server_name, "transport": "stdio"},
            process_inputs=lambda _: {"tool": name, "arguments": arguments},
        )
        def run() -> dict[str, Any]:
            return call()

        return run()

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._traced_tool(name, arguments, lambda: self._run_coro(self._client.call_tool(name, arguments)))

    def validate_process(self, process: BaseModel | dict[str, Any]) -> dict[str, Any]:
        return self._traced_tool("validate_process", {"process": _to_payload(process)}, lambda: self._run_coro(self._client.validate_process(process)))

    def calculate_process_metrics(self, process: BaseModel | dict[str, Any], runs_per_month: int | None = None) -> dict[str, Any]:
        args = {"process": _to_payload(process), "runs_per_month": runs_per_month}
        return self._traced_tool("calculate_process_metrics", args, lambda: self._run_coro(self._client.calculate_process_metrics(process, runs_per_month)))

    def simulate_automation(
        self, process: BaseModel | dict[str, Any], automation: list[dict[str, Any]], runs_per_month: int | None = None
    ) -> dict[str, Any]:
        args = {"process": _to_payload(process), "automation": automation, "runs_per_month": runs_per_month}
        return self._traced_tool("simulate_automation", args, lambda: self._run_coro(self._client.simulate_automation(process, automation, runs_per_month)))

    @property
    def server_name(self) -> str | None:
        return self._client.server_name
