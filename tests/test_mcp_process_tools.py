"""Тесты MCP-инструментов через НАСТОЯЩИЙ MCP-клиент и отдельный процесс сервера (stdio).

Каждый вызов проходит путь: клиент -> JSON-RPC по stdin/stdout -> подпроцесс
`python -m processmind.mcp.server` -> инструмент -> ответ. Локально функции ядра здесь не вызываются.
"""

import asyncio
import sys

import pytest
from mcp import StdioServerParameters

from processmind.mcp import process_tools
from processmind.mcp.client import (
    MCPClient,
    McpClientError,
    McpToolError,
    SyncMCPClient,
    default_server_parameters,
)
from processmind.models import ProcessSpec, ProcessStep

TOOLS = {"validate_process", "calculate_process_metrics", "simulate_automation"}


def step(sid, minutes, manual, **extra):
    return {"id": sid, "name": f"Операция {sid}", "duration_minutes": minutes, "is_manual": manual, "actor": "Сотрудник", **extra}


# Сумма 300 мин; ручные 240 мин; 4 операции, из них 3 ручные (расчёт вручную — как в test_process_tools).
PROCESS = {
    "process_id": "p1",
    "name": "Процесс",
    "steps": [step("s1", 30, True, next_steps=["s2"]), step("s2", 120, True, next_steps=["s3"]),
              step("s3", 60, False, next_steps=["s4"]), step("s4", 90, True)],
}


@pytest.fixture(scope="module")
def mcp():
    with SyncMCPClient() as client:  # один подпроцесс сервера на весь модуль
        yield client


# ---------- транспорт и регистрация ----------


def test_server_is_a_separate_stdio_process_with_registered_tools(mcp):
    assert mcp.server_name == "processmind-mcp"
    assert TOOLS <= set(mcp.list_tools())


def test_tools_really_execute_in_the_server_process_not_in_the_test_process(mcp, monkeypatch):
    """Ломаем функции ядра В ЭТОМ процессе: если бы вызов был локальным, он бы упал."""

    def boom(*args, **kwargs):
        raise AssertionError("инструмент вызван локально, а не через MCP")

    for name in ("validate_process_data", "compute_metrics", "simulate", "load_valid_spec"):
        monkeypatch.setattr(process_tools, name, boom)

    assert mcp.validate_process(PROCESS)["valid"] is True
    assert mcp.calculate_process_metrics(PROCESS, 10)["total_steps"] == 4
    assert mcp.simulate_automation(PROCESS, [{"step_id": "s2", "new_duration_minutes": 10}])["is_model_estimate"] is True


# ---------- успешные вызовы ----------


def test_validate_process_success(mcp):
    report = mcp.validate_process(PROCESS)
    assert report["valid"] is True and report["errors"] == [] and report["steps_checked"] == 4


def test_calculate_process_metrics_success_and_math(mcp):
    m = mcp.calculate_process_metrics(PROCESS, runs_per_month=50)
    assert (m["total_steps"], m["manual_steps"], m["automated_steps"]) == (4, 3, 1)
    assert m["manual_share"] == pytest.approx(0.75)
    assert m["total_time_minutes"] == 300 and m["manual_time_minutes"] == 240
    assert m["monthly_effort_minutes"] == 15000 and m["monthly_effort_hours"] == pytest.approx(250.0)


def test_simulate_automation_success_and_math(mcp):
    r = mcp.simulate_automation(
        PROCESS,
        [{"step_id": "s2", "new_duration_minutes": 10}, {"step_id": "s4", "new_duration_minutes": 15}],
        runs_per_month=50,
    )
    assert r["baseline_time_minutes"] == 300 and r["proposed_time_minutes"] == 115
    assert r["absolute_reduction_minutes"] == 185 and r["reduction_percent"] == pytest.approx(61.6667, abs=1e-4)
    assert r["baseline_manual_share"] == pytest.approx(0.75) and r["proposed_manual_share"] == pytest.approx(0.25)
    assert r["monthly_saving_minutes"] == 9250
    assert r["is_model_estimate"] is True and "Модельная оценка" in r["disclaimer"]
    assert [(a["step_id"], a["new_duration_minutes"]) for a in r["assumptions"]] == [("s2", 10), ("s4", 15)]


def test_client_accepts_process_spec_objects(mcp):
    spec = ProcessSpec(
        process_id="p2",
        name="Из модели",
        steps=[ProcessStep(id="a", name="A", duration_minutes=15, is_manual=True), ProcessStep(id="b", name="B", duration_minutes=5, is_manual=False)],
    )
    m = mcp.calculate_process_metrics(spec, 4)
    assert m["total_time_minutes"] == 20 and m["monthly_effort_minutes"] == 80


def test_unknown_values_survive_the_transport_as_nulls(mcp):
    data = {"process_id": "p", "name": "n", "steps": [{"id": "a", "name": "A"}, step("b", 10, True)]}
    m = mcp.calculate_process_metrics(data)
    assert m["steps_without_duration"] == ["a"] and m["time_is_complete"] is False
    assert m["unknown_type_steps"] == 1 and m["runs_per_month"] is None and m["monthly_effort_minutes"] is None


# ---------- некорректные данные ----------


def test_validate_process_reports_problems_instead_of_failing(mcp):
    bad = {"process_id": "p", "name": "n", "steps": [step("a", -5, True, next_steps=["zz"]), step("a", "10", True)]}
    report = mcp.validate_process(bad)
    codes = {e["code"] for e in report["errors"]}
    assert report["valid"] is False and {"duplicate_step_id", "invalid_value", "unknown_next_step"} <= codes


def test_validate_process_handles_missing_fields(mcp):
    report = mcp.validate_process({"steps": []})
    assert report["valid"] is False
    assert {(e["code"], e["path"]) for e in report["errors"]} >= {("missing_field", "process_id"), ("missing_field", "name"), ("no_steps", "steps")}


@pytest.mark.parametrize("tool", ["calculate_process_metrics", "simulate_automation"])
def test_invalid_process_makes_calculation_tools_fail_with_reasons(mcp, tool):
    bad = {"process_id": "p", "name": "n", "steps": [step("a", 1, True), step("a", 2, True)]}
    args = {"process": bad, "automation": [{"step_id": "a", "new_duration_minutes": 1}]} if tool == "simulate_automation" else {"process": bad}
    with pytest.raises(McpToolError, match="не прошёл валидацию.*уже использован"):
        mcp.call_tool(tool, args)


def test_negative_duration_rejected_by_calculation_tools(mcp):
    bad = {"process_id": "p", "name": "n", "steps": [step("a", -1, True)]}
    with pytest.raises(McpToolError, match="duration_minutes"):
        mcp.calculate_process_metrics(bad)


@pytest.mark.parametrize(
    ("automation", "message"),
    [
        ([], "ни одной операции"),
        ([{"step_id": "nope", "new_duration_minutes": 1}], "нет в процессе"),
        ([{"step_id": "s2", "new_duration_minutes": 1}, {"step_id": "s2", "new_duration_minutes": 2}], "несколько раз"),
        ([{"step_id": "s2", "new_duration_minutes": -1}], "greater_than_equal|greater than or equal"),
        ([{"step_id": "s2"}], "new_duration_minutes"),  # длительность после автоматизации обязательна
        ([{"step_id": "s2", "new_duration_minutes": 1, "reduction_percent": 50}], "reduction_percent"),  # процент — не принимается
    ],
)
def test_simulation_rejects_bad_assumptions(mcp, automation, message):
    with pytest.raises(McpToolError, match=message):
        mcp.simulate_automation(PROCESS, automation)


def test_simulation_rejects_step_with_unknown_original_duration(mcp):
    data = {"process_id": "p", "name": "n", "steps": [{"id": "a", "name": "A"}, step("b", 10, True)]}
    with pytest.raises(McpToolError, match="не указана исходная длительность"):
        mcp.simulate_automation(data, [{"step_id": "a", "new_duration_minutes": 1}])


@pytest.mark.parametrize("runs", [-1, "много", 2.5])
def test_invalid_runs_per_month_rejected(mcp, runs):
    with pytest.raises(McpToolError):
        mcp.call_tool("calculate_process_metrics", {"process": PROCESS, "runs_per_month": runs})


def test_missing_and_mistyped_arguments_rejected(mcp):
    with pytest.raises(McpToolError):
        mcp.call_tool("calculate_process_metrics", {})
    with pytest.raises(McpToolError):
        mcp.call_tool("validate_process", {"process": "не словарь"})


def test_unknown_tool_rejected(mcp):
    with pytest.raises(McpToolError):
        mcp.call_tool("no_such_tool", {})


def test_server_survives_errors_and_keeps_serving(mcp):
    with pytest.raises(McpToolError):
        mcp.call_tool("calculate_process_metrics", {})
    assert mcp.calculate_process_metrics(PROCESS)["total_steps"] == 4


# ---------- асинхронный клиент и жизненный цикл ----------


def test_async_client_full_roundtrip():
    async def scenario():
        async with MCPClient() as client:
            assert TOOLS <= set(await client.list_tools())
            metrics = await client.calculate_process_metrics(PROCESS, 10)
            with pytest.raises(McpToolError):
                await client.simulate_automation(PROCESS, [])
            return metrics

    assert asyncio.run(scenario())["total_time_minutes"] == 300


def test_client_requires_start():
    with pytest.raises(McpClientError, match="не запущен"):
        SyncMCPClient().list_tools()

    async def not_entered():
        await MCPClient().list_tools()

    with pytest.raises(McpClientError, match="не запущен"):
        asyncio.run(not_entered())


def test_closed_client_refuses_calls():
    client = SyncMCPClient().start()
    assert client.calculate_process_metrics(PROCESS)["total_steps"] == 4
    client.close()
    with pytest.raises(McpClientError, match="не запущен"):
        client.list_tools()


def test_server_process_that_is_not_an_mcp_server_fails_fast_with_clear_error():
    params = StdioServerParameters(command=sys.executable, args=["-c", "import time; time.sleep(30)"])
    with pytest.raises(McpClientError, match="Не удалось запустить"):
        SyncMCPClient(params, timeout=2).start()


def test_missing_server_executable_gives_clear_error():
    with pytest.raises(McpClientError):
        SyncMCPClient(StdioServerParameters(command="definitely-not-a-real-executable-xyz"), timeout=3).start()


def test_default_server_parameters_launch_module_with_same_interpreter():
    params = default_server_parameters()
    assert params.command == sys.executable and params.args == ["-m", "processmind.mcp.server"]
