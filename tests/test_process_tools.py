"""Тесты расчётного ядра MCP-инструментов: валидация, метрики, симуляция.

Ожидаемые числа посчитаны вручную (в комментариях). Здесь проверяется сама математика; то, что
инструменты действительно вызываются через MCP по stdio, проверяют tests/test_mcp_process_tools.py.
"""

import copy
import math

import pytest

from processmind.mcp.process_tools import (
    AutomationChange,
    ProcessToolError,
    compute_metrics,
    load_valid_spec,
    simulate,
    validate_process_data,
)


def step(sid, minutes, manual, **extra):
    data = {"id": sid, "name": f"Операция {sid}", "duration_minutes": minutes, "is_manual": manual, "actor": "Сотрудник"}
    data.update(extra)
    return data


def process(*steps, **extra):
    return {"process_id": "p1", "name": "Процесс", "steps": list(steps), **extra}


# Процесс A: неполные данные — есть операция без длительности и с неуказанным типом.
PROCESS_A = process(
    step("s1", 30, True),
    step("s2", 120, True),
    step("s3", 60, False),
    step("s4", 10, None),
    step("s5", None, True),
)
# Процесс B: полные данные. Сумма: 30 + 120 + 60 + 90 = 300 мин; ручные: 30 + 120 + 90 = 240 мин.
PROCESS_B = process(step("s1", 30, True), step("s2", 120, True), step("s3", 60, False), step("s4", 90, True))


def spec_of(data):
    return load_valid_spec(data)[0]


def codes(issues):
    return [i.code for i in issues]


# ---------- метрики ----------


def test_metrics_complete_process_hand_calculated():
    m = compute_metrics(spec_of(PROCESS_B), runs_per_month=50)
    assert (m.total_steps, m.manual_steps, m.automated_steps, m.unknown_type_steps) == (4, 3, 1, 0)
    assert m.manual_share == pytest.approx(3 / 4)  # 0.75
    assert m.manual_share_upper_bound == pytest.approx(0.75)
    assert m.total_time_minutes == 300 and m.manual_time_minutes == 240 and m.automated_time_minutes == 60
    assert m.time_is_complete and m.steps_without_duration == []
    assert m.monthly_effort_minutes == 300 * 50 == 15000
    assert m.monthly_effort_hours == pytest.approx(250.0)  # 15000 / 60
    assert m.monthly_manual_effort_minutes == 240 * 50 == 12000
    assert m.monthly_manual_effort_hours == pytest.approx(200.0)


def test_metrics_incomplete_process_does_not_invent_missing_values():
    m = compute_metrics(spec_of(PROCESS_A), runs_per_month=200)
    assert (m.total_steps, m.manual_steps, m.automated_steps, m.unknown_type_steps) == (5, 3, 1, 1)
    assert m.manual_share == pytest.approx(3 / 5)  # неизвестный тип НЕ считается ручным
    assert m.manual_share_upper_bound == pytest.approx(4 / 5)  # верхняя граница: неизвестный — ручной
    assert m.total_time_minutes == 220  # 30 + 120 + 60 + 10; s5 без длительности не равна нулю, а не учтена
    assert m.manual_time_minutes == 150 and m.automated_time_minutes == 60 and m.unclassified_time_minutes == 10
    assert m.steps_without_duration == ["s5"] and m.time_is_complete is False
    assert m.monthly_effort_minutes == 44000 and m.monthly_effort_hours == pytest.approx(733.3333, abs=1e-4)
    assert any("s5" in w for w in m.warnings) and any("s4" in w for w in m.warnings)


def test_monthly_metrics_absent_without_runs_per_month():
    m = compute_metrics(spec_of(PROCESS_B))
    assert m.runs_per_month is None
    assert m.monthly_effort_minutes is None and m.monthly_effort_hours is None
    assert m.monthly_manual_effort_minutes is None and m.monthly_manual_effort_hours is None
    assert m.total_time_minutes == 300  # остальное считается


def test_zero_runs_gives_zero_monthly_effort():
    assert compute_metrics(spec_of(PROCESS_B), 0).monthly_effort_minutes == 0


def test_float_durations_summed_without_drift():
    data = process(*[step(f"s{i}", 0.1, True) for i in range(10)])
    assert compute_metrics(spec_of(data)).total_time_minutes == 1.0  # 10 × 0.1 без накопления погрешности


def test_metrics_flags_branching():
    data = process(step("a", 10, True, next_steps=["b", "c"]), step("b", 5, True), step("c", 5, True))
    m = compute_metrics(spec_of(data))
    assert m.has_branching and any("ветвлен" in w for w in m.warnings)
    assert m.total_time_minutes == 20  # сумма по всем веткам — и об этом предупреждается


@pytest.mark.parametrize("runs", [-1, True])
def test_metrics_reject_invalid_runs(runs):
    with pytest.raises(ProcessToolError, match="runs_per_month"):
        compute_metrics(spec_of(PROCESS_B), runs)


# ---------- симуляция ----------


def test_simulation_hand_calculated():
    changes = [AutomationChange(step_id="s2", new_duration_minutes=10), AutomationChange(step_id="s4", new_duration_minutes=15)]
    r = simulate(spec_of(PROCESS_B), changes, runs_per_month=50)

    assert r.baseline_time_minutes == 300
    assert r.proposed_time_minutes == 30 + 10 + 60 + 15 == 115
    assert r.absolute_reduction_minutes == 185
    assert r.reduction_percent == pytest.approx(185 / 300 * 100, abs=1e-4)  # 61.6667
    assert r.baseline_manual_share == pytest.approx(0.75)
    assert r.proposed_manual_share == pytest.approx(0.25)  # ручной остаётся только s1
    assert r.baseline_manual_time_minutes == 240 and r.proposed_manual_time_minutes == 30
    assert r.monthly_saving_minutes == 185 * 50 == 9250
    assert r.monthly_saving_hours == pytest.approx(9250 / 60, abs=1e-4)  # 154.1667


def test_simulation_is_labelled_model_estimate_and_lists_assumptions():
    r = simulate(spec_of(PROCESS_B), [AutomationChange(step_id="s2", new_duration_minutes=10)])
    assert r.is_model_estimate is True
    assert "Модельная оценка" in r.disclaimer and "предположени" in r.disclaimer
    (a,) = r.assumptions
    assert (a.step_id, a.original_duration_minutes, a.new_duration_minutes, a.was_manual) == ("s2", 120, 10, True)


def test_no_percentage_reduction_is_applied_automatically():
    """Если пользователь задал ту же длительность — сокращения нет: ничего не подставляется «по умолчанию»."""
    r = simulate(spec_of(PROCESS_B), [AutomationChange(step_id="s2", new_duration_minutes=120)])
    assert r.proposed_time_minutes == r.baseline_time_minutes == 300
    assert r.absolute_reduction_minutes == 0 and r.reduction_percent == 0
    assert r.proposed_manual_share == pytest.approx(0.5)  # изменился только тип выполнения


def test_new_duration_is_required():
    with pytest.raises(ValueError):
        AutomationChange.model_validate({"step_id": "s2"})


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_new_duration_must_be_finite_non_negative(bad):
    with pytest.raises(ValueError):
        AutomationChange(step_id="s2", new_duration_minutes=bad)


def test_slower_automation_gives_negative_reduction_and_warning():
    r = simulate(spec_of(PROCESS_B), [AutomationChange(step_id="s1", new_duration_minutes=40)])
    assert r.absolute_reduction_minutes == -10 and r.reduction_percent == pytest.approx(-10 / 300 * 100, abs=1e-4)
    assert any("вырастет" in w for w in r.warnings)


def test_simulation_on_incomplete_process_stays_like_for_like():
    r = simulate(spec_of(PROCESS_A), [AutomationChange(step_id="s2", new_duration_minutes=20)])
    assert r.baseline_time_minutes == 220 and r.proposed_time_minutes == 120  # s5 без длительности — в обеих суммах одинаково
    assert r.time_is_complete is False and any("s5" in w for w in r.warnings)
    assert r.baseline_manual_share_upper_bound == pytest.approx(4 / 5)
    assert r.proposed_manual_share == pytest.approx(2 / 5) and r.proposed_manual_share_upper_bound == pytest.approx(3 / 5)


def test_percent_is_none_when_baseline_is_zero():
    data = process(step("a", 0, True), step("b", 0, True))
    r = simulate(spec_of(data), [AutomationChange(step_id="a", new_duration_minutes=0)])
    assert r.baseline_time_minutes == 0 and r.reduction_percent is None


def test_simulation_monthly_saving_absent_without_runs():
    r = simulate(spec_of(PROCESS_B), [AutomationChange(step_id="s2", new_duration_minutes=10)])
    assert r.monthly_saving_minutes is None and r.monthly_saving_hours is None


def test_already_automated_step_is_warned_about():
    r = simulate(spec_of(PROCESS_B), [AutomationChange(step_id="s3", new_duration_minutes=30)])
    assert any("уже помечена как автоматизированная" in w for w in r.warnings)


@pytest.mark.parametrize(
    ("changes", "runs", "message"),
    [
        ([], None, "ни одной операции"),
        ([AutomationChange(step_id="zzz", new_duration_minutes=1)], None, "нет в процессе"),
        ([AutomationChange(step_id="s2", new_duration_minutes=1)] * 2, None, "несколько раз"),
        ([AutomationChange(step_id="s2", new_duration_minutes=1)], -5, "runs_per_month"),
    ],
)
def test_simulation_input_errors(changes, runs, message):
    with pytest.raises(ProcessToolError, match=message):
        simulate(spec_of(PROCESS_B), changes, runs)


def test_simulation_requires_known_original_duration():
    with pytest.raises(ProcessToolError, match="не указана исходная длительность"):
        simulate(spec_of(PROCESS_A), [AutomationChange(step_id="s5", new_duration_minutes=1)])


def test_simulation_does_not_mutate_input_spec():
    spec = spec_of(PROCESS_B)
    before = spec.model_dump()
    simulate(spec, [AutomationChange(step_id="s2", new_duration_minutes=10)])
    assert spec.model_dump() == before


# ---------- валидация ----------


def test_valid_process_has_no_errors():
    report = validate_process_data(PROCESS_B)
    assert report.valid and report.errors == [] and report.steps_checked == 4


def test_valid_process_with_missing_optional_data_gets_warnings_not_errors():
    report = validate_process_data(PROCESS_A)
    assert report.valid
    assert {"missing_duration", "missing_execution_type"} <= set(codes(report.warnings))
    assert next(w for w in report.warnings if w.code == "missing_duration").path == "steps[4].duration_minutes"


def test_duplicate_step_ids_are_errors():
    report = validate_process_data(process(step("a", 1, True), step("a", 2, True)))
    (err,) = [e for e in report.errors if e.code == "duplicate_step_id"]
    assert err.path == "steps[1].id" and "steps[0]" in err.message and not report.valid


@pytest.mark.parametrize("duration", [-5, "30", float("nan"), float("inf"), True])
def test_invalid_durations_are_errors(duration):
    report = validate_process_data(process(step("a", duration, True)))
    assert not report.valid and any(e.path == "steps[0].duration_minutes" for e in report.errors)


def test_zero_duration_is_only_a_warning():
    report = validate_process_data(process(step("a", 0, True)))
    assert report.valid and "zero_duration" in codes(report.warnings)


@pytest.mark.parametrize("field", ["process_id", "name", "steps"])
def test_missing_required_top_level_fields(field):
    data = copy.deepcopy(PROCESS_B)
    del data[field]
    report = validate_process_data(data)
    assert not report.valid and any(e.code == "missing_field" and e.path == field for e in report.errors)


@pytest.mark.parametrize("field", ["id", "name"])
def test_missing_required_step_fields(field):
    data = copy.deepcopy(PROCESS_B)
    del data["steps"][2][field]
    report = validate_process_data(data)
    assert any(e.code == "missing_field" and e.path == f"steps[2].{field}" for e in report.errors)


def test_blank_required_values_are_errors():
    report = validate_process_data({"process_id": " ", "name": "", "steps": [step("", 1, True), step("b", 1, True, name=" ")]})
    assert {e.path for e in report.errors if e.code == "empty_field"} == {"process_id", "name", "steps[0].id", "steps[1].name"}


def test_no_steps_is_an_error():
    report = validate_process_data(process())
    assert not report.valid and "no_steps" in codes(report.errors)


def test_strict_types_reject_stringly_typed_values():
    report = validate_process_data(process(step("a", 1, "yes")))
    assert any(e.path == "steps[0].is_manual" for e in report.errors)


@pytest.mark.parametrize("payload", [None, [], "процесс", 42])
def test_non_object_input_is_reported_not_raised(payload):
    report = validate_process_data(payload)
    assert not report.valid and report.errors[0].code == "invalid_type"


def test_dangling_next_step_is_an_error():
    report = validate_process_data(process(step("a", 1, True, next_steps=["nope"])))
    assert any(e.code == "unknown_next_step" and e.path == "steps[0].next_steps[0]" for e in report.errors)


def test_cycles_and_unreachable_steps_are_warnings():
    data = process(
        step("a", 1, True, next_steps=["b"]),
        step("b", 1, True, next_steps=["a"]),
        step("c", 1, True),
    )
    report = validate_process_data(data)
    assert report.valid
    assert sorted(w.path for w in report.warnings if w.code == "cycle") == ["steps[0]", "steps[1]"]
    assert [w.path for w in report.warnings if w.code == "unreachable_step"] == ["steps[2]"]


def test_self_loop_counts_as_cycle():
    report = validate_process_data(process(step("a", 1, True, next_steps=["a"])))
    assert "cycle" in codes(report.warnings)


def test_long_chain_is_not_a_cycle():
    steps = [step(f"s{i}", 1, True, next_steps=[f"s{i + 1}"] if i < 200 else []) for i in range(201)]
    report = validate_process_data(process(*steps))
    assert report.valid and "cycle" not in codes(report.warnings) and "unreachable_step" not in codes(report.warnings)


def test_unknown_fields_are_warned_about():
    data = process(step("a", 1, True, colour="red"), extra_top=1)
    report = validate_process_data(data)
    assert {w.path for w in report.warnings if w.code == "unknown_field"} == {"extra_top", "steps[0].colour"}


def test_load_valid_spec_raises_with_all_reasons():
    with pytest.raises(ProcessToolError, match="не прошёл валидацию") as exc:
        load_valid_spec(process(step("a", -1, True), step("a", 1, True)))
    assert "duration_minutes" in str(exc.value) and "уже использован" in str(exc.value)


def test_numbers_are_finite_in_results():
    m = compute_metrics(spec_of(PROCESS_B), 10).model_dump()
    assert all(math.isfinite(v) for v in m.values() if isinstance(v, float))
