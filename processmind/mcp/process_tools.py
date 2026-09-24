"""Расчётное ядро MCP-инструментов: валидация процесса, метрики, симуляция автоматизации.

Только обычный Python (без LLM и без зависимости от MCP): функции детерминированы и тестируются
напрямую. В приложении они вызываются исключительно через MCP-сервер (см. `server.py`) и MCP-клиент
(`client.py`) — отдельным процессом по stdio.

Принципы:
- неизвестное остаётся неизвестным: неуказанная длительность не превращается в 0, тип выполнения
  «не указан» не считается ни ручным, ни автоматизированным;
- никаких «типовых процентов сокращения»: время после автоматизации задаёт вызывающий;
- время суммируется по всем операциям (процесс считается выполняемым последовательно, все ветки
  учитываются) — об этом всегда сообщается в `notes`.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from processmind.models import ProcessSpec, ProcessStep

MINUTES_PER_HOUR = 60.0
ROUND_DIGITS = 4

DISCLAIMER = (
    "Модельная оценка на основе заданных предположений. Это не измерение и не прогноз: результат "
    "верен ровно настолько, насколько верны исходные длительности операций, предполагаемые "
    "длительности после автоматизации и число запусков процесса."
)


class ProcessToolError(ValueError):
    """Входные данные не позволяют выполнить расчёт (сообщение показывается вызывающему)."""


# ---------- модели результатов ----------


class Issue(BaseModel):
    code: str
    path: str
    message: str


class ValidationReport(BaseModel):
    valid: bool
    errors: list[Issue]
    warnings: list[Issue]
    steps_checked: int


class ProcessMetrics(BaseModel):
    total_steps: int
    manual_steps: int
    automated_steps: int
    unknown_type_steps: int
    manual_share: float = Field(description="Доля ручных операций (операции с неуказанным типом не считаются ручными)")
    manual_share_upper_bound: float = Field(description="Доля ручных, если ВСЕ операции с неуказанным типом ручные")
    total_time_minutes: float = Field(description="Сумма известных длительностей всех операций одного экземпляра")
    manual_time_minutes: float
    automated_time_minutes: float
    unclassified_time_minutes: float = Field(description="Время операций с неуказанным типом выполнения")
    steps_without_duration: list[str]
    time_is_complete: bool = Field(description="False, если у части операций длительность не указана — суммы занижены")
    has_branching: bool
    runs_per_month: int | None
    monthly_effort_minutes: float | None
    monthly_effort_hours: float | None
    monthly_manual_effort_minutes: float | None
    monthly_manual_effort_hours: float | None
    warnings: list[str]
    notes: list[str]


class AutomationChange(BaseModel):
    """Операция, которую предполагается автоматизировать, и её длительность после автоматизации."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(description="ID операции из ProcessSpec")
    new_duration_minutes: float = Field(
        ge=0, allow_inf_nan=False, description="Предполагаемая длительность операции после автоматизации, минуты"
    )


class AppliedChange(BaseModel):
    step_id: str
    step_name: str
    original_duration_minutes: float
    new_duration_minutes: float
    was_manual: bool | None


class SimulationResult(BaseModel):
    is_model_estimate: bool
    disclaimer: str
    assumptions: list[AppliedChange]
    baseline_time_minutes: float
    proposed_time_minutes: float
    absolute_reduction_minutes: float
    reduction_percent: float | None
    baseline_manual_share: float
    proposed_manual_share: float
    baseline_manual_share_upper_bound: float
    proposed_manual_share_upper_bound: float
    baseline_manual_time_minutes: float
    proposed_manual_time_minutes: float
    time_is_complete: bool
    runs_per_month: int | None
    monthly_saving_minutes: float | None
    monthly_saving_hours: float | None
    warnings: list[str]
    notes: list[str]


# ---------- валидация ----------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _schema_issues(exc: ValidationError) -> list[Issue]:
    issues = []
    for err in exc.errors():
        path = ""
        for part in err["loc"]:
            path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
        if err["type"] == "missing":
            issues.append(Issue(code="missing_field", path=path, message="Обязательное поле отсутствует"))
        else:
            issues.append(Issue(code="invalid_value", path=path, message=f"Некорректное значение: {err['msg']}"))
    return issues


def _find_cycle_members(graph: dict[str, list[str]]) -> set[str]:
    """Вершины, лежащие на циклах (итеративный поиск сильно связных компонент, алгоритм Тарьяна)."""
    index, low, on_stack = {}, {}, set()
    stack: list[str] = []
    members: set[str] = set()
    counter = 0
    for root in graph:
        if root in index:
            continue
        work = [(root, iter(graph[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(graph[child])))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1 or node in graph[node]:
                    members.update(component)
    return members


def validate_process_data(data: Any) -> ValidationReport:
    """Проверяет структуру процесса; сам никогда не бросает исключений на «плохих» данных."""
    errors: list[Issue] = []
    warnings: list[Issue] = []

    if not isinstance(data, dict):
        errors.append(Issue(code="invalid_type", path="", message="Процесс должен быть объектом (словарём) ProcessSpec"))
        return ValidationReport(valid=False, errors=errors, warnings=warnings, steps_checked=0)

    # 1. Схема ProcessSpec (строгая: "30" вместо 30 или "yes" вместо true — ошибка).
    try:
        ProcessSpec.model_validate(data, strict=True)
    except ValidationError as exc:
        errors.extend(_schema_issues(exc))

    for key in data:
        if key not in ProcessSpec.model_fields:
            warnings.append(Issue(code="unknown_field", path=str(key), message="Неизвестное поле будет проигнорировано"))

    # 2. Обязательные поля не должны быть пустыми.
    for field in ("process_id", "name"):
        if isinstance(data.get(field), str) and not data[field].strip():
            errors.append(Issue(code="empty_field", path=field, message="Поле не может быть пустым"))

    raw_steps = data.get("steps")
    if "steps" not in data:  # в модели у steps есть значение по умолчанию, но для процесса оно обязательно
        errors.append(Issue(code="missing_field", path="steps", message="Обязательное поле отсутствует"))
    if not isinstance(raw_steps, list):
        return ValidationReport(valid=not errors, errors=errors, warnings=warnings, steps_checked=0)
    if not raw_steps:
        errors.append(Issue(code="no_steps", path="steps", message="В процессе нет ни одной операции"))

    # 3. Операции: ID, названия, длительности, обязательные признаки.
    seen: dict[str, int] = {}
    valid_ids: list[str] = []
    steps = [(i, s) for i, s in enumerate(raw_steps) if isinstance(s, dict)]
    for i, step in steps:
        base = f"steps[{i}]"
        for key in step:
            if key not in ProcessStep.model_fields:
                warnings.append(Issue(code="unknown_field", path=f"{base}.{key}", message="Неизвестное поле будет проигнорировано"))

        step_id = step.get("id")
        if isinstance(step_id, str):
            if not step_id.strip():
                errors.append(Issue(code="empty_field", path=f"{base}.id", message="ID операции не может быть пустым"))
            elif step_id in seen:
                errors.append(
                    Issue(
                        code="duplicate_step_id",
                        path=f"{base}.id",
                        message=f"ID «{step_id}» уже использован операцией steps[{seen[step_id]}]",
                    )
                )
            else:
                seen[step_id] = i
                valid_ids.append(step_id)

        if isinstance(step.get("name"), str) and not step["name"].strip():
            errors.append(Issue(code="empty_field", path=f"{base}.name", message="Название операции не может быть пустым"))

        duration = step.get("duration_minutes")
        if duration is None:
            warnings.append(
                Issue(code="missing_duration", path=f"{base}.duration_minutes", message="Длительность не указана: метрики времени будут неполными")
            )
        elif _is_number(duration):
            if not math.isfinite(duration):
                errors.append(Issue(code="invalid_duration", path=f"{base}.duration_minutes", message="Длительность должна быть конечным числом"))
            elif duration == 0:
                warnings.append(Issue(code="zero_duration", path=f"{base}.duration_minutes", message="Нулевая длительность: проверьте, что это не пропуск данных"))
            # отрицательные значения уже отмечены схемой (ge=0)

        if step.get("is_manual") is None:
            warnings.append(Issue(code="missing_execution_type", path=f"{base}.is_manual", message="Тип выполнения (ручной/автоматизированный) не указан"))
        if not step.get("actor"):
            warnings.append(Issue(code="missing_actor", path=f"{base}.actor", message="Исполнитель не указан"))

    # 4. Связи между операциями.
    graph: dict[str, list[str]] = {}
    known = set(valid_ids)
    for i, step in steps:
        step_id = step.get("id")
        links = step.get("next_steps")
        if not (isinstance(step_id, str) and step_id in known and isinstance(links, list)):
            continue
        targets = []
        for j, target in enumerate(links):
            if not isinstance(target, str):
                continue
            if target not in known:
                errors.append(
                    Issue(code="unknown_next_step", path=f"steps[{i}].next_steps[{j}]", message=f"Связь ведёт на несуществующую операцию «{target}»")
                )
            else:
                targets.append(target)
        graph[step_id] = targets

    if any(graph.values()):
        cyclic = _find_cycle_members({sid: graph.get(sid, []) for sid in valid_ids})
        for sid in valid_ids:
            if sid in cyclic:
                warnings.append(Issue(code="cycle", path=f"steps[{seen[sid]}]", message=f"Операция «{sid}» входит в цикл: суммарное время не учитывает повторения"))
        reachable, frontier = set(), [valid_ids[0]]
        while frontier:
            node = frontier.pop()
            if node not in reachable:
                reachable.add(node)
                frontier.extend(graph.get(node, []))
        for sid in valid_ids:
            if sid not in reachable:
                warnings.append(Issue(code="unreachable_step", path=f"steps[{seen[sid]}]", message=f"Операция «{sid}» недостижима от первой операции"))

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, steps_checked=len(raw_steps))


def load_valid_spec(data: Any) -> tuple[ProcessSpec, ValidationReport]:
    """ProcessSpec из сырых данных; при любых ошибках валидации — ProcessToolError со списком причин."""
    report = validate_process_data(data)
    if not report.valid:
        details = "; ".join(f"{e.path or 'process'}: {e.message}" for e in report.errors)
        raise ProcessToolError(f"Процесс не прошёл валидацию ({len(report.errors)} ошибок): {details}")
    return ProcessSpec.model_validate(data, strict=True), report


# ---------- метрики ----------


def _r(value: float) -> float:
    return round(value, ROUND_DIGITS)


def _check_runs(runs_per_month: int | None) -> None:
    if runs_per_month is not None and (isinstance(runs_per_month, bool) or runs_per_month < 0):
        raise ProcessToolError("runs_per_month должно быть неотрицательным целым числом")


def _manual_shares(steps: list[ProcessStep]) -> tuple[float, float]:
    total = len(steps)
    manual = sum(1 for s in steps if s.is_manual is True)
    unknown = sum(1 for s in steps if s.is_manual is None)
    return manual / total, (manual + unknown) / total


def _time_by_type(steps: list[ProcessStep]) -> dict[str, float]:
    """Суммы известных длительностей; неуказанная длительность в сумму не входит (и не равна нулю — см. missing)."""
    parts: dict[str, list[float]] = {"total": [], "manual": [], "automated": [], "unclassified": []}
    for s in steps:
        minutes = s.duration_minutes or 0.0
        parts["total"].append(minutes)
        parts["manual" if s.is_manual is True else "automated" if s.is_manual is False else "unclassified"].append(minutes)
    return {k: math.fsum(v) for k, v in parts.items()}


def _has_branching(spec: ProcessSpec) -> bool:
    return any(len(s.next_steps) > 1 for s in spec.steps)


def _common_notes(spec: ProcessSpec, missing: list[str], report_warnings: list[str]) -> tuple[list[str], list[str]]:
    warnings = list(report_warnings)
    if missing:
        warnings.append(f"Длительность не указана у операций: {', '.join(missing)} — суммы времени занижены")
    notes = ["Время суммируется по всем операциям: процесс считается выполняемым последовательно."]
    if _has_branching(spec):
        warnings.append("В процессе есть ветвления: сумма включает все ветки, хотя в одном экземпляре выполняется одна из них")
    return warnings, notes


def compute_metrics(spec: ProcessSpec, runs_per_month: int | None = None) -> ProcessMetrics:
    _check_runs(runs_per_month)
    steps = spec.steps
    share, share_upper = _manual_shares(steps)
    times = _time_by_type(steps)
    missing = [s.id for s in steps if s.duration_minutes is None]
    unknown_type = [s.id for s in steps if s.is_manual is None]

    warnings, notes = _common_notes(spec, missing, [])
    if unknown_type:
        warnings.append(
            f"Тип выполнения не указан у операций: {', '.join(unknown_type)} — они не считаются ручными; "
            "см. manual_share_upper_bound"
        )
    notes.append("Месячная трудоёмкость — условная: (время одного экземпляра) × (число запусков в месяц).")
    if runs_per_month is None:
        notes.append("runs_per_month не задано — месячные показатели не рассчитаны.")

    def monthly(minutes: float) -> float | None:
        return None if runs_per_month is None else _r(minutes * runs_per_month)

    def hours(minutes: float | None) -> float | None:
        return None if minutes is None else _r(minutes / MINUTES_PER_HOUR)

    monthly_total, monthly_manual = monthly(times["total"]), monthly(times["manual"])
    return ProcessMetrics(
        total_steps=len(steps),
        manual_steps=sum(1 for s in steps if s.is_manual is True),
        automated_steps=sum(1 for s in steps if s.is_manual is False),
        unknown_type_steps=len(unknown_type),
        manual_share=_r(share),
        manual_share_upper_bound=_r(share_upper),
        total_time_minutes=_r(times["total"]),
        manual_time_minutes=_r(times["manual"]),
        automated_time_minutes=_r(times["automated"]),
        unclassified_time_minutes=_r(times["unclassified"]),
        steps_without_duration=missing,
        time_is_complete=not missing,
        has_branching=_has_branching(spec),
        runs_per_month=runs_per_month,
        monthly_effort_minutes=monthly_total,
        monthly_effort_hours=hours(monthly_total),
        monthly_manual_effort_minutes=monthly_manual,
        monthly_manual_effort_hours=hours(monthly_manual),
        warnings=warnings,
        notes=notes,
    )


# ---------- симуляция автоматизации ----------


def simulate(spec: ProcessSpec, changes: list[AutomationChange], runs_per_month: int | None = None) -> SimulationResult:
    """Пересчитывает процесс при заданных предположениях; процент сокращения НЕ подставляется автоматически."""
    _check_runs(runs_per_month)
    if not changes:
        raise ProcessToolError("Не выбрано ни одной операции для автоматизации")

    by_id = {s.id: s for s in spec.steps}
    seen: set[str] = set()
    applied: list[AppliedChange] = []
    warnings: list[str] = []
    for change in changes:
        if change.step_id not in by_id:
            raise ProcessToolError(f"Операции «{change.step_id}» нет в процессе")
        if change.step_id in seen:
            raise ProcessToolError(f"Операция «{change.step_id}» указана для автоматизации несколько раз")
        seen.add(change.step_id)
        step = by_id[change.step_id]
        if step.duration_minutes is None:
            raise ProcessToolError(
                f"У операции «{step.id}» не указана исходная длительность — сравнение до/после невозможно. "
                "Укажите длительность в процессе."
            )
        if step.is_manual is False:
            warnings.append(f"Операция «{step.id}» уже помечена как автоматизированная")
        if change.new_duration_minutes > step.duration_minutes:
            warnings.append(
                f"Операция «{step.id}»: длительность после автоматизации ({change.new_duration_minutes:g} мин) "
                f"больше исходной ({step.duration_minutes:g} мин) — время процесса вырастет"
            )
        applied.append(
            AppliedChange(
                step_id=step.id,
                step_name=step.name,
                original_duration_minutes=step.duration_minutes,
                new_duration_minutes=change.new_duration_minutes,
                was_manual=step.is_manual,
            )
        )

    new_by_id = {c.step_id: c.new_duration_minutes for c in changes}
    proposed_steps = [
        s.model_copy(update={"duration_minutes": new_by_id[s.id], "is_manual": False}) if s.id in new_by_id else s
        for s in spec.steps
    ]

    baseline_times, proposed_times = _time_by_type(spec.steps), _time_by_type(proposed_steps)
    base_share, base_upper = _manual_shares(spec.steps)
    prop_share, prop_upper = _manual_shares(proposed_steps)
    missing = [s.id for s in spec.steps if s.duration_minutes is None]

    baseline, proposed = baseline_times["total"], proposed_times["total"]
    reduction = baseline - proposed
    monthly = None if runs_per_month is None else _r(reduction * runs_per_month)

    more_warnings, notes = _common_notes(spec, missing, warnings)
    notes.append("Показатели «после» рассчитаны из предполагаемых длительностей, которые задал вызывающий; ничего не подставлялось автоматически.")
    notes.append("Автоматизированные операции считаются не ручными; остальные операции остаются как в исходном процессе.")
    if runs_per_month is None:
        notes.append("runs_per_month не задано — месячная экономия не рассчитана.")

    return SimulationResult(
        is_model_estimate=True,
        disclaimer=DISCLAIMER,
        assumptions=applied,
        baseline_time_minutes=_r(baseline),
        proposed_time_minutes=_r(proposed),
        absolute_reduction_minutes=_r(reduction),
        reduction_percent=_r(reduction / baseline * 100) if baseline > 0 else None,
        baseline_manual_share=_r(base_share),
        proposed_manual_share=_r(prop_share),
        baseline_manual_share_upper_bound=_r(base_upper),
        proposed_manual_share_upper_bound=_r(prop_upper),
        baseline_manual_time_minutes=_r(baseline_times["manual"]),
        proposed_manual_time_minutes=_r(proposed_times["manual"]),
        time_is_complete=not missing,
        runs_per_month=runs_per_month,
        monthly_saving_minutes=monthly,
        monthly_saving_hours=None if monthly is None else _r(monthly / MINUTES_PER_HOUR),
        warnings=more_warnings,
        notes=notes,
    )
