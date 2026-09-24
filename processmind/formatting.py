"""Единое форматирование значений для интерфейса, диаграмм и отчёта."""

from __future__ import annotations

MISSING = "не указано"


def _num(value: float) -> str:
    """Число без лишних нулей, с десятичной запятой."""
    return f"{value:g}".replace(".", ",")


def format_duration(minutes: float | None, *, with_minutes: bool = False, missing: str = MISSING) -> str:
    """Длительность в удобных единицах: минуты, часы или дни.

    with_minutes=True — для сумм: «4535 мин (≈ 3,1 дн)», чтобы точное значение не терялось.
    """
    if minutes is None:
        return missing
    if minutes < 60:
        return f"{_num(minutes)} мин"
    if minutes < 1440:
        human = f"{_num(round(minutes / 60, 1))} ч"
    else:
        human = f"{_num(round(minutes / 1440, 1))} дн"
    return f"{_num(minutes)} мин (≈ {human})" if with_minutes else human


def format_hours(hours: float | None, *, missing: str = MISSING) -> str:
    return missing if hours is None else f"{_num(round(hours, 1))} ч"


def format_share(share: float) -> str:
    return f"{round(share * 100):d}%"
