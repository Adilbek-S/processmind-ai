"""Схемы процесса в Graphviz: AS-IS и TO-BE.

AS-IS: последовательность операций, название, исполнитель, длительность; ручные операции выделены цветом.
TO-BE: та же последовательность и те же связи (маршруты не перестраиваются, шлюзы не добавляются);
операции, для которых принята автоматизация, выделены и подписаны «было → стало» и паттерном.

Метки узлов — HTML-подобные таблицы Graphviz; весь пользовательский текст экранируется.
"""

from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass
from typing import Sequence

import graphviz

from processmind.formatting import format_duration
from processmind.models import ProcessSpec, ProcessStep

# Палитра: те же цвета использует легенда в интерфейсе (processmind/ui/style.py).
COLORS = {
    "manual": {"fill": "#FEF3C7", "border": "#D97706", "label": "Ручная операция"},
    "automated": {"fill": "#D1FAE5", "border": "#059669", "label": "Автоматизирована"},
    "unknown": {"fill": "#F3F4F6", "border": "#9CA3AF", "label": "Тип выполнения не указан"},
    "proposed": {"fill": "#DBEAFE", "border": "#2563EB", "label": "Автоматизация предложена (модельная оценка)"},
}
EDGE_COLOR = "#6B7280"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#4B5563"
WRAP_WIDTH = 26
FONT = "Helvetica,Arial,sans-serif"


@dataclass(frozen=True)
class StepChange:
    """Принятое изменение операции (для TO-BE): длительность до/после и применённый паттерн."""

    step_id: str
    original_minutes: float | None
    new_minutes: float
    pattern_id: str | None = None


def step_kind(step: ProcessStep) -> str:
    return "unknown" if step.is_manual is None else ("manual" if step.is_manual else "automated")


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _wrapped(text: str) -> str:
    # BR ALIGN="LEFT" выравнивает по левому краю строку, которая стоит ПЕРЕД тегом, поэтому он нужен и после последней
    lines = textwrap.wrap(text, WRAP_WIDTH, break_long_words=False) or [text]
    return "".join(f'{_esc(line)}<BR ALIGN="LEFT"/>' for line in lines)


def _row(content: str, size: int = 10, color: str = MUTED_COLOR, bold: bool = False) -> str:
    inner = f"<B>{content}</B>" if bold else content
    return f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="{size}" COLOR="{color}">{inner}</FONT></TD></TR>'


def _label(number: int, step: ProcessStep, kind: str, change: StepChange | None) -> str:
    rows = [_row(_wrapped(f"{number}. {step.name}"), size=12, color=TEXT_COLOR, bold=True)]
    rows.append(_row(f"Исполнитель: {_esc(step.actor or 'не указан')}"))
    if change is not None:
        rows.append(_row(f"Время: {_esc(format_duration(change.original_minutes))} → {_esc(format_duration(change.new_minutes))}", color=COLORS['proposed']['border'], bold=True))
        rows.append(_row("АВТОМАТИЗАЦИЯ ПРЕДЛОЖЕНА" + (f" · {_esc(change.pattern_id)}" if change.pattern_id else ""), size=9, color=COLORS['proposed']['border'], bold=True))
    else:
        tag = {"manual": "ВРУЧНУЮ", "automated": "АВТОМАТИЧЕСКИ", "unknown": "ТИП НЕ УКАЗАН"}[kind]
        rows.append(
            '<TR><TD ALIGN="LEFT">'
            f'<FONT POINT-SIZE="10" COLOR="{MUTED_COLOR}">Время: {_esc(format_duration(step.duration_minutes))}</FONT>'
            f'<FONT POINT-SIZE="9" COLOR="{COLORS[kind]["border"]}"><B>&nbsp;&nbsp;·&nbsp;&nbsp;{tag}</B></FONT></TD></TR>'
        )
    return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">' + "".join(rows) + "</TABLE>>"


def _edges(spec: ProcessSpec) -> list[tuple[str, str]]:
    """Связи из next_steps; если их нет ни у одной операции — последовательно по порядку в списке."""
    if any(step.next_steps for step in spec.steps):
        return [(s.id, target) for s in spec.steps for target in s.next_steps]
    return [(a.id, b.id) for a, b in zip(spec.steps, spec.steps[1:])]


def _build(spec: ProcessSpec, changes: dict[str, StepChange], name: str) -> graphviz.Digraph:
    dot = graphviz.Digraph(name=name, format="svg")
    dot.attr(rankdir="TB", bgcolor="transparent", nodesep="0.35", ranksep="0.45", pad="0.2")
    dot.attr("node", shape="box", style="rounded,filled", fontname=FONT, margin="0.14,0.08")
    dot.attr("edge", color=EDGE_COLOR, arrowsize="0.8", penwidth="1.4")

    for number, step in enumerate(spec.steps, start=1):
        change = changes.get(step.id)
        kind = "proposed" if change is not None else step_kind(step)
        colors = COLORS[kind]
        dot.node(
            step.id,
            _label(number, step, step_kind(step), change),
            fillcolor=colors["fill"],
            color=colors["border"],
            penwidth="2.6" if change is not None else "1.4",
        )
    for source, target in _edges(spec):
        dot.edge(source, target)
    return dot


def build_as_is_diagram(spec: ProcessSpec) -> graphviz.Digraph:
    """AS-IS: последовательность, название, исполнитель, время; ручные операции — жёлтые."""
    return _build(spec, {}, "as_is")


def build_to_be_diagram(spec: ProcessSpec, changes: Sequence[StepChange]) -> graphviz.Digraph:
    """TO-BE: исходные операции и связи (`spec` — исходный процесс); изменённые операции выделены и подписаны «было → стало»."""
    known = {s.id for s in spec.steps}
    unknown = [c.step_id for c in changes if c.step_id not in known]
    if unknown:
        raise ValueError(f"В процессе нет операций: {', '.join(unknown)}")
    return _build(spec, {c.step_id: c for c in changes}, "to_be")
