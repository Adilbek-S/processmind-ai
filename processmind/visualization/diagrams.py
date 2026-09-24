"""Построение AS-IS / TO-BE диаграмм процесса через Graphviz."""

from __future__ import annotations

import graphviz

from processmind.models import ProcessSpec

MANUAL_COLOR = "#f4c542"
AUTOMATED_COLOR = "#7fd17f"
UNKNOWN_COLOR = "#d9d9d9"


def build_process_diagram(spec: ProcessSpec, title: str | None = None) -> graphviz.Digraph:
    """Строит граф шагов процесса. Ручные шаги — жёлтые, автоматизированные — зелёные, тип не указан — серые.

    Если у шага заданы `next_steps`, граф использует явные связи; иначе шаги
    соединяются последовательно в порядке их следования в списке.
    """
    dot = graphviz.Digraph(name=spec.process_id.replace("-", "_"), format="png")
    dot.attr(rankdir="LR", label=title or spec.name, fontsize="14", labelloc="t")

    for step in spec.steps:
        if step.is_manual is None:
            color = UNKNOWN_COLOR
        else:
            color = MANUAL_COLOR if step.is_manual else AUTOMATED_COLOR
        label = f"{step.name}\n({step.actor or 'н/д'})"
        dot.node(step.id, label, style="filled", fillcolor=color, shape="box")

    has_explicit_links = any(step.next_steps for step in spec.steps)
    if has_explicit_links:
        for step in spec.steps:
            for next_id in step.next_steps:
                dot.edge(step.id, next_id)
    else:
        for prev_step, next_step in zip(spec.steps, spec.steps[1:]):
            dot.edge(prev_step.id, next_step.id)

    return dot
