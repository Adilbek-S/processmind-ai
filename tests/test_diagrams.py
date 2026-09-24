import graphviz

from processmind.models import ProcessSpec, ProcessStep
from processmind.visualization.diagrams import build_process_diagram


def test_build_process_diagram_sequential_edges():
    spec = ProcessSpec(
        process_id="p1",
        name="Процесс",
        steps=[
            ProcessStep(id="step-1", name="Шаг 1"),
            ProcessStep(id="step-2", name="Шаг 2"),
        ],
    )
    dot = build_process_diagram(spec)
    assert isinstance(dot, graphviz.Digraph)
    source = dot.source
    assert "step-1" in source
    assert "step-2" in source
    assert '"step-1" -> "step-2"' in source


def test_build_process_diagram_uses_explicit_next_steps():
    spec = ProcessSpec(
        process_id="p2",
        name="Процесс с ветвлением",
        steps=[
            ProcessStep(id="a", name="A", next_steps=["b", "c"]),
            ProcessStep(id="b", name="B"),
            ProcessStep(id="c", name="C"),
        ],
    )
    dot = build_process_diagram(spec)
    source = dot.source
    assert "a -> b" in source
    assert "a -> c" in source
