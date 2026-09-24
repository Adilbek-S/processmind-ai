"""Тесты Graphviz-схем AS-IS и TO-BE."""

import re
import xml.etree.ElementTree as ET

import graphviz
import pytest

from processmind.models import ProcessSpec, ProcessStep
from processmind.visualization.diagrams import COLORS, StepChange, build_as_is_diagram, build_to_be_diagram


def spec(*steps: ProcessStep, **kw) -> ProcessSpec:
    return ProcessSpec(process_id="p", name="Процесс", steps=list(steps), **kw)


SPEC = spec(
    ProcessStep(id="a", name="Принять заявку", actor="Менеджер", duration_minutes=30, is_manual=True),
    ProcessStep(id="b", name="Согласовать бюджет", actor="Финдиректор", duration_minutes=120, is_manual=None),
    ProcessStep(id="c", name="Оплатить счёт", actor="Бухгалтер", duration_minutes=60, is_manual=False),
    ProcessStep(id="d", name="Выбрать поставщика"),
)


def node_line(dot: graphviz.Digraph, node_id: str) -> str:
    return next(line for line in dot.source.splitlines() if line.strip().startswith(f"{node_id} ["))


def edges(dot: graphviz.Digraph) -> set[tuple[str, str]]:
    return set(re.findall(r"^\s*(\S+) -> (\S+)\s*$", dot.source, flags=re.MULTILINE))


def nodes(dot: graphviz.Digraph) -> set[str]:
    return set(re.findall(r"^\s*(\S+) \[label=", dot.source, flags=re.MULTILINE))


# ---------- AS-IS ----------


def test_as_is_is_a_graphviz_digraph_with_all_steps_in_order():
    dot = build_as_is_diagram(SPEC)
    assert isinstance(dot, graphviz.Digraph)
    assert nodes(dot) == {"a", "b", "c", "d"}
    assert edges(dot) == {("a", "b"), ("b", "c"), ("c", "d")}  # последовательность по порядку операций


def test_as_is_node_shows_number_name_actor_and_duration():
    line = node_line(build_as_is_diagram(SPEC), "a")
    for text in ("1. Принять заявку", "Исполнитель: Менеджер", "Время: 30 мин"):
        assert text in line


def test_as_is_highlights_manual_operations_by_color():
    dot = build_as_is_diagram(SPEC)
    assert COLORS["manual"]["fill"] in node_line(dot, "a")
    assert COLORS["unknown"]["fill"] in node_line(dot, "b")
    assert COLORS["automated"]["fill"] in node_line(dot, "c")
    assert COLORS["manual"]["fill"] not in node_line(dot, "c")
    assert "ВРУЧНУЮ" in node_line(dot, "a") and "АВТОМАТИЧЕСКИ" in node_line(dot, "c") and "ТИП НЕ УКАЗАН" in node_line(dot, "b")


def test_as_is_marks_missing_values_explicitly_instead_of_inventing_them():
    line = node_line(build_as_is_diagram(SPEC), "d")
    assert "Исполнитель: не указан" in line and "Время: не указано" in line


def test_as_is_long_durations_are_readable():
    dot = build_as_is_diagram(spec(ProcessStep(id="x", name="Долго", duration_minutes=4320), ProcessStep(id="y", name="Час", duration_minutes=90)))
    assert "Время: 3 дн" in node_line(dot, "x") and "Время: 1,5 ч" in node_line(dot, "y")


def test_explicit_links_are_used_when_present_including_branches():
    branching = spec(
        ProcessStep(id="a", name="A", next_steps=["b", "c"]),
        ProcessStep(id="b", name="B"),
        ProcessStep(id="c", name="C"),
    )
    assert edges(build_as_is_diagram(branching)) == {("a", "b"), ("a", "c")}


def test_user_text_is_html_escaped_in_labels():
    dot = build_as_is_diagram(spec(ProcessStep(id="x", name='Проверить <b>"данные"</b> & сверить', actor="A<B")))
    line = node_line(dot, "x")
    assert "&lt;b&gt;" in line and "&amp;" in line and "A&lt;B" in line and "<b>" not in line


def test_long_names_are_wrapped():
    line = node_line(build_as_is_diagram(spec(ProcessStep(id="x", name="Очень длинное название операции которое должно перенестись"))), "x")
    assert '<BR ALIGN="LEFT"/>' in line


@pytest.mark.parametrize("builder", [lambda: build_as_is_diagram(SPEC), lambda: build_to_be_diagram(SPEC, [StepChange("a", 30, 5, "AP-003")])])
def test_html_labels_are_well_formed(builder):
    """HTML-подобные метки Graphviz должны быть корректным XML, иначе схема не отрисуется в браузере."""
    dot = builder()
    labels = re.findall(r"label=<(<TABLE.*?</TABLE>)>", dot.source, flags=re.DOTALL)
    assert len(labels) == 4
    for label in labels:
        ET.fromstring(label.replace("&nbsp;", "\u00a0"))  # исключение = невалидная метка


# ---------- TO-BE ----------


def test_to_be_preserves_operations_and_links_exactly():
    as_is = build_as_is_diagram(SPEC)
    to_be = build_to_be_diagram(SPEC, [StepChange("a", 30, 5, "AP-003"), StepChange("b", 120, 20, None)])
    assert nodes(to_be) == nodes(as_is) and edges(to_be) == edges(as_is)  # маршруты не перестраиваются, шлюзы не добавляются


def test_to_be_highlights_changed_operations_with_before_after_and_pattern():
    dot = build_to_be_diagram(SPEC, [StepChange("a", 30, 5, "AP-003")])
    line = node_line(dot, "a")
    assert COLORS["proposed"]["fill"] in line
    assert "Время: 30 мин → 5 мин" in line and "АВТОМАТИЗАЦИЯ ПРЕДЛОЖЕНА" in line and "AP-003" in line
    assert "1. Принять заявку" in line and "Исполнитель: Менеджер" in line


def test_to_be_keeps_asis_look_for_unchanged_operations():
    as_is, to_be = build_as_is_diagram(SPEC), build_to_be_diagram(SPEC, [StepChange("a", 30, 5)])
    for node in ("b", "c", "d"):
        assert node_line(to_be, node) == node_line(as_is, node)
    assert node_line(to_be, "a") != node_line(as_is, "a")


def test_to_be_change_without_pattern_omits_pattern_badge():
    assert "AP-" not in node_line(build_to_be_diagram(SPEC, [StepChange("a", 30, 5)]), "a")


def test_to_be_shows_unknown_original_duration_explicitly():
    line = node_line(build_to_be_diagram(SPEC, [StepChange("d", None, 5)]), "d")
    assert "Время: не указано → 5 мин" in line


def test_to_be_rejects_changes_for_unknown_steps():
    with pytest.raises(ValueError, match="zzz"):
        build_to_be_diagram(SPEC, [StepChange("zzz", 1, 1)])
