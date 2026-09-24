"""Юнит-тесты программной валидации рекомендаций, сигналов проблем, Skill и решения пользователя."""

import pytest

from processmind.analysis.recommendations import (
    LLMRecommendationSet,
    build_llm_context,
    build_messages,
    build_problem_signals,
    validate_recommendations,
)
from processmind.analysis.report import AnalysisReport, build_to_be, report_to_markdown
from processmind.analysis.skill import SKILL_PATH, SkillError, load_skill
from processmind.models import ProcessSpec
from processmind.workflow.approval import ApprovalError, parse_decision
from tests.helpers_workflow import PROCESS, rec

SPEC = ProcessSpec.model_validate(PROCESS)
PATTERNS = {"AP-001": {"title": "Маршрутизация", "source": "AP-001.md"}, "AP-002": {"title": "Согласование", "source": "AP-002.md"}}


def check(*recs):
    return validate_recommendations(LLMRecommendationSet(recommendations=list(recs), notes=[]), SPEC, PATTERNS)


# ---------- validate_recommendations ----------


def test_valid_recommendation_gets_id_pattern_link_and_simulatable_flag():
    outcome = check(rec("s1", "AP-001", 5, "допущение"))
    (r,) = outcome.accepted
    assert (r.rec_id, r.step_id, r.step_name, r.kb_based) == ("R1", "s1", "Принять заявку", True)
    assert (r.pattern_id, r.pattern_title, r.pattern_source) == ("AP-001", "Маршрутизация", "AP-001.md")
    assert (r.original_duration_minutes, r.new_duration_minutes, r.simulatable) == (30, 5, True)


def test_recommendation_without_pattern_is_allowed_but_not_kb_based():
    (r,) = check(rec("s1", None, 5)).accepted
    assert r.kb_based is False and r.pattern_id is None and r.pattern_title is None


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_blank_pattern_id_means_no_kb_reference(blank):
    (r,) = check(rec("s1", blank)).accepted
    assert r.pattern_id is None and not r.kb_based


def test_nonexistent_step_id_is_rejected():
    outcome = check(rec("s42", "AP-001"))
    assert outcome.accepted == [] and "нет в процессе" in outcome.rejected[0].reasons[0]


def test_reference_to_pattern_that_was_not_retrieved_is_rejected():
    outcome = check(rec("s1", "AP-999"))
    assert outcome.accepted == [] and "AP-999" in outcome.rejected[0].reasons[0]


def test_already_automated_step_is_rejected():
    assert "уже автоматизирована" in check(rec("s3", "AP-001")).rejected[0].reasons[0]


def test_unknown_type_step_without_explicit_signs_is_not_recommended():
    spec = ProcessSpec.model_validate({**PROCESS, "steps": [{"id": "a", "name": "Согласовать", "duration_minutes": 100}]})
    outcome = validate_recommendations(LLMRecommendationSet(recommendations=[rec("a", "AP-001", 10)], notes=[]), spec, PATTERNS)
    assert outcome.accepted == [] and "нет явных признаков ручной работы" in outcome.rejected[0].reasons[0]


@pytest.mark.parametrize("extra", [{"pain_points": ["ручной ввод"]}, {"description": "Сотрудник вручную сверяет данные"}])
def test_unknown_type_step_with_explicit_sign_is_allowed(extra):
    spec = ProcessSpec.model_validate({**PROCESS, "steps": [{"id": "a", "name": "Согласовать", "duration_minutes": 100, **extra}]})
    assert validate_recommendations(LLMRecommendationSet(recommendations=[rec("a", "AP-001", 10)], notes=[]), spec, PATTERNS).accepted


def test_plain_description_is_not_a_sign_of_manual_work():
    """Описание, скопированное из документа («Финансовый директор согласовывает бюджет»), ручную работу не доказывает."""
    spec = ProcessSpec.model_validate({**PROCESS, "steps": [{"id": "a", "name": "Согласовать", "duration_minutes": 100, "description": "Финансовый директор согласовывает бюджет закупки (2 часа)."}]})
    outcome = validate_recommendations(LLMRecommendationSet(recommendations=[rec("a", "AP-001", 10)], notes=[]), spec, PATTERNS)
    assert outcome.accepted == [] and "нет явных признаков" in outcome.rejected[0].reasons[0]


def test_process_facts_come_from_the_process_not_from_the_llm():
    outcome = check(rec("s1", "AP-001", 5), rec("s4", None))
    assert outcome.accepted[0].process_facts == "Исполнитель: Менеджер; тип выполнения: ручной; длительность: 30 мин"
    assert outcome.accepted[1].process_facts == "Исполнитель: Менеджер; тип выполнения: ручной; длительность: не указана"


def test_only_first_recommendation_per_step_is_kept():
    outcome = check(rec("s1", "AP-001"), rec("s1", "AP-002"))
    assert [r.pattern_id for r in outcome.accepted] == ["AP-001"] and "уже есть рекомендация" in outcome.rejected[0].reasons[0]


@pytest.mark.parametrize("effect", ["Сократит время на 50%", "Быстрее в 3 раза", "Экономия 30 %", "быстрее в 2,5 раза"])
def test_quantitative_effect_claims_are_rejected(effect):
    assert "количественное утверждение" in check(rec("s1", "AP-001", expected_effect=effect)).rejected[0].reasons[0]


def test_qualitative_effect_with_digits_in_words_is_fine():
    assert check(rec("s1", "AP-001", expected_effect="Заявка сразу попадает к исполнителю с 1-го шага")).accepted


@pytest.mark.parametrize("field", ["problem", "solution", "rationale", "expected_effect"])
def test_blank_text_fields_are_rejected(field):
    assert f"пустое поле {field}" in check(rec("s1", "AP-001", **{field: "  "})).rejected[0].reasons[0]


def test_pattern_reference_requires_checked_applicability_conditions():
    outcome = check(rec("s1", "AP-001", conditions_confirmed=[], conditions_unverified=["  "]))
    assert outcome.accepted == [] and "не проверены условия применимости" in outcome.rejected[0].reasons[0]
    (ok,) = check(rec("s1", "AP-001", conditions_confirmed=[], conditions_unverified=["нет данных о системе"])).accepted
    assert ok.conditions_unverified == ["нет данных о системе"]  # достаточно честно назвать непроверенное


def test_condition_confirmed_by_absence_of_data_is_moved_to_unverified():
    (r,) = check(
        rec(
            "s1",
            "AP-001",
            conditions_confirmed=["Условие: есть перечень полей → Факт: нет данных о перечне", "Условие: заявки повторяются → Факт: операция ручная"],
            conditions_unverified=["Доступ к системе"],
        )
    ).accepted
    assert r.conditions_confirmed == ["Условие: заявки повторяются → Факт: операция ручная"]
    assert r.conditions_unverified == ["Доступ к системе", "Условие: есть перечень полей → Факт: нет данных о перечне"]
    assert any("перенесены" in w for w in r.warnings)


def test_rec_ids_are_sequential_over_accepted_only():
    outcome = check(rec("s1", "AP-001"), rec("nope", None), rec("s2", "AP-001"))
    assert [r.rec_id for r in outcome.accepted] == ["R1", "R2"]


# длительность после автоматизации: отбрасывается (рекомендация остаётся), а не выдумывается

def test_duration_without_basis_is_dropped_not_kept():
    (r,) = check(rec("s1", "AP-001", 5, basis="  ")).accepted
    assert r.new_duration_minutes is None and r.duration_basis is None and not r.simulatable
    assert "не указано основание" in r.warnings[0]


def test_duration_dropped_when_original_is_unknown():
    (r,) = check(rec("s4", "AP-001", 5, "допущение")).accepted  # у s4 нет длительности
    assert r.new_duration_minutes is None and r.original_duration_minutes is None and not r.simulatable
    assert "не указана исходная" in r.warnings[0]


def test_duration_larger_than_original_is_dropped():
    (r,) = check(rec("s1", "AP-001", 45, "допущение")).accepted  # исходная 30
    assert r.new_duration_minutes is None and "больше исходной" in r.warnings[0]


def test_negative_duration_rejects_recommendation():
    assert check(rec("s1", "AP-001", -1, "допущение")).accepted == []


def test_basis_without_duration_is_cleared():
    (r,) = check(rec("s1", "AP-001", None, basis="что-то")).accepted
    assert r.duration_basis is None


# ---------- сигналы проблем и контекст ----------


def test_problem_signals_contain_only_facts_from_data():
    metrics = {"total_time_minutes": 450}
    signals = {(s["step_id"], s["kind"]) for s in build_problem_signals(SPEC, metrics)}
    assert {("s1", "manual_step"), ("s4", "manual_step"), ("s4", "missing_duration"), ("s5", "unknown_type"), ("s5", "pain_point")} <= signals
    assert not any(step == "s3" and kind in ("manual_step", "unknown_type") for step, kind in signals)
    assert ("s5", "long_step") in signals and ("s2", "long_step") in signals


def test_pain_points_become_signals():
    spec = ProcessSpec.model_validate({**PROCESS, "steps": [{**PROCESS["steps"][0], "pain_points": ["дублирование данных"]}]})
    assert {"step_id": "s1", "kind": "pain_point", "text": "дублирование данных"} in build_problem_signals(spec, {})


def test_messages_put_skill_body_in_system_prompt():
    skill = load_skill()
    context = build_llm_context(SPEC, {"warnings": []}, {"total_time_minutes": 450}, [])
    system, user = build_messages(skill, context)
    assert system == {"role": "system", "content": skill.body} and user["role"] == "user" and "Согласование закупки" in user["content"]


# ---------- Skill ----------


def test_real_skill_file_has_required_sections_and_metadata():
    skill = load_skill()
    assert skill.name == "process-analysis" and skill.description and len(skill.sha256) == 64
    assert SKILL_PATH.as_posix().endswith(".claude/skills/process-analysis/SKILL.md")
    for section in (
        "Когда применять", "Методика анализа", "Правила выявления ручных операций", "Правила выбора паттернов автоматизации",
        "Требования к обоснованию рекомендации", "Запрет на выдумывание фактов", "Правила представления модельных оценок",
    ):
        assert section in skill.body, section


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("без frontmatter", "нет frontmatter"),
        ("---\nname: x\ndescription: y\n", "не закрыт"),
        ("---\ndescription: y\n---\nтело\n", "«name»"),
        ("---\nname: x\n---\nтело\n", "«description»"),
        ("---\nname: x\ndescription: y\n---\n   \n", "пустая методика"),
    ],
)
def test_malformed_skill_rejected(tmp_path, text, message):
    path = tmp_path / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SkillError, match=message):
        load_skill(path)


def test_missing_skill_file_rejected(tmp_path):
    with pytest.raises(SkillError, match="Не удалось прочитать"):
        load_skill(tmp_path / "nope.md")


# ---------- решение пользователя ----------

RECS = [
    {"rec_id": "R1", "step_id": "s1", "original_duration_minutes": 30, "new_duration_minutes": 5},
    {"rec_id": "R2", "step_id": "s2", "original_duration_minutes": 120, "new_duration_minutes": None},
    {"rec_id": "R3", "step_id": "s4", "original_duration_minutes": None, "new_duration_minutes": None},
]


def test_parse_decision_uses_llm_duration_unless_user_overrides():
    approved = parse_decision({"selections": [{"rec_id": "R1"}, {"rec_id": "R2", "new_duration_minutes": 10}]}, RECS)
    assert [(a.rec_id, a.new_duration_minutes, a.duration_source) for a in approved] == [("R1", 5, "llm"), ("R2", 10, "user")]


def test_user_value_equal_to_llm_value_counts_as_llm():
    (a,) = parse_decision({"selections": [{"rec_id": "R1", "new_duration_minutes": 5}]}, RECS)
    assert a.duration_source == "llm"


def test_empty_decision_selects_nothing():
    assert parse_decision({"selections": []}, RECS) == [] and parse_decision({}, RECS) == []


@pytest.mark.parametrize(
    "decision",
    [
        {"selections": [{"rec_id": "R2"}]},  # у R2 нет длительности от LLM и пользователь её не задал
        {"selections": [{"rec_id": "R3", "new_duration_minutes": 1}]},  # нет исходной длительности
        {"selections": [{"rec_id": "R9"}]},
        {"selections": [{"rec_id": "R1"}, {"rec_id": "R1"}]},
        {"selections": [{"rec_id": "R1", "new_duration_minutes": float("nan")}]},
        {"selections": [{"rec_id": "R1", "new_duration_minutes": -1}]},
        "R1",
        None,
    ],
)
def test_invalid_decisions_raise(decision):
    with pytest.raises(ApprovalError):
        parse_decision(decision, RECS)


# ---------- отчёт ----------


def test_to_be_changes_only_approved_steps():
    from processmind.analysis.report import ApprovedChange

    tobe = build_to_be(SPEC, [ApprovedChange(rec_id="R1", step_id="s1", new_duration_minutes=5, duration_source="llm")])
    steps = {s.id: s for s in tobe.steps}
    assert steps["s1"].is_manual is False and steps["s1"].duration_minutes == 5
    assert steps["s2"] == next(s for s in SPEC.steps if s.id == "s2") and tobe.process_id == "demo-tobe"
    assert SPEC.steps[0].is_manual is True  # исходный процесс не изменён


def test_markdown_report_states_model_estimate_and_missing_values():
    report = AnalysisReport(process_id="p", process_name="П", status="no_selection", errors=["e"], warnings=["w"])
    text = report_to_markdown(report)
    assert "no_selection" in text and "- e" in text and "- w" in text
