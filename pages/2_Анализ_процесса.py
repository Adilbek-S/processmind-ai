"""Страница «Анализ процесса»: главный LangGraph workflow с human-in-the-loop, TO-BE и итоговый отчёт.

validate_process -> calculate_metrics -> retrieve_automation_patterns -> generate_recommendations
-> validate_recommendations -> [пауза: выбор пользователя] -> simulate_automation -> generate_report
"""

import html

import streamlit as st

from processmind.analysis.recommendations import build_problem_signals
from processmind.analysis.report import AnalysisReport, metrics_comparison, report_to_markdown
from processmind.formatting import format_duration
from processmind.mcp.client import McpClientError
from processmind.ui.components import (
    changes_from_report,
    comparison_table_html,
    initial_metric_items,
    metric_cards,
    recommendations_table_html,
)
from processmind.ui.state import CONFIRMED_KEY, RUN_KEY, SPEC_KEY
from processmind.ui.style import apply_style, legend, nav_link, notice, page_header, section, stepper
from processmind.visualization.diagrams import build_as_is_diagram, build_to_be_diagram
from processmind.workflow.graph import (
    AnalysisRun,
    ApprovalStateError,
    build_workflow,
    create_default_deps,
    peek_state,
    resume_analysis,
    start_analysis,
)

st.set_page_config(page_title="AI-анализ — ProcessMind AI", page_icon="🔍", layout="wide")
apply_style()
page_header(
    "AI-анализ процесса",
    "Проверка и метрики — MCP-сервер · паттерны — RAG · рекомендации — LLM по методике Skill «process-analysis» · оценка эффекта — по вашему подтверждению.",
)
stepper_slot = st.empty()
stepper(container=stepper_slot)

_PROBLEM_TITLES = {
    "manual_step": "Ручные операции",
    "unknown_type": "Тип выполнения не указан",
    "missing_duration": "Длительность не указана",
    "long_step": "Самые длительные операции",
    "pain_point": "Указанные проблемы",
}


@st.cache_resource(show_spinner="Запускаю MCP-сервер…")
def get_workflow():
    """Один процесс MCP-сервера и один граф на всё приложение (состояние запусков хранится в checkpointer)."""
    return build_workflow(create_default_deps())


confirmed = st.session_state.get(CONFIRMED_KEY)
if confirmed is None:
    st.info("Сначала распознайте и подтвердите процесс.")
    nav_link("pages/1_Распознавание_процесса.py", "Перейти к распознаванию →", "📄")
    st.stop()

try:
    graph = get_workflow()
except McpClientError as exc:
    st.error(f"Не удалось запустить MCP-сервер: {exc}")
    st.stop()

# Подтверждённый процесс изменился — прежний запуск неактуален.
spec_json = confirmed.model_dump_json()
if st.session_state.get(SPEC_KEY) != spec_json:
    st.session_state[SPEC_KEY] = spec_json
    st.session_state.pop(RUN_KEY, None)
    stepper(container=stepper_slot)

# ---------- запуск ----------
head, controls = st.columns([3, 2])
head.markdown(f"**Процесс:** {html.escape(confirmed.name)} · операций: {len(confirmed.steps)}", unsafe_allow_html=True)
runs = controls.number_input(
    "Число запусков процесса в месяц",
    min_value=0,
    step=1,
    value=None,
    placeholder="не задано — месячные показатели не считаются",
    help="Нужно только для условной месячной трудоёмкости и экономии; необязательно.",
)
if st.button("Запустить AI-анализ", type="primary"):
    try:
        with st.spinner("Проверяю процесс, считаю метрики, ищу паттерны, формирую рекомендации…"):
            st.session_state[RUN_KEY] = start_analysis(graph, confirmed, int(runs) if runs is not None else None)
        stepper(container=stepper_slot)
    except Exception as exc:  # непредвиденный сбой: показываем, а не роняем страницу
        st.session_state.pop(RUN_KEY, None)
        st.error(f"Анализ не выполнен: {exc}")

run: AnalysisRun | None = st.session_state.get(RUN_KEY)
if run is None:
    section(1, "Схема процесса (AS-IS)")
    legend(["manual", "automated", "unknown"])
    st.graphviz_chart(build_as_is_diagram(confirmed), width="content")
    st.stop()


def render_problems(metrics: dict) -> None:
    """Выявленные проблемы — факты из данных процесса (те же сигналы получает LLM)."""
    names = {s.id: s.name for s in confirmed.steps}
    signals = build_problem_signals(confirmed, metrics)
    for kind, title in _PROBLEM_TITLES.items():
        items = [x for x in signals if x["kind"] == kind]
        if items:
            with st.expander(f"{title} ({len(items)})", expanded=kind in ("manual_step", "long_step")):
                for x in items:
                    st.markdown(f"- **{names.get(x['step_id'], x['step_id'])}** — {x['text']}")


# ---------- пауза: метрики, рекомендации, выбор ----------
if run.status == "awaiting_approval":
    request = run.approval_request
    state = peek_state(graph, run.thread_id)
    metrics = state["metrics"]

    section(1, "Исходные метрики (AS-IS)")
    for warning in state.get("warnings", []):
        st.warning(warning)
    metric_cards(initial_metric_items(metrics))
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="pm-graph-title">Схема AS-IS</div>', unsafe_allow_html=True)
        legend(["manual", "automated", "unknown"])
        st.graphviz_chart(build_as_is_diagram(confirmed), width="content")
    with right:
        st.markdown('<div class="pm-graph-title">Выявленные проблемы</div>', unsafe_allow_html=True)
        render_problems(metrics)
        with st.expander(f"Найденные паттерны автоматизации ({len(state.get('patterns', []))})"):
            for p in state.get("patterns", []):
                st.markdown(f"- **{p['pattern_id']}** — {p['title']} · `{p['source']}` · сходство {p['score']:.3f}")

    section(2, "Рекомендации по автоматизации")
    st.markdown(recommendations_table_html(request["recommendations"]), unsafe_allow_html=True)

    section(3, "Выбор рекомендаций для TO-BE")
    st.info("Workflow **приостановлен** и ждёт вашего решения: симуляция выполнится только после подтверждения. Длительность после автоматизации — ваше допущение.")
    if request.get("error"):
        st.error(f"Решение не принято: {request['error']}")

    selections = []
    for rec in request["recommendations"]:
        rid = rec["rec_id"]
        can_simulate = rec["original_duration_minutes"] is not None
        with st.container(border=True):
            pick, title, minutes_col = st.columns([1.3, 4, 2.4])
            chosen = pick.checkbox("В TO-BE", key=f"sel_{run.thread_id}_{rid}", disabled=not can_simulate)
            pattern = f" · паттерн **{rec['pattern_id']}**" if rec["pattern_id"] else " · _без паттерна_"
            title.markdown(f"**{rid}. {html.escape(rec['step_name'])}**{pattern}  \n{html.escape(rec['solution'])}", unsafe_allow_html=True)
            if can_simulate:
                minutes = minutes_col.number_input(
                    f"После автоматизации, мин (было {rec['original_duration_minutes']:g})",
                    min_value=0.0,
                    value=rec["new_duration_minutes"],
                    placeholder="ваше допущение",
                    key=f"dur_{run.thread_id}_{rid}",
                    help=rec["duration_basis"] or "LLM не предложила длительность: задайте своё допущение.",
                )
                if chosen:
                    selections.append({"rec_id": rid, "new_duration_minutes": minutes})
            else:
                minutes_col.caption("Исходная длительность не указана — эффект смоделировать нельзя.")
            with st.expander("Обоснование, условия применимости, ограничения"):
                st.caption(f"Факты об операции (из процесса): {rec['process_facts']}")
                st.markdown(f"**Проблема:** {rec['problem']}")
                st.markdown(f"**Обоснование:** {rec['rationale']}")
                if rec["conditions_confirmed"]:
                    st.markdown("**Подтверждено данными:** " + "; ".join(rec["conditions_confirmed"]))
                if rec["conditions_unverified"]:
                    st.markdown("**Не подтверждено данными (проверить):** " + "; ".join(rec["conditions_unverified"]))
                if rec["duration_basis"]:
                    st.markdown(f"**Основание допущения LLM о длительности:** {rec['duration_basis']}")
                for w in rec["warnings"]:
                    st.caption(f"⚠️ {w}")

    missing = [s["rec_id"] for s in selections if s["new_duration_minutes"] is None]
    if missing:
        st.warning(f"Укажите допущение о длительности после автоматизации для: {', '.join(missing)}")
    if st.button("Сформировать TO-BE и отчёт", type="primary", disabled=bool(missing)):
        try:
            with st.spinner("Продолжаю: симуляция, TO-BE, отчёт…"):
                st.session_state[RUN_KEY] = resume_analysis(graph, run.thread_id, {"selections": selections})
            st.rerun()
        except ApprovalStateError as exc:
            st.session_state.pop(RUN_KEY, None)
            st.error(str(exc))
    st.stop()


# ---------- итог: TO-BE и отчёт ----------
report: AnalysisReport = run.report

if report.status == "invalid":
    st.error("Процесс некорректен. Исправьте ошибки на странице распознавания и запустите анализ снова:")
    for error in report.errors:
        st.markdown(f"- {error}")
elif report.status == "failed":
    st.error("Анализ не завершён из-за сбоя:")
    for error in report.errors:
        st.markdown(f"- {error}")
elif report.status == "no_recommendations":
    st.info("По методике не найдено обоснованных рекомендаций по автоматизации. Причины отклонения — в предупреждениях отчёта.")
elif report.status == "no_selection":
    st.info("Рекомендации не выбраны — TO-BE не формировался.")

if report.metrics:
    section(1, "Исходные метрики (AS-IS)")
    metric_cards(initial_metric_items(report.metrics))

if report.status == "completed":
    notice(report.model_notice)
    section(2, "Сравнение AS-IS и TO-BE")
    st.markdown(comparison_table_html(metrics_comparison(report.metrics, report.tobe_metrics)), unsafe_allow_html=True)

    section(3, "Схемы процесса")
    col_as_is, col_to_be = st.columns(2)
    with col_as_is:
        st.markdown('<div class="pm-graph-title">AS-IS — исходный процесс</div>', unsafe_allow_html=True)
        legend(["manual", "automated", "unknown"])
        st.graphviz_chart(build_as_is_diagram(report.process), width="content")
    with col_to_be:
        st.markdown('<div class="pm-graph-title">TO-BE — модельная оценка (порядок операций сохранён)</div>', unsafe_allow_html=True)
        legend(["proposed", "manual", "automated", "unknown"])
        st.graphviz_chart(build_to_be_diagram(report.process, changes_from_report(report)), width="content")

    st.markdown('<div class="pm-graph-title">Применённые изменения</div>', unsafe_allow_html=True)
    by_id = {r.rec_id: r for r in report.recommendations}
    for a in report.approved:
        rec = by_id[a.rec_id]
        who = "пользователя" if a.duration_source == "user" else "LLM"
        st.markdown(
            f"- **{html.escape(rec.step_name)}**: {format_duration(rec.original_duration_minutes)} → **{format_duration(a.new_duration_minutes)}** "
            f"(допущение {who}) · паттерн {rec.pattern_id or 'нет'}"
        )

if report.recommendations:
    section(4, "Рекомендации и решения")
    st.markdown(recommendations_table_html(report.recommendations, show_decision=True), unsafe_allow_html=True)

section(5, "Итоговый отчёт")
markdown = report_to_markdown(report)
st.download_button("Скачать отчёт (Markdown)", markdown, file_name="analysis_report.md", mime="text/markdown", type="primary")
tab_report, tab_json = st.tabs(["Предпросмотр отчёта", "AnalysisReport (JSON)"])
with tab_report:
    st.markdown(markdown)
with tab_json:
    st.download_button("Скачать JSON", report.model_dump_json(indent=2), file_name="analysis_report.json", mime="application/json")
    st.json(report.model_dump(mode="json"), expanded=False)
