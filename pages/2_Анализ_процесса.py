"""Страница «Анализ процесса»: загрузка описания и запуск LangGraph-пайплайна."""

import streamlit as st

from processmind.parsing.document_parser import DocumentParser
from processmind.visualization.diagrams import build_process_diagram
from processmind.workflow.graph import build_workflow

st.set_page_config(page_title="Анализ процесса — ProcessMind AI", page_icon="🔍", layout="wide")
st.title("🔍 Анализ бизнес-процесса")

st.caption(
    "Пайплайн: анализ ProcessSpec → черновик TO-BE → оценка качества. Для текстового ввода без "
    "предварительного распознавания шаги извлекаются построчно (без LLM)."
)

confirmed_spec = st.session_state.get("confirmed_process_spec")
raw_text = ""

if confirmed_spec is not None:
    st.success(
        f"Используется подтверждённый процесс «{confirmed_spec.name}» "
        f"({len(confirmed_spec.steps)} операций) со страницы «Распознавание процесса»."
    )
    if st.checkbox("Проанализировать другой текст вместо него"):
        confirmed_spec = None

if confirmed_spec is None:
    st.info(
        "Чтобы распознать процесс из PDF или изображения, используйте страницу "
        "«Распознавание процесса». Здесь можно загрузить текстовый файл или ввести шаги построчно."
    )
    uploaded = st.file_uploader("Загрузите текстовое описание процесса", type=["txt"])
    if uploaded is not None:
        raw_text = DocumentParser().parse(uploaded.name, uploaded.read())
        st.text_area("Извлечённый текст", raw_text, height=200)
    else:
        raw_text = st.text_area(
            "...или вставьте текстовое описание процесса вручную (по одному шагу на строку)",
            placeholder=(
                "1. Клиент оставляет заявку на сайте\n"
                "2. Менеджер вручную проверяет данные клиента в CRM\n"
                "3. Менеджер согласовывает условия по телефону\n"
                "4. Бухгалтер вручную выставляет счёт"
            ),
            height=200,
        )

if st.button("Запустить анализ", type="primary", disabled=confirmed_spec is None and not raw_text.strip()):
    workflow = build_workflow()
    initial_state = {"raw_text": raw_text, "errors": []}
    if confirmed_spec is not None:
        initial_state["process_spec"] = confirmed_spec
    result = workflow.invoke(initial_state)

    process_spec = result["process_spec"]
    analysis = result["analysis"]
    tobe_spec = result["tobe_spec"]
    evaluation = result["evaluation"]

    st.success("Пайплайн LangGraph выполнен: extract_process → analyze → generate_tobe → evaluate")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### AS-IS")
        st.graphviz_chart(build_process_diagram(process_spec, title="AS-IS"))
    with col2:
        st.markdown("#### TO-BE (черновик)")
        st.graphviz_chart(build_process_diagram(tobe_spec, title="TO-BE"))

    st.markdown("#### Возможности автоматизации")
    if analysis.opportunities:
        for opp in analysis.opportunities:
            st.markdown(f"- **{opp.step_name}** (score={opp.score}): {', '.join(opp.reasons)}")
    else:
        st.info("Возможностей автоматизации не найдено.")

    with st.expander("ProcessSpec — AS-IS (JSON)"):
        st.json(process_spec.model_dump())
    with st.expander("ProcessSpec — TO-BE (JSON)"):
        st.json(tobe_spec.model_dump())
    with st.expander("Evaluation"):
        st.json(evaluation)
