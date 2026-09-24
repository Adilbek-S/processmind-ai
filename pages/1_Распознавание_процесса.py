"""Страница «Распознавание процесса»: PDF/PNG/JPG -> ProcessSpec -> просмотр и правка -> подтверждение."""

import pandas as pd
import streamlit as st

from processmind.config import get_settings
from processmind.parsing.review import EXEC_OPTIONS, ReviewError, apply_edits, spec_to_rows
from processmind.parsing.schemas import ExtractionError
from processmind.parsing.service import IMAGE_EXTENSIONS, extract_process
from processmind.ui.state import CONFIRMED_KEY, DRAFT_FILE_KEY, DRAFT_KEY, NONCE_KEY
from processmind.ui.style import apply_style, legend, nav_link, page_header, section, stepper
from processmind.visualization.diagrams import build_as_is_diagram

st.set_page_config(page_title="Распознавание процесса — ProcessMind AI", page_icon="📄", layout="wide")
apply_style()
page_header(
    "Распознавание процесса",
    "PDF разбирается PyMuPDF и LLM, PNG/JPG — Vision-моделью. Значения, которых нет в документе, остаются пустыми: они не выдумываются.",
)
stepper_slot = st.empty()
stepper(container=stepper_slot)

if not get_settings().openai_api_key:
    st.warning("OPENAI_API_KEY не задан в .env — распознавание недоступно.")

# ---------- 1. Документ ----------
section(1, "Загрузка документа")
uploaded = st.file_uploader("PDF, PNG или JPG с описанием процесса", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    if uploaded.name.lower().rsplit(".", 1)[-1] in {e.lstrip(".") for e in IMAGE_EXTENSIONS}:
        st.image(file_bytes, caption=uploaded.name, width=520)

    if st.button("Распознать процесс", type="primary"):
        try:
            with st.spinner("Распознаю процесс…"):
                result = extract_process(uploaded.name, file_bytes)
        except ExtractionError as exc:
            st.session_state.pop(DRAFT_KEY, None)
            st.error(f"Не удалось распознать процесс: {exc}")
        else:
            st.session_state[DRAFT_KEY] = result
            st.session_state[DRAFT_FILE_KEY] = uploaded.file_id
            st.session_state[NONCE_KEY] = st.session_state.get(NONCE_KEY, 0) + 1
            st.session_state.pop(CONFIRMED_KEY, None)  # новый черновик — прежнее подтверждение не действует
            stepper(container=stepper_slot)

draft = st.session_state.get(DRAFT_KEY)
if draft is None or uploaded is None or st.session_state.get(DRAFT_FILE_KEY) != uploaded.file_id:
    if uploaded is None:
        st.info("Загрузите файл и нажмите «Распознать процесс».")
    st.stop()

spec = draft.spec
nonce = st.session_state[NONCE_KEY]

# ---------- 2. Операции ----------
section(2, "Операции: просмотр и корректировка")
st.success(f"Распознано операций: {len(spec.steps)}")
if draft.warnings:
    with st.expander(f"Замечания распознавания ({len(draft.warnings)}) — проверьте их в таблице"):
        for warning in draft.warnings:
            st.markdown(f"- {warning}")

col_name, col_goal = st.columns(2)
name = col_name.text_input("Название процесса", value=spec.name, key=f"name_{nonce}")
goal = col_goal.text_input("Цель процесса", value=spec.goal or "", placeholder="не указана в документе", key=f"goal_{nonce}")
st.markdown("**Участники:** " + (", ".join(spec.actors) if spec.actors else "_не указаны в документе_"))

st.caption("Исправьте название, участника, длительность или тип выполнения. Пустая ячейка = значение не указано.")
edited = st.data_editor(
    pd.DataFrame(spec_to_rows(spec)),
    key=f"editor_{nonce}",
    hide_index=True,
    num_rows="fixed",
    width="stretch",
    disabled=["№", "Следующие"],
    column_config={
        "Операция": st.column_config.TextColumn(required=True),
        "Участник": st.column_config.TextColumn(help="Пусто — участник не указан"),
        "Длительность, мин": st.column_config.NumberColumn(min_value=0, help="Пусто — длительность не указана"),
        "Тип выполнения": st.column_config.SelectboxColumn(options=EXEC_OPTIONS, required=True),
        "Следующие": st.column_config.TextColumn(help="Номера операций, следующих за этой"),
    },
)

candidate = None
try:
    candidate = apply_edits(
        spec.model_copy(update={"name": name.strip() or spec.name, "goal": goal.strip() or None}),
        edited.astype(object).where(edited.notna(), None).to_dict("records"),
    )
except ReviewError as exc:
    st.error(str(exc))

if candidate is not None:
    st.markdown('<div class="pm-graph-title">Схема процесса (AS-IS) — обновляется по мере правок</div>', unsafe_allow_html=True)
    legend(["manual", "automated", "unknown"])
    st.graphviz_chart(build_as_is_diagram(candidate), width="content")

# ---------- 3. Подтверждение ----------
section(3, "Подтверждение")
confirmed = st.session_state.get(CONFIRMED_KEY)
if st.button("Подтвердить результат", type="primary", disabled=candidate is None):
    st.session_state[CONFIRMED_KEY] = candidate
    confirmed = candidate
    stepper(container=stepper_slot)

if confirmed is not None:
    if candidate is not None and candidate != confirmed:
        st.info("Есть изменения, которые ещё не подтверждены. Нажмите «Подтвердить результат».")
    st.success(f"Процесс «{confirmed.name}» подтверждён ({len(confirmed.steps)} операций) и готов к анализу.")
    nav_link("pages/2_Анализ_процесса.py", "Перейти к AI-анализу →", "🔍")
    with st.expander("ProcessSpec (JSON)"):
        st.json(confirmed.model_dump())
