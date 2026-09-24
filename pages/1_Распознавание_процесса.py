"""Страница «Распознавание процесса»: PDF/PNG/JPG -> ProcessSpec -> правка -> подтверждение."""

import pandas as pd
import streamlit as st

from processmind.config import get_settings
from processmind.parsing.review import EXEC_OPTIONS, ReviewError, apply_edits, spec_to_rows
from processmind.parsing.schemas import ExtractionError
from processmind.parsing.service import IMAGE_EXTENSIONS, extract_process

DRAFT_KEY = "extraction_draft"  # ExtractionResult последнего распознавания
DRAFT_FILE_KEY = "extraction_draft_file"  # id файла, из которого получен черновик
NONCE_KEY = "extraction_nonce"  # меняется при каждом распознавании и сбрасывает состояние редактора
CONFIRMED_KEY = "confirmed_process_spec"  # подтверждённый ProcessSpec для дальнейшего анализа

st.set_page_config(page_title="Распознавание процесса — ProcessMind AI", page_icon="📄", layout="wide")
st.title("📄 Распознавание бизнес-процесса")
st.caption(
    "PDF — текст извлекается PyMuPDF и разбирается LLM. PNG/JPG — схему распознаёт GPT-4o-mini Vision. "
    "Результат всегда приводится к ProcessSpec и проверяется Pydantic. "
    "Значения, которых нет в документе, остаются пустыми — они не выдумываются."
)

if not get_settings().openai_api_key:
    st.warning("OPENAI_API_KEY не задан в .env — распознавание будет недоступно.")

uploaded = st.file_uploader("Загрузите документ с описанием процесса", type=["pdf", "png", "jpg", "jpeg"])

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    if uploaded.name.lower().rsplit(".", 1)[-1] in {e.lstrip(".") for e in IMAGE_EXTENSIONS}:
        st.image(file_bytes, caption=uploaded.name, width=480)

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

draft = st.session_state.get(DRAFT_KEY)
if draft is None or uploaded is None or st.session_state.get(DRAFT_FILE_KEY) != uploaded.file_id:
    if uploaded is None:
        st.info("Загрузите файл и нажмите «Распознать процесс».")
    st.stop()

spec = draft.spec
nonce = st.session_state[NONCE_KEY]

st.success(f"Распознано операций: {len(spec.steps)}")
for warning in draft.warnings:
    st.warning(warning)

col_name, col_goal = st.columns(2)
name = col_name.text_input("Название процесса", value=spec.name, key=f"name_{nonce}")
goal = col_goal.text_input(
    "Цель процесса", value=spec.goal or "", placeholder="не указана в документе", key=f"goal_{nonce}"
)
st.markdown("**Участники:** " + (", ".join(spec.actors) if spec.actors else "_не указаны в документе_"))

st.markdown("#### Операции")
st.caption("Исправьте название, участника, длительность или тип выполнения. Пустая ячейка = значение не указано.")
edited = st.data_editor(
    pd.DataFrame(spec_to_rows(spec)),
    key=f"editor_{nonce}",
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
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

confirmed = st.session_state.get(CONFIRMED_KEY)
if st.button("Подтвердить результат", type="primary", disabled=candidate is None):
    st.session_state[CONFIRMED_KEY] = candidate
    confirmed = candidate

if confirmed is not None:
    if candidate is not None and candidate != confirmed:
        st.info("Есть изменения, которые ещё не подтверждены. Нажмите «Подтвердить результат».")
    st.success(
        f"Процесс «{confirmed.name}» подтверждён и сохранён для дальнейшего анализа "
        f"({len(confirmed.steps)} операций). Перейдите на страницу «Анализ процесса»."
    )
    with st.expander("ProcessSpec (JSON)"):
        st.json(confirmed.model_dump())
