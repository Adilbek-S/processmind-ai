"""Ключи состояния сессии Streamlit и определение текущего этапа пользовательского сценария."""

from __future__ import annotations

from typing import Any, Mapping

DRAFT_KEY = "extraction_draft"  # ExtractionResult последнего распознавания
DRAFT_FILE_KEY = "extraction_draft_file"  # id файла, из которого получен черновик
NONCE_KEY = "extraction_nonce"  # меняется при каждом распознавании и сбрасывает состояние редактора
CONFIRMED_KEY = "confirmed_process_spec"  # подтверждённый ProcessSpec для анализа
RUN_KEY = "analysis_run"  # AnalysisRun: пауза на подтверждении либо готовый отчёт
SPEC_KEY = "analysis_spec_json"  # снимок подтверждённого процесса, для которого сделан запуск

STAGES = [
    ("Документ", "Загрузка PDF или PNG"),
    ("Операции", "Распознавание, просмотр и правка"),
    ("AI-анализ", "Метрики и рекомендации"),
    ("Выбор", "Рекомендации для TO-BE"),
    ("Отчёт", "TO-BE и скачивание"),
]


def current_stage(state: Mapping[str, Any]) -> int:
    """Индекс этапа (0-4) по состоянию сессии: чем далеко продвинулся пользователь."""
    run = state.get(RUN_KEY)
    if run is not None and state.get(CONFIRMED_KEY) is not None:
        return 3 if run.status == "awaiting_approval" else 4
    if state.get(CONFIRMED_KEY) is not None:
        return 2
    if state.get(DRAFT_KEY) is not None:
        return 1
    return 0
