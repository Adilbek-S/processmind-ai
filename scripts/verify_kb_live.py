"""Живая проверка RAG: РЕАЛЬНЫЕ эмбеддинги OpenAI (нужен OPENAI_API_KEY) + ChromaDB.

Запуск: `python -m scripts.verify_kb_live`

Для каждого паттерна задан сценарий из жизни, сформулированный «своими словами» (без названий
паттернов). Проверяется, что нужный паттерн попадает в Top-3. Индекс строится в отдельном каталоге
`data/chroma_verify` и не затрагивает индекс приложения. Код возврата 0 — все сценарии пройдены.
"""

from __future__ import annotations

import sys

from processmind.config import get_settings
from processmind.rag.errors import KnowledgeBaseError
from processmind.rag.knowledge_base import build_knowledge_base, search_automation_patterns

PERSIST_DIR = "data/chroma_verify"

SCENARIOS: list[tuple[str, str]] = [
    ("AP-001", "Все входящие обращения падают в общий ящик, и координатор вручную решает, кому из сотрудников передать каждое."),
    ("AP-002", "Договоры ходят по кабинетам на бумаге, подписи собираем вручную, никто не знает, на каком этапе сейчас документ."),
    ("AP-003", "Клиенты присылают формы с пустыми полями и неправильными датами, приходится перезванивать и уточнять."),
    ("AP-004", "Исполнители узнают о новых задачах случайно, а клиенты постоянно спрашивают, что с их заявкой."),
    ("AP-005", "Одни и те же данные клиента мы набираем в трёх разных местах, и в копиях появляются расхождения."),
    ("AP-006", "Статус из CRM не попадает в учётную систему, программы не обмениваются данными друг с другом."),
    ("AP-007", "Приходят сканы счетов на бумаге, оператор вручную перепечатывает из них реквизиты и суммы."),
    ("AP-008", "Письма, звонки и формы с сайта заносят в журнал вручную, часть теряется, заявитель не получает подтверждения."),
    ("AP-009", "О просрочке мы узнаём уже после того, как срок прошёл, хотелось бы получать предупреждение заранее."),
    ("AP-010", "Каждое типовое письмо и акт составляем заново, копируя из старых файлов, в них остаются чужие данные."),
    ("AP-011", "Сотрудник читает каждое сообщение, чтобы понять тему и срочность, похожие обращения получают разные категории."),
    ("AP-012", "Каждый месяц аналитик выгружает данные из разных систем и вручную собирает сводную таблицу для руководителя."),
    ("AP-013", "Срочные задачи тонут среди рутинных, одни сотрудники перегружены, а другие ждут работы."),
    ("AP-014", "Раз в квартал мы вручную сравниваем записи в двух системах и ищем, где они разошлись."),
]


def main() -> int:
    settings = get_settings()
    try:
        report = build_knowledge_base(persist_dir=PERSIST_DIR)
        print(
            f"Индекс: паттернов {report.total_patterns}, фрагментов {report.total_chunks}, "
            f"отправлено на эмбеддинг {report.embedded_chunks} (модель {settings.embedding_model})"
        )
        again = build_knowledge_base(persist_dir=PERSIST_DIR)
        dedup_ok = again.embedded_chunks == 0 and len(again.unchanged) == report.total_patterns
        print(f"Повторная индексация без дублей и без новых эмбеддингов: {'OK' if dedup_ok else 'ОШИБКА'}")

        misses = 0
        for expected, query in SCENARIOS:
            matches = search_automation_patterns(query, top_k=3, persist_dir=PERSIST_DIR)
            ranked = [m.pattern_id for m in matches]
            ok = expected in ranked
            misses += not ok
            print(f"\n[{'OK' if ok else 'MISS'}] ожидался {expected}: {query}")
            for i, m in enumerate(matches, start=1):
                best = m.chunks[0]
                print(f"   {i}. {m.pattern_id} {m.title} ({m.score:.3f}) <- «{best.section}»")
    except KnowledgeBaseError as exc:
        print(f"ОШИБКА: {exc}")
        return 1

    print(f"\nИтог: {len(SCENARIOS) - misses}/{len(SCENARIOS)} сценариев — нужный паттерн в Top-3")
    return 0 if misses == 0 and dedup_ok else 1


if __name__ == "__main__":
    sys.exit(main())
