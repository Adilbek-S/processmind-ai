# ProcessMind AI

Интеллектуальный анализ и автоматизация бизнес-процессов. Финальный проект курса
по LLM Engineering — MVP.

Пользователь загружает описание бизнес-процесса (текст, PDF или изображение схемы).
Система извлекает шаги процесса в единую модель **ProcessSpec**, анализирует процесс,
находит возможности автоматизации и формирует черновик целевой модели **TO-BE** с
AS-IS / TO-BE визуализацией.

> ⚠️ **Статус MVP:** извлечение процесса из PDF/PNG/JPG (GPT-4o-mini), поиск паттернов автоматизации
> (RAG: text-embedding-3-small + ChromaDB) и рекомендации по автоматизации (GPT-4o-mini) работают на
> реальных вызовах OpenAI и **требуют `OPENAI_API_KEY`**. Все числовые оценки эффекта — модельные
> (на основе заданных предположений), а не измерения.

## Стек

- Python 3.11+, Streamlit — интерфейс
- LangGraph — оркестрация пайплайна анализа
- OpenAI GPT-4o-mini — извлечение и анализ процессов (следующая итерация)
- ChromaDB + OpenAI Embeddings (`text-embedding-3-small`) — RAG по базе знаний
- Python MCP SDK / FastMCP — собственный MCP-сервер
- PyMuPDF — парсинг PDF
- Graphviz — визуализация AS-IS / TO-BE
- LangSmith — трассировка и оценка
- Pytest — тесты

## Архитектура

```
app.py                          # Streamlit: страница «Главная»
pages/
  1_Распознавание_процесса.py   # PDF/PNG/JPG -> ProcessSpec, правка таблицы, подтверждение
  2_Анализ_процесса.py          # LangGraph-пайплайн над подтверждённым ProcessSpec
  3_База_знаний.py              # Automation Knowledge Base: индексация и семантический поиск (Top-3)
  4_О_проекте.py                # архитектура, стек, ограничения MVP
processmind/
  config.py                     # настройки из переменных окружения (.env)
  models.py                     # Pydantic-модели: ProcessSpec и др.
  parsing/
    document_parser.py          # Document Parser (PyMuPDF)
    text_extractor.py           # сценарий А: текст PDF -> LLM -> RawProcess
    vision_parser.py            # сценарий Б: изображение -> GPT-4o-mini Vision -> RawProcess
    process_builder.py          # RawProcess -> ProcessSpec (нормализация, защита от выдумок, Pydantic)
    review.py                   # применение правок пользователя
    service.py                  # extract_process(filename, bytes) -> ExtractionResult
  ui/
    style.py                    # единый стиль: CSS, шапка, индикатор этапов, легенды
    components.py               # карточки метрик, таблица сравнения, таблица рекомендаций
    state.py                    # ключи состояния сессии и определение этапа
  visualization/
    diagrams.py                 # Graphviz-схемы AS-IS и TO-BE
  workflow/
    graph.py                    # главный LangGraph workflow (8 узлов, human-in-the-loop)
    approval.py                 # разбор решения пользователя на шаге human_approval
  analysis/
    skill.py                    # загрузка Skill process-analysis (методика -> системный промпт LLM)
    recommendations.py          # схемы рекомендаций, контекст для LLM, программная валидация
    report.py                   # AnalysisReport, построение TO-BE, Markdown-отчёт
  rag/
    patterns.py                 # загрузка Markdown-паттернов, chunking по разделам
    embeddings.py               # эмбеддинги OpenAI (без офлайн-подстановки)
    knowledge_base.py           # build_knowledge_base, search_automation_patterns, get_pattern_by_id
  mcp/
    server.py                   # собственный MCP-сервер (FastMCP)
  evaluation/
    pipeline.py                 # Evaluation Pipeline
.claude/skills/process-analysis/
  SKILL.md                      # методика анализа: и Skill Claude Code, и системный промпт LLM-узла
knowledge_base/
  automation_patterns/          # 14 синтетических паттернов автоматизации (по одному .md на паттерн)
scripts/                        # demo_data, verify_live, verify_kb_live
tests/                          # pytest
```

## Запуск

1. Создайте и активируйте виртуальное окружение (Python 3.11+):

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Установите зависимости:

   ```powershell
   pip install -r requirements.txt
   ```

3. Скопируйте `.env.example` в `.env` и при необходимости укажите свой `OPENAI_API_KEY`
   (без ключа распознавание процесса недоступно и сообщает об этом явной ошибкой):

   ```powershell
   copy .env.example .env
   ```

4. Запустите приложение одной командой:

   ```powershell
   streamlit run app.py
   ```

   Приложение откроется в браузере на `http://localhost:8501`.

## Интерфейс и пользовательский сценарий

Единый стиль без внешних frontend-библиотек: тема в `.streamlit/config.toml`, CSS и компоненты в `processmind/ui/`.
Индикатор этапов на каждой странице показывает, где находится пользователь.

| Этап | Страница | Что происходит |
|---|---|---|
| 1. Документ | Распознавание процесса | загрузка PDF или PNG |
| 2. Операции | Распознавание процесса | распознавание, просмотр и правка таблицы; схема AS-IS обновляется по мере правок; подтверждение |
| 3. AI-анализ | AI-анализ | запуск workflow; исходные метрики (MCP), схема AS-IS, выявленные проблемы, найденные паттерны |
| 4. Выбор | AI-анализ | таблица рекомендаций; выбор рекомендаций и допущений о длительности (workflow на паузе) |
| 5. Отчёт | AI-анализ | сравнение AS-IS и TO-BE, схемы рядом, таблица решений, отчёт в Markdown (предпросмотр и скачивание) |

**Схемы Graphviz** (`processmind/visualization/diagrams.py`). AS-IS: последовательность операций, название, исполнитель,
длительность; ручные операции жёлтые, автоматизированные зелёные, с неуказанным типом серые. TO-BE: те же операции
и связи (маршруты не перестраиваются, шлюзы не добавляются); операции с принятой автоматизацией синие и подписаны
«было → стало» и ID паттерна.

**Сравнение AS-IS / TO-BE:** общее время, число и доля ручных операций, условная месячная трудоёмкость. Метрики TO-BE
считает тот же MCP-инструмент `calculate_process_metrics`, что и AS-IS.

**Отчёт в Markdown** содержит описание процесса, исходные метрики, выявленные проблемы, таблицу рекомендаций
(Операция | Проблема | Предложение | Паттерн автоматизации | Ожидаемый эффект), использованные паттерны с документами
базы знаний, модельные показатели TO-BE, предупреждения; вверху и внизу — оговорка о синтетических данных и
оценочном характере эффекта.

## Главный workflow анализа (LangGraph)

```
validate_process ─(ошибки)──────────────────────────────────────────────┐
   │ MCP                                                                 │
calculate_metrics ── MCP                                                 │
   │                                                                     │
retrieve_automation_patterns ── RAG (ChromaDB + эмбеддинги)              │
   │                                                                     │
generate_recommendations ── LLM + Skill process-analysis                 │
   │                                                                     │
validate_recommendations ─(нет обоснованных рекомендаций)────────────────┤
   │                                                                     │
human_approval ══ ПАУЗА (interrupt) до решения пользователя ─(ничего не выбрано)┤
   │                                                                     │
simulate_automation ── MCP                                               │
   ▼                                                                     ▼
generate_report ◄────────────────────────────────────────────────────────┘
```

Любой исход заканчивается структурированным `AnalysisReport` (статусы: `completed`, `no_selection`,
`no_recommendations`, `invalid`, `failed`). Один граф, без многоагентной архитектуры.

- **Компоненты не дублируются:** валидацию, метрики и симуляцию выполняет MCP-сервер (отдельный процесс, stdio,
  `SyncMCPClient`); поиск — `search_automation_patterns` и `get_pattern_by_id`; вызов OpenAI — общий
  `parsing/llm.py`; интерфейс — страница «Анализ процесса».
- **Human-in-the-loop:** `human_approval` вызывает `interrupt()`, состояние сохраняется в checkpointer
  (`MemorySaver`: живёт, пока работает приложение). Симуляция запускается только после
  `resume_analysis(graph, thread_id, {"selections": [{"rec_id": "R1", "new_duration_minutes": 5}]})`.
  Некорректное решение не роняет граф: запрос повторяется с текстом ошибки.
- **Skill применяется по-настоящему:** `.claude/skills/process-analysis/SKILL.md` читается с диска при каждом
  запуске и становится системным промптом узла `generate_recommendations`; в отчёт попадает sha256 применённой
  версии. Правка методики меняет поведение без перезапуска.
- **LLM предлагает, код проверяет** (`validate_recommendations`): отклоняются несуществующие ID операций, ссылки
  на паттерны, которых не было в результатах поиска, уже автоматизированные операции, операции с неуказанным
  типом выполнения без явных признаков ручной работы, количественные утверждения об эффекте. Факты об
  операции (исполнитель, тип, длительность) подставляет код, а не LLM.
- **Длительность после автоматизации — допущение.** LLM может её предложить (с обоснованием); пользователь
  подтверждает или задаёт своё значение перед симуляцией. Процент сокращения никогда не подставляется.

Известные ограничения: качество рекомендаций определяется моделью (GPT-4o-mini часто не предлагает длительность
и может неточно оценивать применимость паттерна — поэтому есть шаг подтверждения и явное разделение
«подтверждено данными / не подтверждено»; модель настраивается через `OPENAI_MODEL`).

```powershell
python -m scripts.run_analysis --assume R1=5   # сквозной прогон демо-процесса на реальных OpenAI, RAG и MCP
```

## Распознавание процесса

Страница «Распознавание процесса»: загрузите PDF/PNG/JPG → «Распознать процесс» → исправьте
таблицу операций (название, участник, длительность, тип выполнения) → «Подтвердить результат».
Подтверждённый `ProcessSpec` хранится в `st.session_state["confirmed_process_spec"]` и
используется страницей «Анализ процесса».

Правила: длительность, участник и тип выполнения не выдумываются — если их нет в документе,
поле остаётся пустым. Для PDF длительность принимается только с дословной цитатой из текста, а
участник — только если найден в тексте; иначе значение отбрасывается с предупреждением.
Для изображений такая сверка невозможна, поэтому распознанные длительности помечаются
предупреждением «проверьте в таблице».

Синтетические демо-файлы и эталон:

```powershell
python -m scripts.demo_data        # samples/demo_process.pdf, .png, demo_process_truth.json
python -m scripts.verify_live      # живой прогон обоих сценариев с реальным OpenAI, сравнение с эталоном
```

## Automation Knowledge Base (RAG)

`knowledge_base/automation_patterns/` — 14 синтетических паттернов автоматизации, каждый в своём
Markdown-файле (ID, название, описание проблемы, условия применимости, предлагаемый подход,
ожидаемый качественный эффект, ограничения). Числовых эффектов и ссылок на нормативные документы
в паттернах нет.

Пайплайн: Markdown → chunking по разделам (1 раздел = 1 chunk) → `text-embedding-3-small` →
ChromaDB (косинусная метрика) → поиск по вектору запроса → Top-3 паттернов. С каждым фрагментом
хранятся `pattern_id`, `title`, `source`, `section`, `chunk_id`.

```python
from processmind.rag.knowledge_base import build_knowledge_base, search_automation_patterns, get_pattern_by_id

build_knowledge_base()                                  # идемпотентно: неизменённые документы не пересчитываются
matches = search_automation_patterns("вручную перепечатываем данные из сканов", top_k=3)
pattern = get_pattern_by_id(matches[0].pattern_id)      # полный паттерн из индекса
```

Повторная индексация не создаёт дублей: chunk_id детерминирован, документ сравнивается по хэшу
содержимого, изменённый паттерн заменяется целиком, удалённый из каталога — убирается из индекса.
Коллекция называется по модели эмбеддингов, поэтому векторы разных моделей не смешиваются.
Без `OPENAI_API_KEY` индексация и поиск завершаются ошибкой (подмены эмбеддингов нет).

В интерфейсе («База знаний») запрос можно ввести вручную или сформировать из подтверждённого
процесса (весь процесс либо отдельная операция).

```powershell
python -m scripts.verify_kb_live   # живая проверка: 14 сценариев, нужный паттерн должен быть в Top-3
```

## Тесты

```powershell
pytest
```

Тесты не обращаются к OpenAI: клиент подменён либо используется локальный HTTP-двойник для проверки
формата запросов SDK. В тестах RAG эмбеддинги лексические (тестовый двойник), поэтому они проверяют
обвязку — индексацию, дедупликацию, метаданные, группировку, а не семантику. Качество
распознавания и поиска с реальными моделями проверяют `scripts.verify_live` и
`scripts.verify_kb_live`.

## Собственный MCP-сервер и MCP-клиент

Сервер (`processmind/mcp/server.py`, Python MCP SDK, `MCPServer` — бывший FastMCP) работает отдельным
процессом с транспортом stdio:

```powershell
python -m processmind.mcp.server
```

| Инструмент | Что делает |
|---|---|
| `validate_process` | Проверка структуры: схема ProcessSpec, обязательные поля, уникальность ID операций, длительности, связи, циклы. Возвращает `errors` и `warnings` |
| `calculate_process_metrics` | Число операций (всего/ручных/автоматизированных), доля ручных, время процесса и ручных операций, условная месячная трудоёмкость (нужен `runs_per_month`) |
| `simulate_automation` | Модельная оценка: время до/после, абсолютное и процентное сокращение, доля ручных до/после, условная месячная экономия |
| `search_knowledge_base` | Поиск паттернов автоматизации (RAG) |
| `get_process_spec_schema` | JSON-схема ProcessSpec |

Все расчёты — обычный Python (`processmind/mcp/process_tools.py`), без LLM. Неуказанные длительности
не заменяются нулями (они перечисляются в `steps_without_duration`), операции с неуказанным типом не
считаются ручными (для доли есть `manual_share_upper_bound`). В `simulate_automation` длительность
после автоматизации задаёт вызывающий для каждой операции — процент сокращения не подставляется;
результат помечен `is_model_estimate: true`.

Клиент (`processmind/mcp/client.py`) — стандартный `stdio_client` + `ClientSession`: запускает сервер
подпроцессом и вызывает инструменты по JSON-RPC. Для LangGraph есть `MCPClient` (async) и
`SyncMCPClient` (одна долгоживущая сессия для sync-узлов):

```python
from processmind.mcp.client import SyncMCPClient

with SyncMCPClient() as mcp:
    metrics = mcp.calculate_process_metrics(spec, runs_per_month=120)
    result = mcp.simulate_automation(spec, [{"step_id": "step-2", "new_duration_minutes": 2}], 120)
```

```powershell
python -m scripts.demo_mcp   # живая демонстрация: сервер + клиент + результаты
```
