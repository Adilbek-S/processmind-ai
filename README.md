# ProcessMind AI

Интеллектуальный анализ и автоматизация бизнес-процессов. Финальный проект курса
по LLM Engineering — MVP.

Пользователь загружает описание бизнес-процесса (текст, PDF или изображение схемы).
Система извлекает шаги процесса в единую модель **ProcessSpec**, анализирует процесс,
находит возможности автоматизации и формирует черновик целевой модели **TO-BE** с
AS-IS / TO-BE визуализацией.

> ⚠️ **Статус MVP:** модуль извлечения процесса из PDF/PNG/JPG работает на реальных вызовах
> OpenAI (GPT-4o-mini) и **требует `OPENAI_API_KEY`**. Анализ процесса, TO-BE и оценка пока
> построены на детерминированных правилах (без LLM).

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
  3_База_знаний.py              # RAG: добавление документов и поиск
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
  workflow/
    graph.py                    # LangGraph Workflow
  analysis/
    process_analyzer.py         # Process Analyzer + генерация TO-BE-черновика
  rag/
    engine.py                   # RAG Engine (ChromaDB)
  mcp/
    server.py                   # собственный MCP-сервер (FastMCP)
  visualization/
    diagrams.py                 # AS-IS / TO-BE диаграммы (Graphviz)
  evaluation/
    pipeline.py                 # Evaluation Pipeline
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

## Тесты

```powershell
pytest
```

Тесты извлечения не обращаются к OpenAI: клиент подменён (или используется локальный HTTP-двойник
для проверки формата запроса SDK). Качество распознавания моделью проверяет только
`scripts.verify_live`.

## Собственный MCP-сервер

Запустить MCP-сервер ProcessMind отдельно (для подключения из MCP-клиента):

```powershell
python -m processmind.mcp.server
```
