"""Генерация EVALS.md из фактических результатов прогона (`evals/results/latest.json`).

В отчёт не попадает ни одного числа, которого нет в результатах; выводы формируются кодом строго по этим числам.
"""

from __future__ import annotations

from typing import Any

EPS = 0.005  # различия меньше этой величины (для долей) считаем отсутствием различий
F1_NOISE_MARGIN = 0.05  # условное соглашение: при десятках примеров различия F1 меньше 0.05 неотличимы от шума


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "—"


def _f(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _delta(a: float, b: float, digits: int = 3) -> str:
    d = b - a
    return f"{d:+.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _verdict(a: float, b: float, higher_better: bool = True, eps: float = EPS) -> str:
    if abs(b - a) < eps:
        return "различие в пределах шума (по условному порогу)"
    better = (b > a) == higher_better
    return "B лучше A" if better else "A лучше B"


def render_meta(meta: dict[str, Any]) -> str:
    p = meta["params"]
    rows = [
        ["Дата прогона (UTC)", meta["generated_at"]],
        ["Длительность прогона", f"{meta['duration_s']:.0f} с"],
        ["Версия кода (git)", meta["git"]],
        ["LLM", meta["model"]],
        ["Модель эмбеддингов", meta["embedding_model"]],
        ["Параметры генерации в оценках", f"temperature={p['temperature']}, top_p={p['top_p']}, max_tokens={p['max_tokens']}, seed={p['seed']}"],
        ["Golden Dataset (sha256)", f"`{meta['dataset_sha'][:16]}…`"],
        ["База знаний (отпечаток)", f"`{meta['kb_sha'][:16]}…`"],
        ["Skill process-analysis (sha256)", f"`{meta['skill_sha'][:16]}…`"],
        ["Версии библиотек", ", ".join(f"{k} {v}" for k, v in meta["versions"].items())],
    ]
    return _table(["Параметр", "Значение"], rows)


def render_tracing(meta: dict[str, Any]) -> str:
    ls = meta["langsmith"]
    lines = []
    if ls.get("verification"):
        v = ls["verification"]
        lines.append(f"**LangSmith: трейсы проверены на реальном прогоне.** Проект `{v['project']}`, trace `{v['trace_id']}`, статус анализа `{v['status']}`.")
        lines.append("")
        lines.append(_table(["Проверка", "Результат"], [[k, "выполнена" if ok else "НЕ выполнена"] for k, ok in v["checks"].items()]))
        lines.append("")
        lines.append(f"Runs в трейсе: {v['runs']} ({v['by_type']}); токены LLM: {v['tokens']}; длительность корневого трейса: {v['duration_s']} с; runs с ошибкой: {v['error_runs'] or 'нет'}.")
    else:
        lines.append("**LangSmith: реальные трейсы в этом прогоне НЕ проверены.** " + (ls.get("reason") or "трассировка не настроена") + ".")
        lines.append("")
        lines.append(
            "Инструментация (узлы LangGraph, LLM-вызовы с токенами, RAG retrieval, вызовы MCP, длительности, ошибки) проверена автоматическими "
            "тестами `tests/test_tracing.py` на локальном двойнике LangSmith API — это подтверждает, ЧТО именно отправляется, но не работу облачного "
            "сервиса. Чтобы получить реальные трейсы, настройте LangSmith (см. README, раздел «Мониторинг: LangSmith») и выполните "
            "`python -m evals.verify_tracing`; затем перезапустите `python -m evals.run_all`, и этот раздел заполнится фактическими данными.")
    return "\n".join(lines)


def render_dataset(d: dict[str, Any]) -> str:
    cat_rows = [[c, n] for c, n in d["by_category"].items()]
    kind_names = {"positive": "positive (есть ожидаемые паттерны)", "negative": "negative (автоматизация не обоснована)", "ambiguous": "ambiguous (неоднозначное описание)"}
    return (
        f"{d['n']} синтетических примеров (`evals/golden_dataset.json`, исходник `evals/dataset_source.py`). Все процессы выдуманы; реальные регламенты не используются.\n\n"
        + _table(["Категория", "Примеров"], cat_rows)
        + "\n\n"
        + _table(["Тип", "Примеров"], [[kind_names[k], n] for k, n in d["by_kind"].items()])
        + "\n\nДля каждого примера хранятся: `test_id`, описание процесса/операции, структурированный процесс, вопрос, ожидаемые ID паттернов, допустимые и недопустимые рекомендации."
    )


def render_retrieval(r: dict[str, Any]) -> str:
    text = (
        f"На {r['n_positive']} positive-примерах (запрос строится для целевой операции так же, как в узле `retrieve_automation_patterns`):\n\n"
        + _table(
            ["Метрика", "Значение"],
            [
                ["**Retrieval Hit@3**", f"**{_pct(r['hits'], r['n_positive'])}**"],
                ["Recall@3 (среднее по примерам)", _f(r["mean_recall_at_3"])],
                ["Hit по пулу workflow (объединение Top-3 для процесса и каждой операции)", f"{r['pool_hit_rate']:.0%}, средний размер пула {r['mean_pool_size']:.1f}"],
                ["Задержка одного поиска (среднее / медиана / p95)", f"{r['latency_s']['mean']:.2f} / {r['latency_s']['median']:.2f} / {r['latency_s']['p95']:.2f} с"],
                ["Токены эмбеддингов на поиск (всего)", r["embedding_tokens_total"]],
            ],
        )
    )
    misses = [row for row in r["rows"] if not row["hit_at_3"]]
    if misses:
        text += "\n\nПромахи Hit@3:\n\n" + _table(["Пример", "Ожидалось", "Найдено в Top-3"], [[m["test_id"], ", ".join(m["expected"]), ", ".join(m["top3"])] for m in misses])
    else:
        text += "\n\nПромахов Hit@3 нет."
    return text


def _cfg(m: dict[str, Any], key: str, sub: str | None = None) -> Any:
    return m[key] if sub is None else m[key][sub]


def render_ab(ab: dict[str, Any]) -> str:
    A, B = ab["metrics"]["A"], ab["metrics"]["B"]
    fr_a, fr_b = A["f1_resolved"], B["f1_resolved"]
    rows = [
        ["Оценено примеров (ошибок LLM)", f"{A['evaluated']}/{A['n']} ({len(A['errors'])})", f"{B['evaluated']}/{B['n']} ({len(B['errors'])})", ""],
        ["**Recommendation F1, macro** (по смыслу рекомендации¹)", _f(fr_a["macro_f1"]), _f(fr_b["macro_f1"]), _delta(fr_a["macro_f1"], fr_b["macro_f1"])],
        ["Recommendation F1, micro (по смыслу¹)", _f(fr_a["micro_f1"]), _f(fr_b["micro_f1"]), _delta(fr_a["micro_f1"], fr_b["micro_f1"])],
        ["Precision / Recall, macro (по смыслу¹)", f"{_f(fr_a['mean_precision'])} / {_f(fr_a['mean_recall'])}", f"{_f(fr_b['mean_precision'])} / {_f(fr_b['mean_recall'])}", ""],
        ["Recommendation F1, macro (по явному `pattern_id`²)", _f(A["f1_explicit"]["macro_f1"]), _f(B["f1_explicit"]["macro_f1"]), "A по построению не может назвать ID"],
        ["Отрицательные: корректно нет рекомендаций (после валидации)", _pct(A["negatives"]["correct_after_validation"], A["negatives"]["n"]), _pct(B["negatives"]["correct_after_validation"], B["negatives"]["n"]), ""],
        ["Отрицательные: корректно нет рекомендаций (сырой ответ LLM)", _pct(A["negatives"]["correct_raw_llm"], A["negatives"]["n"]), _pct(B["negatives"]["correct_raw_llm"], B["negatives"]["n"]), ""],
        ["Неоднозначные: корректный ответ", _pct(A["ambiguous"]["correct"], A["ambiguous"]["n"]), _pct(B["ambiguous"]["correct"], B["ambiguous"]["n"]), ""],
        ["Недопустимые рекомендации (примеров с нарушением)", _pct(A["violations_resolved"]["examples_with_violation"], A["violations_resolved"]["n"]), _pct(B["violations_resolved"]["examples_with_violation"], B["violations_resolved"]["n"]), ""],
        ["Рекомендаций на пример (после валидации)", _f(A["justification"]["per_example"], 2), _f(B["justification"]["per_example"], 2), ""],
        ["Отклонено программной валидацией (из сырых)", f"{A['rejected_by_validation']} из {A['raw_recommendations']}", f"{B['rejected_by_validation']} из {B['raw_recommendations']}", ""],
        ["Обоснование присутствует (≥40 символов)", f"{A['justification']['rationale_present']:.0%}", f"{B['justification']['rationale_present']:.0%}", ""],
        ["Ссылка на паттерн базы знаний", f"{A['justification']['kb_reference']:.0%}", f"{B['justification']['kb_reference']:.0%}", ""],
        ["Разобраны условия применимости паттерна", f"{A['justification']['conditions_analyzed']:.0%}", f"{B['justification']['conditions_analyzed']:.0%}", ""],
        ["Latency LLM, среднее / p95, с", f"{A['llm_latency_s']['mean']:.2f} / {A['llm_latency_s']['p95']:.2f}", f"{B['llm_latency_s']['mean']:.2f} / {B['llm_latency_s']['p95']:.2f}", _delta(A["llm_latency_s"]["mean"], B["llm_latency_s"]["mean"], 2) + " с (среднее)"],
        ["Latency сквозная (поиск + LLM), среднее / p95, с", f"{A['total_latency_s']['mean']:.2f} / {A['total_latency_s']['p95']:.2f}", f"{B['total_latency_s']['mean']:.2f} / {B['total_latency_s']['p95']:.2f}", _delta(A["total_latency_s"]["mean"], B["total_latency_s"]["mean"], 2) + " с (среднее)"],
        ["Токены prompt, среднее на пример", f"{A['prompt_tokens']['mean']:.0f}", f"{B['prompt_tokens']['mean']:.0f}", f"{B['prompt_tokens']['mean'] - A['prompt_tokens']['mean']:+.0f}"],
        ["Токены completion, среднее на пример", f"{A['completion_tokens']['mean']:.0f}", f"{B['completion_tokens']['mean']:.0f}", f"{B['completion_tokens']['mean'] - A['completion_tokens']['mean']:+.0f}"],
        [f"Токены LLM всего ({A['evaluated']} примеров)", f"{A['total_tokens']['total']:.0f}", f"{B['total_tokens']['total']:.0f}", f"{B['total_tokens']['total'] - A['total_tokens']['total']:+.0f}"],
        ["Токены эмбеддингов запросов поиска (всего)", "0", str(B["embedding_tokens_total"]), ""],
    ]
    table = _table(["Метрика", "A: LLM без RAG", "B: LLM с RAG", "B − A"], rows)
    notes = (
        "\n\n¹ *По смыслу*: паттерн, ближайший (по эмбеддингу) к тексту «проблема + решение» каждой принятой рекомендации, определяется одинаково для A и B — "
        "иначе сравнивать нельзя, потому что A не знает ID паттернов. ² *По явному ID*: паттерн, названный LLM в `pattern_id` (для A всегда пусто).\n"
    )
    return table + notes + "\n" + render_ab_conclusions(A, B)


def _takeaway(A: dict[str, Any], B: dict[str, Any]) -> str:
    """Одно-два предложения по существу, строго из чисел."""
    fa, fb = A["f1_resolved"]["macro_f1"], B["f1_resolved"]["macro_f1"]
    if abs(fb - fa) < F1_NOISE_MARGIN:
        quality = "RAG не дал измеримого прироста качества паттернов (различие macro-F1 в пределах шума)"
    else:
        quality = f"RAG {'повысил' if fb > fa else 'понизил'} macro-F1 на {abs(fb - fa):.3f}"
    na, nb = A["negatives"], B["negatives"]
    neg_a, neg_b = na["correct_after_validation"], nb["correct_after_validation"]
    if neg_b < neg_a:
        neg = f"на отрицательных примерах с RAG корректных ответов меньше ({neg_b}/{nb['n']} против {neg_a}/{na['n']}; выборка всего {nb['n']} примера) — возможная причина: получив паттерны, модель чаще предлагает автоматизацию там, где она не обоснована (причинность этой выборкой не доказана)"
    elif neg_b > neg_a:
        neg = f"на отрицательных примерах с RAG корректных ответов больше ({neg_b}/{nb['n']} против {neg_a}/{na['n']})"
    else:
        neg = f"на отрицательных примерах разницы нет ({neg_b}/{nb['n']})"
    trace = "; зато только с RAG рекомендации содержат ссылки на паттерны базы знаний и разбор условий применимости" if B["justification"]["kb_reference"] > A["justification"]["kb_reference"] else ""
    return f"{quality}; {neg}{trace}. Цена — рост задержки и токенов (см. таблицу)."


def render_ab_conclusions(A: dict[str, Any], B: dict[str, Any]) -> str:
    fa, fb = A["f1_resolved"]["macro_f1"], B["f1_resolved"]["macro_f1"]
    lines = ["**Выводы (сформированы по числам выше):**", ""]
    lines.append(f"- Качество (macro-F1 по смыслу): A = {fa:.3f}, B = {fb:.3f}, разница {fb - fa:+.3f} — {_verdict(fa, fb, eps=F1_NOISE_MARGIN)} (порог {F1_NOISE_MARGIN}, условное соглашение для малой выборки).")
    na, nb = A["negatives"], B["negatives"]
    lines.append(f"- Отрицательные примеры: после программной валидации корректны {na['correct_after_validation']}/{na['n']} (A) и {nb['correct_after_validation']}/{nb['n']} (B); "
                 f"сырой ответ LLM без валидации: {na['correct_raw_llm']}/{na['n']} (A) и {nb['correct_raw_llm']}/{nb['n']} (B).")
    lines.append(f"- Недопустимые рекомендации: примеров с нарушением {A['violations_resolved']['examples_with_violation']} (A) и {B['violations_resolved']['examples_with_violation']} (B).")
    lines.append(f"- Обоснования: у B ссылку на паттерн базы знаний содержит {B['justification']['kb_reference']:.0%} рекомендаций (у A — {A['justification']['kb_reference']:.0%}); "
                 f"условия применимости разобраны у {B['justification']['conditions_analyzed']:.0%} (B) и {A['justification']['conditions_analyzed']:.0%} (A) рекомендаций.")
    ta, tb = A["total_latency_s"]["mean"], B["total_latency_s"]["mean"]
    lines.append(f"- Задержка: сквозная средняя {ta:.2f} с (A) против {tb:.2f} с (B), разница {tb - ta:+.2f} с; из них поиск в B — {B['retrieval_latency_s']['mean']:.2f} с в среднем.")
    pa, pb = A["prompt_tokens"]["mean"], B["prompt_tokens"]["mean"]
    lines.append(f"- Токены: prompt в среднем {pa:.0f} (A) против {pb:.0f} (B) — у B входных токенов в {pb / pa:.2f} раза больше (паттерны из базы знаний добавляются в запрос); completion {A['completion_tokens']['mean']:.0f} (A) и {B['completion_tokens']['mean']:.0f} (B)." if pa else "- Токены: нет данных.")
    lines.append(f"- **Главное:** {_takeaway(A, B)}")
    lines.append(f"- Ограничения интерпретации: {A['f1_resolved']['n_positive']} positive-примеров, один запуск на конфигурацию (seed фиксирован), проверки значимости нет — разницы это точечные оценки, а не статистически подтверждённые эффекты.")
    return "\n".join(lines)


def render_metrics_b(ab: dict[str, Any]) -> str:
    B = ab["metrics"]["B"]
    f = B["f1_resolved"]
    return (
        "Рабочая конфигурация (B — LLM с RAG), все примеры датасета:\n\n"
        + _table(
            ["Метрика", "Значение"],
            [
                ["Recommendation F1 (macro / micro) — по смыслу", f"{f['macro_f1']:.3f} / {f['micro_f1']:.3f} (TP={f['tp']}, FP={f['fp']}, FN={f['fn']})"],
                ["Recommendation F1 (macro / micro) — по явному `pattern_id`", f"{B['f1_explicit']['macro_f1']:.3f} / {B['f1_explicit']['micro_f1']:.3f}"],
                ["Отрицательные примеры: нет необоснованных рекомендаций", _pct(B["negatives"]["correct_after_validation"], B["negatives"]["n"])],
                ["Latency LLM (среднее / медиана / p95)", f"{B['llm_latency_s']['mean']:.2f} / {B['llm_latency_s']['median']:.2f} / {B['llm_latency_s']['p95']:.2f} с"],
                ["Токены на пример (prompt / completion / total, среднее)", f"{B['prompt_tokens']['mean']:.0f} / {B['completion_tokens']['mean']:.0f} / {B['total_tokens']['mean']:.0f}"],
            ],
        )
        + "\n\nОпределения метрик — в `evals/scoring.py`. F1 считается по множествам паттернов на примере: TP = найденные ожидаемые, FP = предложенные вне ожидаемых и допустимых, "
        "FN = ожидаемые, но не предложенные; допустимые паттерны не поощряются и не штрафуются."
    )


def _rule_check(temp: dict[str, Any]) -> str:
    """Проверка каждого условия правила выбора по фактическим числам."""
    low, high = (str(t) for t in sorted(temp["temperatures"]))
    lo, hi = temp["summary"][low], temp["summary"][high]
    gain, rule = hi["mean_f1"] - lo["mean_f1"], temp["min_f1_gain_rule"]
    checks = [
        (f"прирост macro-F1 при temperature={high} не менее {rule}", gain >= rule, f"{gain:+.3f}"),
        ("устойчивость (Jaccard) не хуже", hi["stability_jaccard"] >= lo["stability_jaccard"], f"{hi['stability_jaccard']:.3f} против {lo['stability_jaccard']:.3f}"),
        ("доля запусков с нарушением не хуже", hi["violation_rate"] <= lo["violation_rate"], f"{hi['violation_rate']:.0%} против {lo['violation_rate']:.0%}"),
    ]
    text = "Проверка условий правила по фактическим числам:\n\n" + _table(["Условие для выбора большей temperature", "Выполнено", "Значение"], [[c, "да" if ok else "нет", v] for c, ok, v in checks])
    failed = [c for c, ok, _ in checks if not ok]
    if failed:
        text += f"\n\nБольшая temperature не выбрана: не выполнено — {'; '.join(failed)}."
        if gain >= rule and failed:
            text += " Обратите внимание: прирост F1 был существенным, а решение определило условие, различие по которому невелико, — при таких малых выборках это скорее тенденция, чем твёрдый вывод."
    return text


def render_hyperparams(chosen: dict[str, Any], temp: dict[str, Any] | None, default_matches: bool) -> str:
    text = _table(
        ["Параметр", "Выбранное значение", "Основание"],
        [
            ["temperature", chosen["temperature"], "по результатам эксперимента ниже" if temp else "эксперимент не выполнялся"],
            ["top_p", chosen["top_p"], "не варьировался; значение по умолчанию API (без усечения распределения)"],
            ["max_tokens", chosen["max_tokens"], f"наибольшая наблюдавшаяся длина ответа в оценках — {chosen['observed_max_completion_tokens']} токенов; лимит не достигался" if chosen["observed_max_completion_tokens"] < chosen["max_tokens"] else "ВНИМАНИЕ: лимит достигался — увеличьте"],
        ],
    )
    text += "\n\nЗначения зафиксированы в `processmind/config.py` (`LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_MAX_TOKENS`, переопределяются в `.env`)" + ("." if default_matches else ", НО значение по умолчанию отличается от выбранного — обновите его.")
    if not temp:
        return text
    rule = temp["min_f1_gain_rule"]
    s = temp["summary"]
    rows = []
    for t in temp["temperatures"]:
        m = s[str(t)]
        rows.append([t, m["runs"], _f(m["mean_f1"]), _f(m["stability_jaccard"]), f"{m['violation_rate']:.0%}", f"{m['negative_correct_rate']:.0%}", f"{m['mean_llm_latency_s']:.2f}", f"{m['mean_total_tokens']:.0f}"])
    text += (
        f"\n\n### Эксперимент со temperature\n\nПримеры (фиксированный набор, по одному из шести категорий): {', '.join(temp['examples'])}. Конфигурация B, значения temperature {temp['temperatures']}, "
        f"по 3 повтора на пример с фиксированными seed {temp['seeds']}; поиск выполнен один раз на пример, меняется только temperature. Всего запусков на значение — {s[str(temp['temperatures'][0])]['runs']}.\n\n"
        f"**Правило выбора (задано до эксперимента):** большая temperature выбирается только если даёт прирост macro-F1 не менее {rule} и не ухудшает устойчивость и долю нарушений; иначе выбирается меньшая (воспроизводимость важнее).\n\n"
        + _table(["temperature", "Запусков", "F1 (среднее)", "Устойчивость (Jaccard между повторами)", "Доля запусков с нарушением", "Отрицательные: корректно", "Latency LLM, с", "Токены (среднее)"], rows)
        + "\n\n" + _rule_check(temp)
        + f"\n\n**Выбрано temperature = {temp['chosen_temperature']}.**\n\n"
        "Оговорки: 6 примеров × 3 повтора — малая выборка; seed в OpenAI API обеспечивает воспроизводимость лишь по возможности (`system_fingerprint` может меняться); "
        "эксперимент показывает тенденцию, а не статистически значимую разницу."
    )
    return text


LIMITATIONS = """\
- Golden Dataset и база знаний написаны одним автором и синтетические: формулировки примеров близки по смыслу к текстам паттернов, поэтому Hit@3 и F1 вероятно **оптимистичнее**, чем были бы на реальных описаниях процессов.
- 30 примеров (23 positive) — малая выборка; доверительные интервалы не считались, различия между конфигурациями — точечные оценки.
- Качество рекомендаций «по смыслу» определяется ближайшим паттерном по эмбеддингу текста рекомендации (без порога): это приближение, а не экспертная оценка; рекомендация, не похожая ни на один паттерн, всё равно получает ближайший.
- «Обоснование присутствует» проверяет наличие текста, а не его правдивость; правдивость обоснований эти метрики не оценивают.
- Один запуск на конфигурацию; недетерминизм LLM полностью не устраняется даже при temperature=0 и фиксированном seed.
- В сравнении A/B пользовательский вопрос из датасета добавляется в запрос к LLM (в приложении такого вопроса нет).
"""


def render_appendix(dataset_rows: list[dict[str, Any]]) -> str:
    return _table(
        ["Пример", "Категория", "Тип", "Ожидалось", "A (по смыслу)", "B (по смыслу)", "B (явный ID)"],
        [[r["test_id"], r["category"], r["kind"], ", ".join(r["expected"]) or "—", ", ".join(r["a"]) or "—", ", ".join(r["b"]) or "—", ", ".join(r["b_explicit"]) or "—"] for r in dataset_rows],
    )


def render(results: dict[str, Any]) -> str:
    meta = results["meta"]
    parts = [
        "# EVALS — автоматизированные оценки ProcessMind AI",
        "",
        "> Файл сгенерирован командой `python -m evals.run_all`. Все числа получены в реальном прогоне (настоящие вызовы OpenAI, RAG над ChromaDB и MCP-сервер); "
        "результаты в `evals/results/latest.json`. Вручную файл не редактируется — перезапуск команды пересоздаёт его.",
        "",
        "## 0. Условия прогона",
        "",
        render_meta(meta),
        "",
        "## 1. Мониторинг: LangSmith",
        "",
        render_tracing(meta),
        "",
        "## 2. Golden Dataset",
        "",
        render_dataset(results["dataset"]),
        "",
        "## 3. Метрики",
        "",
        "### 3.1 Retrieval Hit@3",
        "",
        render_retrieval(results["retrieval"]),
        "",
        "### 3.2 Recommendation F1, отрицательные примеры, latency и токены",
        "",
        render_metrics_b(results["ab"]),
        "",
        "## 4. A/B-эксперимент: LLM без RAG (A) и с RAG (B)",
        "",
        "Одинаковые: модель, тестовые данные (одни и те же примеры), основные инструкции (Skill `process-analysis` как системный промпт), вопрос и температура/параметры генерации. "
        "Различие одно: у A блок `knowledge_base_patterns` пуст, у B содержит паттерны, найденные RAG.",
        "",
        render_ab(results["ab"]),
        "",
        "## 5. Гиперпараметры",
        "",
        render_hyperparams(results["chosen"], results.get("temperature"), results["chosen"]["default_matches"]),
        "",
        "## 6. Ограничения оценки",
        "",
        LIMITATIONS,
        "## Приложение: результаты по примерам",
        "",
        render_appendix(results["appendix"]),
        "",
    ]
    return "\n".join(parts)
