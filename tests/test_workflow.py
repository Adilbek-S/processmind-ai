from processmind.workflow.graph import build_workflow

SAMPLE_TEXT = """
1. Клиент оставляет заявку на сайте
2. Менеджер вручную проверяет данные клиента в CRM
3. Бухгалтер вручную выставляет счёт
"""


def test_workflow_runs_end_to_end_without_llm():
    workflow = build_workflow()
    result = workflow.invoke({"raw_text": SAMPLE_TEXT, "errors": []})

    assert result["process_spec"] is not None
    assert len(result["process_spec"].steps) == 3
    assert result["analysis"] is not None
    assert result["tobe_spec"] is not None
    assert result["evaluation"] is not None
    assert result["evaluation"]["score"] > 0


def test_workflow_handles_empty_text():
    workflow = build_workflow()
    result = workflow.invoke({"raw_text": "", "errors": []})
    assert len(result["process_spec"].steps) == 1
