from processmind.rag.engine import RAGEngine


def test_rag_add_and_query_offline(tmp_path):
    engine = RAGEngine(collection_name="test_kb", persist_dir=str(tmp_path))
    engine.add_documents(
        [
            "Процесс согласования договора занимает много времени из-за ручной пересылки email.",
            "Клиент оставляет заявку через веб-форму на сайте компании.",
        ]
    )
    results = engine.query("согласование договора", n_results=1)
    assert len(results) == 1
    assert "договора" in results[0]


def test_rag_query_empty_collection_returns_empty_list(tmp_path):
    engine = RAGEngine(collection_name="empty_kb", persist_dir=str(tmp_path))
    assert engine.query("что угодно") == []
