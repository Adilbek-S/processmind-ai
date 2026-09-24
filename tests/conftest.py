"""Общая изоляция тестов.

Ключи LangSmith (и OpenAI) могут лежать в `.env` разработчика. `Settings` читает `.env`, поэтому без защиты любой тест,
который запускает граф или вызывает LLM-обёртку, отправил бы синтетические трейсы в реальный LangSmith-проект.
На время тестов `.env` не читается (а кэш настроек и состояние трассировки сбрасываются); тесты, которым нужны
настройки, задают их явно.
"""

import pytest

from processmind import observability
from processmind.config import Settings, get_settings

Settings.model_config["env_file"] = None  # уже на этапе импорта: часть модулей читает настройки при импорте


@pytest.fixture(autouse=True)
def _do_not_read_dotenv(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setattr(observability, "_configured", None)
    yield
    get_settings.cache_clear()
