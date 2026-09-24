"""Дымовой тест интерфейса: главная и все страницы Streamlit запускаются без исключений.

Появился после того, как синтаксическая ошибка в app.py не была поймана ни одним тестом.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_page_runs_without_exceptions(script):
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception, [e.value for e in app.exception]
