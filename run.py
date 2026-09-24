"""Запуск ProcessMind AI одной командой: `python run.py`.

Что делает (каждый шаг пропускается, если уже выполнен):
1. создаёт виртуальное окружение `.venv` и ставит зависимости из requirements.txt;
2. создаёт `.env` из `.env.example`, если файла нет;
3. если в `.env` задан OPENAI_API_KEY — строит базу знаний (повторный запуск ничего не пересчитывает);
4. запускает Streamlit-приложение.

Дополнительные аргументы передаются Streamlit, например: `python run.py --server.port 8600`.
Значения из `.env` скрипт не печатает.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
STAMP = VENV_DIR / ".requirements.sha256"


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def has_env_value(name: str) -> bool:
    """True, если в .env задано непустое значение переменной (само значение не читается наружу)."""
    if not ENV_FILE.exists():
        return False
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name and value.strip().strip("\"'"):
            return True
    return False


def ensure_environment() -> Path:
    py = venv_python()
    if not py.exists():
        print("Создаю виртуальное окружение .venv …")
        venv.create(VENV_DIR, with_pip=True)
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not STAMP.exists() or STAMP.read_text(encoding="utf-8").strip() != digest:
        print("Устанавливаю зависимости (первый запуск может занять несколько минут) …")
        subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)], cwd=ROOT, check=True)
        STAMP.write_text(digest, encoding="utf-8")
    return py


def ensure_env_file() -> None:
    if not ENV_FILE.exists():
        shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
        print("Создан файл .env из .env.example.")


def ensure_knowledge_base(py: Path) -> None:
    if not has_env_value("OPENAI_API_KEY"):
        print(
            "\nВНИМАНИЕ: OPENAI_API_KEY не задан. Откройте файл .env, впишите ключ (OPENAI_API_KEY=...) и запустите `python run.py` снова.\n"
            "  Приложение запустится, но распознавание, поиск и анализ без ключа работать не будут (ошибка с инструкцией, без имитации)."
        )
        return
    code = (
        "from processmind.rag.knowledge_base import build_knowledge_base;"
        "r = build_knowledge_base();"
        "print(f'База знаний готова: паттернов {r.total_patterns}, новых эмбеддингов {r.embedded_chunks}')"
    )
    result = subprocess.run([str(py), "-c", code], cwd=ROOT)
    if result.returncode != 0:
        print("ВНИМАНИЕ: не удалось построить базу знаний (см. сообщение выше). Приложение запустится; базу можно построить на странице «База знаний».")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # консоль Windows (cp1251) не должна ронять запуск из-за символа
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ensure_env_file()
    py = ensure_environment()
    ensure_knowledge_base(py)
    print("\nЗапускаю приложение: http://localhost:8501 (остановка — Ctrl+C)\n")
    try:
        return subprocess.call([str(py), "-m", "streamlit", "run", "app.py", *sys.argv[1:]], cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
