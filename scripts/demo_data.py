"""Эталонный демо-процесс и генерация синтетических PDF/PNG по нему.

Запуск: `python -m scripts.demo_data` — создаёт `samples/demo_process.pdf`, `samples/demo_process.png`
и `samples/demo_process_truth.json` (эталон для проверки полноты извлечения).

В эталоне намеренно есть операции без участника и без длительности: по ним видно,
выдумывает ли модель то, чего в документе нет.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

PROCESS_NAME = "Согласование заявки на закупку оборудования"
PROCESS_GOAL = "Обеспечить своевременное согласование и оформление закупки оборудования."
PARTICIPANTS = ["Инициатор", "Менеджер по закупкам", "Финансовый директор", "Бухгалтер"]

# name — каноническое название; sentence — как операция описана в PDF; label — подпись блока на схеме.
# duration_minutes / actor / is_manual == None — в документе НЕ указаны.
TRUTH_STEPS = [
    {
        "name": "Подать заявку на закупку",
        "actor": "Инициатор",
        "duration_minutes": None,
        "is_manual": None,
        "sentence": "Инициатор подаёт заявку на закупку оборудования через внутренний портал.",
        "label": "Подать заявку на закупку",
        "duration_text": None,
        "type_text": None,
    },
    {
        "name": "Проверить комплектность заявки",
        "actor": "Менеджер по закупкам",
        "duration_minutes": 30,
        "is_manual": True,
        "sentence": "Менеджер по закупкам вручную проверяет комплектность заявки (30 минут).",
        "label": "Проверить комплектность заявки",
        "duration_text": "30 мин",
        "type_text": "вручную",
    },
    {
        "name": "Согласовать бюджет",
        "actor": "Финансовый директор",
        "duration_minutes": 120,
        "is_manual": None,
        "sentence": "Финансовый директор согласовывает бюджет закупки (2 часа).",
        "label": "Согласовать бюджет",
        "duration_text": "2 часа",
        "type_text": None,
    },
    {
        "name": "Выбрать поставщика",
        "actor": "Менеджер по закупкам",
        "duration_minutes": None,
        "is_manual": True,
        "sentence": "Менеджер по закупкам вручную выбирает поставщика по результатам сравнения предложений.",
        "label": "Выбрать поставщика",
        "duration_text": None,
        "type_text": "вручную",
    },
    {
        "name": "Заключить договор с поставщиком",
        "actor": None,
        "duration_minutes": 4320,
        "is_manual": None,
        "sentence": "Заключается договор с поставщиком (3 дня).",
        "label": "Заключить договор с поставщиком",
        "duration_text": "3 дня",
        "type_text": None,
    },
    {
        "name": "Оплатить счёт",
        "actor": "Бухгалтер",
        "duration_minutes": 60,
        "is_manual": False,
        "sentence": "Бухгалтер оплачивает счёт; платёж формируется автоматически в банк-клиенте (1 час).",
        "label": "Оплатить счёт",
        "duration_text": "1 час",
        "type_text": "автоматически",
    },
    {
        "name": "Принять оборудование",
        "actor": "Инициатор",
        "duration_minutes": None,
        "is_manual": None,
        "sentence": "Инициатор принимает поставленное оборудование.",
        "label": "Принять оборудование",
        "duration_text": None,
        "type_text": None,
    },
    {
        "name": "Закрыть заявку",
        "actor": "Менеджер по закупкам",
        "duration_minutes": 5,
        "is_manual": False,
        "sentence": "Менеджер по закупкам закрывает заявку; статус в 1С обновляется автоматически (5 минут).",
        "label": "Закрыть заявку",
        "duration_text": "5 мин",
        "type_text": "автоматически",
    },
]

_FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")]
_REGULAR = ["arial.ttf", "DejaVuSans.ttf", "Arial.ttf"]
_BOLD = ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]


def _find_font(candidates: list[str]) -> Path:
    for directory in _FONT_DIRS:
        for name in candidates:
            if (directory / name).exists():
                return directory / name
    raise FileNotFoundError(f"Не найден шрифт с кириллицей среди {candidates}")


def truth_dict() -> dict:
    return {
        "name": PROCESS_NAME,
        "goal": PROCESS_GOAL,
        "participants": PARTICIPANTS,
        "steps": [
            {k: s[k] for k in ("name", "actor", "duration_minutes", "is_manual")} for s in TRUTH_STEPS
        ],
    }


def build_pdf() -> bytes:
    regular, bold = _find_font(_REGULAR), _find_font(_BOLD)
    doc = pymupdf.open()
    page = doc.new_page()  # A4
    page.insert_font(fontname="reg", fontfile=str(regular))
    page.insert_font(fontname="bold", fontfile=str(bold))

    page.insert_textbox(
        pymupdf.Rect(50, 50, 545, 110), f"Регламент: {PROCESS_NAME}", fontname="bold", fontsize=16
    )
    intro = f"Цель процесса: {PROCESS_GOAL}\nУчастники процесса: {', '.join(PARTICIPANTS)}."
    page.insert_textbox(pymupdf.Rect(50, 115, 545, 190), intro, fontname="reg", fontsize=11)
    page.insert_textbox(pymupdf.Rect(50, 195, 545, 220), "Порядок выполнения операций:", fontname="bold", fontsize=12)
    body = "\n\n".join(f"{i}. {s['sentence']}" for i, s in enumerate(TRUTH_STEPS, start=1))
    leftover = page.insert_textbox(pymupdf.Rect(50, 225, 545, 800), body, fontname="reg", fontsize=11)
    if leftover < 0:
        raise RuntimeError("Текст демо-PDF не поместился на страницу")
    doc.subset_fonts()
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def build_png() -> bytes:
    """Схема: два ряда по 4 блока со стрелками, порядок читается слева направо, затем второй ряд."""
    import io

    regular, bold = str(_find_font(_REGULAR)), str(_find_font(_BOLD))
    f_title = ImageFont.truetype(bold, 26)
    f_label = ImageFont.truetype(bold, 17)
    f_small = ImageFont.truetype(regular, 15)

    width, height = 1500, 640
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 25), PROCESS_NAME, font=f_title, fill="black")

    box_w, box_h, gap = 300, 170, 60
    x0, ys = 40, [110, 380]
    centers: list[tuple[int, int, int, int]] = []
    for i, step in enumerate(TRUTH_STEPS):
        row, col = divmod(i, 4)
        x, y = x0 + col * (box_w + gap), ys[row]
        centers.append((x, y, x + box_w, y + box_h))
        d.rounded_rectangle((x, y, x + box_w, y + box_h), radius=12, outline="#1f4e79", width=3, fill="#eaf2fb")
        d.text((x + 12, y + 8), str(i + 1), font=f_label, fill="#1f4e79")
        lines = _wrap(d, step["label"], f_label, box_w - 24)
        ty = y + 34
        for line in lines:
            d.text((x + 12, ty), line, font=f_label, fill="black")
            ty += 22
        extras = [t for t in (step["actor"] and f"Исполнитель: {step['actor']}", step["duration_text"] and f"Срок: {step['duration_text']}", step["type_text"]) if t]
        ty = y + box_h - 12 - 20 * len(extras)
        for line in extras:
            d.text((x + 12, ty), line, font=f_small, fill="#333333")
            ty += 20

    for i in range(len(TRUTH_STEPS) - 1):
        (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = centers[i], centers[i + 1]
        if ay1 == by1:  # в одном ряду
            _arrow(d, (ax2, ay1 + box_h // 2), (bx1, by1 + box_h // 2))
        else:  # переход на второй ряд: вниз, влево, вниз
            mid_y = ay2 + (by1 - ay2) // 2
            d.line([(ax1 + box_w // 2, ay2), (ax1 + box_w // 2, mid_y), (bx1 + box_w // 2, mid_y)], fill="#1f4e79", width=3)
            _arrow(d, (bx1 + box_w // 2, mid_y), (bx1 + box_w // 2, by1))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill="#1f4e79", width=3)
    ex, ey = end
    if start[1] == end[1]:  # горизонтальная
        draw.polygon([(ex, ey), (ex - 14, ey - 8), (ex - 14, ey + 8)], fill="#1f4e79")
    else:  # вертикальная вниз
        draw.polygon([(ex, ey), (ex - 8, ey - 14), (ex + 8, ey - 14)], fill="#1f4e79")


def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    (SAMPLES_DIR / "demo_process.pdf").write_bytes(build_pdf())
    (SAMPLES_DIR / "demo_process.png").write_bytes(build_png())
    (SAMPLES_DIR / "demo_process_truth.json").write_text(
        json.dumps(truth_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Файлы созданы в {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
