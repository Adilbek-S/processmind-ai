"""Единый стиль интерфейса: CSS без внешних библиотек, шапка страницы, индикатор этапов, легенды."""

from __future__ import annotations

import html

import streamlit as st

from processmind.ui.state import STAGES, current_stage
from processmind.visualization.diagrams import COLORS

CSS = """
<style>
:root { --pm-primary:#2563eb; --pm-border:#e5e7eb; --pm-muted:#6b7280; --pm-text:#111827; }
.block-container { padding-top: 3.6rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.pm-hero { background: linear-gradient(135deg,#eff6ff 0%,#f5f3ff 100%); border:1px solid #dbe4ff; border-radius:16px;
           padding:26px 30px; margin-bottom:18px; }
.pm-hero h1 { margin:0 0 6px 0; font-size:2rem; padding:0; }
.pm-hero p { margin:0; color:#374151; font-size:1.02rem; }
.pm-page-title { margin:0 0 2px 0; font-size:1.7rem; font-weight:700; color:var(--pm-text); }
.pm-page-sub { margin:0 0 14px 0; color:var(--pm-muted); font-size:.95rem; }
.pm-card { background:#fff; border:1px solid var(--pm-border); border-radius:12px; padding:14px 16px;
           box-shadow:0 1px 2px rgba(16,24,40,.04); }
.pm-metric .label { color:var(--pm-muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; font-weight:600; }
.pm-metric .value { font-size:1.55rem; font-weight:700; color:var(--pm-text); line-height:1.25; }
.pm-metric .hint { color:var(--pm-muted); font-size:.8rem; }
.pm-section { margin:22px 0 8px 0; font-size:1.15rem; font-weight:700; color:var(--pm-text); }
.pm-section .num { display:inline-flex; width:26px; height:26px; border-radius:50%; background:var(--pm-primary); color:#fff;
                   align-items:center; justify-content:center; font-size:.85rem; margin-right:10px; }
.pm-stepper { display:flex; gap:8px; margin:6px 0 18px 0; flex-wrap:wrap; }
.pm-step { flex:1; min-width:150px; background:#fff; border:1px solid var(--pm-border); border-radius:12px; padding:9px 12px; }
.pm-step .n { font-weight:700; font-size:.75rem; color:var(--pm-muted); text-transform:uppercase; letter-spacing:.05em; }
.pm-step .t { font-weight:600; color:var(--pm-text); font-size:.95rem; }
.pm-step .d { color:var(--pm-muted); font-size:.76rem; }
.pm-step.done { background:#ecfdf5; border-color:#a7f3d0; } .pm-step.done .n { color:#047857; }
.pm-step.current { background:#eff6ff; border-color:var(--pm-primary); box-shadow:0 0 0 2px rgba(37,99,235,.15); }
.pm-step.current .n { color:var(--pm-primary); }
.pm-table { width:100%; border-collapse:separate; border-spacing:0; font-size:.88rem; background:#fff;
            border:1px solid var(--pm-border); border-radius:12px; overflow:hidden; }
.pm-table th { background:#f3f4f6; text-align:left; padding:9px 12px; font-weight:600; color:#374151; }
.pm-table td { vertical-align:top; padding:10px 12px; border-top:1px solid var(--pm-border); color:var(--pm-text); }
.pm-table td.pm-key { font-weight:600; white-space:nowrap; }
.pm-good { color:#047857; font-weight:700; } .pm-bad { color:#b91c1c; font-weight:700; } .pm-neutral { color:var(--pm-muted); }
.pm-note { color:var(--pm-muted); font-size:.78rem; font-weight:400; display:block; }
.pm-notice { background:#fffbeb; border:1px solid #fde68a; border-left:4px solid #f59e0b; border-radius:10px;
             padding:10px 14px; color:#78350f; font-size:.9rem; margin:8px 0 14px 0; }
.pm-legend { display:flex; flex-wrap:wrap; gap:6px 16px; margin:4px 0 8px 0; font-size:.82rem; color:#374151; }
.pm-legend .dot { display:inline-block; width:13px; height:13px; border-radius:4px; border:2px solid; vertical-align:-2px; margin-right:6px; }
.pm-badge { display:inline-block; padding:2px 9px; border-radius:999px; font-size:.75rem; font-weight:600; background:#dbeafe; color:#1d4ed8; }
.pm-badge.off { background:#f3f4f6; color:#6b7280; }
.pm-graph-title { font-weight:700; margin:0 0 4px 0; color:var(--pm-text); }
</style>
"""


def apply_style() -> None:
    """Подключает единый CSS; вызывается в начале каждой страницы."""
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="pm-page-title">{html.escape(title)}</div><div class="pm-page-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def section(number: int, title: str) -> None:
    st.markdown(f'<div class="pm-section"><span class="num">{number}</span>{html.escape(title)}</div>', unsafe_allow_html=True)


def stepper(stage: int | None = None, container=None) -> None:
    """Индикатор этапов сценария; без аргумента этап берётся из состояния сессии.

    `container` — например st.empty(): позволяет перерисовать индикатор после изменения состояния на странице.
    """
    stage = current_stage(st.session_state) if stage is None else stage
    cells = []
    for i, (title, description) in enumerate(STAGES):
        state = "done" if i < stage else ("current" if i == stage else "")
        mark = "✓ " if i < stage else ""
        cells.append(
            f'<div class="pm-step {state}"><div class="n">{mark}Этап {i + 1}</div>'
            f'<div class="t">{html.escape(title)}</div><div class="d">{html.escape(description)}</div></div>'
        )
    (container or st).markdown(f'<div class="pm-stepper">{"".join(cells)}</div>', unsafe_allow_html=True)


def nav_link(path: str, label: str, icon: str = "➡️") -> None:
    """Ссылка на страницу приложения. Если страницу запустили отдельно от приложения, показывает обычный текст."""
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:  # noqa: BLE001 - StreamlitAPIException: путь недоступен вне основного приложения
        st.markdown(f"{icon} {html.escape(label)}")


def notice(text: str) -> None:
    st.markdown(f'<div class="pm-notice">{html.escape(text)}</div>', unsafe_allow_html=True)


def legend(kinds: list[str]) -> None:
    """Легенда цветов диаграммы (те же цвета, что в Graphviz-схемах)."""
    chips = "".join(
        f'<span><span class="dot" style="background:{COLORS[k]["fill"]};border-color:{COLORS[k]["border"]}"></span>{html.escape(COLORS[k]["label"])}</span>'
        for k in kinds
    )
    st.markdown(f'<div class="pm-legend">{chips}</div>', unsafe_allow_html=True)
