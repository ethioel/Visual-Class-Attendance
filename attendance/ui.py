"""Presentation helpers. Theming = one state owner (the 🌓 widget key in
app.py) + two complete variable/surface blocks (_VARS_DARK / _VARS_LIGHT).
Each block overrides BOTH the CSS variables AND the actual widget surfaces
(dataframes, inputs, selects, buttons, code blocks), so light and dark are
both fully rendered — no half-applied themes."""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


def theme_is_dark() -> bool:
    try:
        return str(getattr(st.context.theme, "type", "light")).lower() == "dark"
    except Exception:
        return False


# ═══════════════ DARK: variables + native widget surfaces ═══════════════
_VARS_DARK = """
:root{--att-bg:#0B1220!important;--att-panel:#111827!important;
--att-text:#E5E7EB!important;--att-muted:#94A3B8!important;
--att-border:rgba(148,163,184,.25)!important;--att-input:#0F172A!important;}

/* page + main surfaces */
[data-testid="stAppViewContainer"]{background:var(--att-bg);}
[data-testid="stHeader"]{background:transparent;}
section.stMain, section.stMain > div{background:var(--att-bg);}
[data-testid="stMarkdownContainer"],.stApp{color:var(--att-text);}
h1,h2,h3,h4{color:#F1F5F9;}

/* sidebar keeps the branded gradient (deliberate, both themes) */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F172A,#1E293B)!important;}
[data-testid="stSidebar"] *{color:#E2E8F0;}

/* cards & metrics */
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--att-panel);
  border:1px solid var(--att-border);border-radius:14px;box-shadow:none;}
[data-testid="stMetric"]{background:var(--att-panel);border:1px solid var(--att-border);
  border-radius:14px;}
[data-testid="stMetricValue"]{color:#F1F5F9;}

/* inputs & selects */
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,
[data-baseweb="select"]>div,[data-testid="stNumberInput"] input,
[data-testid="stTimeInput"] input,[data-testid="stDateInput"] input{
  background:var(--att-input)!important;color:var(--att-text)!important;
  border:1px solid #334155!important;}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="popover"] li{
  background:var(--att-input)!important;color:var(--att-text)!important;}

/* buttons */
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-tertiary"]{
  border-color:#334155!important;color:var(--att-text)!important;}

/* dataframes / editors */
[data-testid="stDataFrame"]{border-color:#1F2937;}
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"]{
  background:#0B1220;}
#view-dataframe, [data-testid="stDataFrame"] *{color-scheme:dark;}

/* misc */
[data-testid="stCaptionContainer"],.muted{color:var(--att-muted);}
[data-testid="stExpander"],hr{border-color:#1F2937;}
[data-testid="stStatusWidget"],[data-testid="stSpinner"] div{color:var(--att-text);}
.pill-muted{background:rgba(128,128,128,.25);color:#CBD5E1;}
.sugg{background:#1E1B4B;color:#C7D2FE;border-color:#3730A3;}
.sugg.ok{background:#052E16;color:#86EFAC;border-color:#166534;}
::-webkit-scrollbar-thumb{background:#334155;}
"""

# ═══════════════ LIGHT: variables + native widget surfaces ═══════════════
_VARS_LIGHT = """
:root{--att-bg:#FFFFFF!important;--att-panel:#FFFFFF!important;
--att-text:#0F172A!important;--att-muted:#64748B!important;
--att-border:rgba(15,23,42,.14)!important;--att-input:#FFFFFF!important;}

/* page + main surfaces — forces light over the dark config base */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] section.stMain,
[data-testid="stAppViewContainer"] section.stMain > div{background:var(--att-bg)!important;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMarkdownContainer"],.stApp{color:var(--att-text)!important;}
h1,h2,h3,h4{color:#0F172A!important;}

/* cards & metrics */
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--att-panel)!important;
  border:1px solid var(--att-border)!important;box-shadow:0 1px 3px rgba(15,23,42,.06);}
[data-testid="stMetric"]{background:#FFFFFF!important;border:1px solid var(--att-border)!important;}
[data-testid="stMetricValue"]{color:#0F172A!important;}
[data-testid="stMetricLabel"] p{color:#64748B!important;}

/* inputs & selects — the surfaces that stayed dark before */
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,
[data-baseweb="select"]>div,[data-testid="stNumberInput"] input,
[data-testid="stTimeInput"] input,[data-testid="stDateInput"] input{
  background:#FFFFFF!important;color:#0F172A!important;
  border:1px solid #CBD5E1!important;}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="popover"] li{
  background:#FFFFFF!important;color:#0F172A!important;}

/* buttons */
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-tertiary"]{
  border-color:#CBD5E1!important;color:#0F172A!important;}

/* dataframes / editors — force light grid */
[data-testid="stDataFrame"]{border-color:#E2E8F0;}
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"]{background:#FFFFFF;}
#view-dataframe,[data-testid="stDataFrame"] *{color-scheme:light;}

/* misc */
[data-testid="stCaptionContainer"],.muted{color:#64748B!important;}
[data-testid="stExpander"],hr{border-color:#E2E8F0;}
.pill-muted{background:rgba(128,128,128,.14);color:#334155;}
.sugg{background:#EEF2FF;color:#3730A3;border-color:#C7D2FE;}
.sugg.ok{background:#DCFCE7;color:#166534;border-color:#BBF7D0;}
::-webkit-scrollbar-thumb{background:#CBD5E1;}
"""

# ---- custom-element styles (variable-driven; theme-agnostic) -------------
_CSS = """
footer{visibility:hidden;}

/* sidebar keeps the branded dark gradient in BOTH modes (deliberate) */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F172A,#1E293B);border-right:none;}
[data-testid="stSidebar"] *{color:#E2E8F0;}
[data-testid="stSidebar"] [data-testid="stMetric"]{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.10);box-shadow:none;}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p{color:#94A3B8;}
[data-testid="stSidebar"] [data-testid="stMetricValue"]{color:#E2E8F0;}

div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px;
  border:1px solid var(--att-border);}
[data-testid="stMetric"]{border-radius:14px;padding:12px 16px;}
[data-testid="stMetricLabel"] p{font-size:.72rem;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;}
[data-testid="stImage"] img,[data-testid="stTextInput"] input{border-radius:10px;}
[data-testid^="stBaseButton"]{border-radius:10px;font-weight:600;}
[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#4F46E5,#6366F1);
  border:none;box-shadow:0 2px 10px rgba(79,70,229,.35);}
[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.07);}

.pill{display:inline-block;padding:.16rem .55rem;border-radius:999px;
  font-size:.70rem;font-weight:700;vertical-align:middle;}
.pill-present{background:rgba(34,197,94,.16);color:#22C55E;}
.pill-late{background:rgba(245,158,11,.16);color:#F59E0B;}
.pill-absent{background:rgba(239,68,68,.16);color:#EF4444;}
.pill-excused{background:rgba(59,130,246,.16);color:#3B82F6;}
.pill-admin{background:rgba(99,102,241,.18);color:#818CF8;}
.pill-teacher{background:rgba(16,185,129,.16);color:#34D399;}
.pill-guest{background:rgba(245,158,11,.18);color:#FBBF24;}

.sec{display:flex;align-items:center;gap:.5rem;font-weight:800;
  font-size:1.15rem;margin:.3rem 0 1rem;}
.sec .ico{display:inline-flex;width:34px;height:34px;border-radius:10px;
  background:var(--primary-color);color:#fff;align-items:center;justify-content:center;}
.muted{font-size:.85rem;}
.sugg{font-size:1.02rem;font-weight:700;padding:.6rem .9rem;border-radius:12px;
  border:1px solid var(--att-border);}
.sugg.ok{}

.userchip{display:flex;gap:.6rem;align-items:center;margin:.6rem 0;}
.avatar{width:38px;height:38px;border-radius:50%;
  background:linear-gradient(135deg,#4F46E5,#8B5CF6);color:#fff;font-weight:800;
  display:flex;align-items:center;justify-content:center;}
.uname{font-weight:700;font-size:.9rem;}

.hero{background:radial-gradient(1200px 400px at 10% -10%,rgba(255,255,255,.18),transparent),
  linear-gradient(135deg,#312E81,#4F46E5 60%,#7C3AED);
  border-radius:20px;padding:2rem 2.2rem;margin-bottom:1rem;}
.hero h1{color:#fff;font-size:1.7rem;margin:0 0 .4rem;}
.hero p{color:#C7D2FE;margin:0;font-size:.95rem;}
.hero .chips span{display:inline-block;background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.25);color:#E0E7FF;border-radius:999px;
  padding:.2rem .7rem;font-size:.72rem;font-weight:600;margin:.5rem .35rem 0 0;}

::-webkit-scrollbar{width:9px;height:9px;}
"""


def inject_css(hide_sidebar: bool = False, dark: "bool | None" = None) -> None:
    """dark=None → follow Streamlit's native theme; True/False → force the
    matching complete block (variables + widget surfaces)."""
    d = theme_is_dark() if dark is None else dark
    css = (_VARS_DARK if d else _VARS_LIGHT) + _CSS \
        + (_HIDE_SIDEBAR if hide_sidebar else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def section(icon: str, title: str) -> None:
    st.markdown(f'<div class="sec"><span class="ico">{icon}</span>{title}</div>',
                unsafe_allow_html=True)


def pill(text: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


STATUS_EMOJI = {"Present": "✅ Present", "Late": "⏰ Late", "Absent": "🚫 Absent",
                "Excused": "📝 Excused"}


def user_chip(user: dict) -> None:
    initials = "".join(w[0] for w in user["username"].replace("-", " ").split()[:2]).upper() or "U"
    st.markdown(f'<div class="userchip"><div class="avatar">{initials}</div><div>'
                f'<div class="uname">{user["username"]}</div>'
                f'{pill(user["role"], user["role"])}</div></div>',
                unsafe_allow_html=True)


def empty_state(icon: str, title: str, hint: str = "") -> None:
    st.markdown(f'<div style="text-align:center;padding:2.2rem 1rem;">'
                f'<div style="font-size:2.2rem;">{icon}</div>'
                f'<div style="font-weight:700;margin:.4rem 0;">{title}</div>'
                f'<div class="muted">{hint}</div></div>', unsafe_allow_html=True)


def empty_data_hero(title: str, lines: list, cta_label: str = "",
                    cta_page: str = "") -> None:
    items = "".join(f"<li style='margin:.3rem 0'>{l}</li>" for l in lines)
    cta = (f'<span style="display:inline-block;margin-top:1rem;'
           f'background:linear-gradient(135deg,#4F46E5,#6366F1);color:#fff;'
           f'padding:.5rem 1.1rem;border-radius:10px;font-weight:700;">'
           f'{cta_label}</span>' if cta_label else "")
    st.markdown(
        f'<div style="text-align:center;padding:2.6rem 1rem;">'
        f'<div style="font-size:2.6rem">🚀</div>'
        f'<div style="font-size:1.25rem;font-weight:800;margin:.5rem 0;">{title}</div>'
        f'<div class="muted" style="max-width:540px;margin:0 auto;text-align:left;">'
        f"<ol style='padding-left:1.3rem'>{items}</ol></div>"
        f"{cta}</div>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown('<div class="hero"><h1>🪪 Visual Attendance</h1>'
                "<p>Face-first classroom operations — enroll once, scan the "
                "room, done.</p>"
                '<div class="chips"><span>Multi-sample enrollment</span>'
                '<span>Roster-scoped matching</span>'
                "<span>🛡️ Liveness check</span><span>Dark mode</span></div></div>",
                unsafe_allow_html=True)


def render_flash() -> None:
    f = st.session_state.pop("flash", None)
    if f:
        {"success": st.success, "error": st.error,
         "info": st.info, "warning": st.warning}[f[0]](f[1])


def flash(kind: str, msg: str) -> None:
    st.session_state["flash"] = (kind, msg)


def suggestion_box(text: str) -> None:
    st.markdown(f'<div class="sugg {"ok" if text.startswith("✅") else ""}">{text}</div>',
                unsafe_allow_html=True)


def face_thumb(rgb: np.ndarray, box, size: int = 88) -> np.ndarray:
    t, r, b, l = box
    h, w = rgb.shape[:2]
    t, l, b, r = max(0, t), max(0, l), min(h, b), min(w, r)
    crop = rgb[t:b, l:r]
    if crop.size == 0:
        return np.full((size, size, 3), 229, dtype=np.uint8)
    return cv2.resize(crop, (size, size))
