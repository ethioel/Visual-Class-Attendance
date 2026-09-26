import base64
import datetime as _dt
import logging
import os
import time
from io import BytesIO

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    for _k, _v in st.secrets.get("env", {}).items():
        os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass


def _install_fragment_noise_filter() -> None:
    class _F(logging.Filter):
        def filter(self, record):
            m = record.getMessage()
            return ("does not exist anymore" not in m
                    and "Couldn't find fragment" not in m)
    seen = set()

    def attach(lg):
        if id(lg) in seen:
            return
        seen.add(id(lg))
        for h in lg.handlers:
            h.addFilter(_F())
    for name in ["", "streamlit"] + list(logging.root.manager.loggerDict):
        try:
            attach(logging.getLogger(name))
        except Exception:
            pass


_install_fragment_noise_filter()

from camera_input_live import camera_input_live

from attendance.auth import Auth
from attendance.config import Config
from attendance.engine import (annotate, best_match, detect_and_encode,
                               largest_face, recognize)
from attendance.ops import OpsStore
from attendance.store import Store
from attendance.ui import (STATUS_EMOJI, empty_data_hero, empty_state,
                           face_thumb, flash, hero, render_flash, section,
                           suggestion_box, user_chip)

st.set_page_config(page_title="Visual Attendance", page_icon="🪪", layout="wide",
                   initial_sidebar_state="collapsed")

MAIN_DB = os.environ.get("ATT_DB_DIR", "attendance_db")
GUEST_DB = "demo_db"
STATUS_OPTS = ["—", "Present", "Late", "Excused", "Absent"]
WORK_WIDTH = 960
LIVE_PROB = 0.85
VIS_W = 640


@st.cache_resource
def resources(db_dir: str):
    cfg = Config(db_dir=db_dir)
    store, auth = Store(cfg), Auth(cfg)
    tz = store.get_settings().get("timezone")
    if tz:
        cfg.timezone = tz
    return cfg, store, auth


@st.cache_resource
def ops_resource(db_dir: str):
    return OpsStore(Config(db_dir=db_dir))


def to_rgb(upload) -> np.ndarray:
    arr = np.frombuffer(upload.getvalue(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode image — retake.")
        st.stop()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _raw_to_bgr(raw):
    """Convert any camera_input_live return type to BGR ndarray (None if empty)."""
    if raw is None:
        return None
    if isinstance(raw, str):                       # data URL
        raw = BytesIO(base64.b64decode(raw.split(",", 1)[1]))
    elif isinstance(raw, (bytes, bytearray)):
        raw = BytesIO(raw)
    img = raw if isinstance(raw, Image.Image) else Image.open(raw)
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def camera_frame_bgr():
    return _raw_to_bgr(camera_input_live())


def to_working(rgb, max_w: int = WORK_WIDTH):
    h, w = rgb.shape[:2]
    if w > max_w:
        s = max_w / w
        return cv2.resize(rgb, (max_w, int(h * s))), 1.0 / s
    return rgb, 1.0


def live_suggestion(n_faces, brightness, sharpness):
    if brightness < 55:
        return "☀️ Too dark — face a light"
    if brightness > 210:
        return "🌤️ Too bright — avoid direct light"
    if sharpness < 30:
        return "📸 Hold still — blurry"
    if n_faces == 0:
        return "🙂 No face in frame — step into view"
    if n_faces > 1:
        return "👥 More than one face — keep one person in frame"
    return "✅ Perfect — hold still"


def _app_base_url() -> str:
    try:
        h = st.context.headers
        host = h.get("Host") or h.get("host") or ""
        proto = h.get("X-Forwarded-Proto") or (
            "https" if host.endswith(".streamlit.app") else "http")
        if host:
            return f"{proto}://{host}"
    except Exception:
        pass
    return ""


# ---------------- theming (self-contained) ----------------
def _theme_is_dark() -> bool:
    try:
        return str(getattr(st.context.theme, "type", "light")).lower() == "dark"
    except Exception:
        return False


def _inject_css(hide_sidebar: bool = False, dark=None) -> None:
    d = _theme_is_dark() if dark is None else dark
    try:
        from attendance.ui import _CSS, _DARK, _HIDE_SIDEBAR
    except Exception:
        _CSS = _DARK = _HIDE_SIDEBAR = ""
    css = _CSS + (_DARK if d else "") + (_HIDE_SIDEBAR if hide_sidebar else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _dark_toggle(right: bool = True) -> bool:
    c1, c2 = st.columns([9, 1]) if right else st.columns([1, 2])
    with c2:
        dark = st.toggle("🌓", value=st.session_state.get(
            "dark_mode", _theme_is_dark()), key="dark_toggle", help="Dark mode")
    st.session_state["dark_mode"] = dark
    return dark


def _dark_active() -> bool:
    return bool(st.session_state.get("dark_mode", _theme_is_dark()))


def _render_heatmap(cid: str, month: str):
    df = store.heatmap(cid, month)
    if df.empty:
        st.caption("No sessions recorded in this month.")
        return
    dates = sorted(df.Date.unique())
    day_of = {d: d[-2:] for d in dates}
    df["Day"] = df.Date.map(day_of)
    names = df.Name.drop_duplicates().tolist()
    no_rec = "#334155" if _dark_active() else "#E5E7EB"
    chart = (alt.Chart(df).mark_rect(stroke="#94A3B8", strokeWidth=0.3)
             .encode(x=alt.X("Day:O", sort=[day_of[d] for d in dates],
                             axis=alt.Axis(labelAngle=0, title=None,
                                           labelFontSize=9)),
                     y=alt.Y("Name:N", sort=names, title=None),
                     color=alt.Color("Status:N", legend=alt.Legend(
                         orient="bottom", title=None),
                         scale=alt.Scale(
                             domain=["Present", "Late", "Absent", "Excused",
                                     "No record"],
                             range=["#16A34A", "#F59E0B", "#EF4444",
                                    "#3B82F6", no_rec])),
                     tooltip=["Name", "Date", "Status"])
             .properties(height=min(26 * len(names) + 40, 520)))
    st.altair_chart(chart, width="stretch")


def capture_section(pid: str, name: str, store: Store, cfg: Config,
                    class_targets=None, on_saved=None):
    """Hands-free enrollment: one in-fragment camera widget at attendance-like
    width. Outside containers are st.empty() (claimed on the full run, filled
    by the fragment) — Streamlit's required pattern. Completion state is read
    INSIDE the fragment, so it always renders."""
    class_targets = class_targets or []
    cap = st.session_state.setdefault(f"cap::{pid}", {"samples": [], "thumbs": []})
    done_key = f"autodone::{pid}"
    mode = st.radio("Capture mode",
                    ["🎥 Auto-capture (hands-free)", "📷 Manual snapshots"],
                    horizontal=True, key=f"cmode::{pid}")

    if mode.startswith("🎥"):
        cam_col, side_col = st.columns([3, 2])
        # Claim slots on the FULL run so the fragment can write into them.
        cam_slot = cam_col.empty()
        side_slot = side_col.empty()

        @st.fragment(run_every=3.0)
        def _auto():
            if st.session_state.get(done_key, False):
                cam_slot.container(border=True).markdown(
                    "✅ **All samples captured** — review them on the right, "
                    "then press **Save**.")
                with side_slot.container():
                    if cap["thumbs"]:
                        st.markdown("**Captured samples**")
                        for i, tmb in enumerate(cap["thumbs"]):
                            c1, c2 = st.columns([3, 1])
                            c1.image(tmb, width="stretch")
                            if c2.button("✕", key=f"dx::{pid}::{i}",
                                         help="Delete this sample"):
                                cap["samples"].pop(i)
                                cap["thumbs"].pop(i)
                                st.rerun(scope="fragment")
                    else:
                        st.caption("No samples yet.")
                    if st.button("↺ Capture more", key=f"more::{pid}",
                                 width="stretch"):
                        st.session_state[done_key] = False
                        st.rerun(scope="fragment")
                return

            with cam_slot.container():
                try:
                    raw = camera_input_live(key=f"camw::{pid}")
                except Exception as e:
                    st.error(f"Camera error: {str(e)[:160]}")
                    return
                if raw is None:
                    st.info("Starting camera… allow the permission pop-up "
                            "(once).")
                    return
            frame = _raw_to_bgr(raw)
            if frame is None:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright = float(gray.mean())
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            try:
                locs, encs = detect_and_encode(rgb, scale=1.0, num_jitters=1)
            except Exception as e:
                st.error(f"Recognition error: {str(e)[:160]}")
                return

            n = len(cap["samples"])
            if n < cfg.n_samples:
                good = (len(locs) == 1 and 55 <= bright <= 210 and sharp >= 30)
                if good and time.time() - st.session_state.get(
                        f"lc::{pid}", 0) > 2.0:
                    cap["samples"].append(encs[0])
                    cap["thumbs"].append(face_thumb(rgb, locs[0], 96))
                    st.session_state[f"lc::{pid}"] = time.time()
                    n += 1
                    st.toast(f"Sample {n}/{cfg.n_samples} captured", icon="📸")
            else:
                st.session_state[done_key] = True
                st.rerun(scope="fragment")

            with cam_slot.container():
                suggestion_box(live_suggestion(len(locs), bright, sharp))
                vis = cv2.cvtColor(annotate(
                    cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                    [(l, "ok" if len(locs) == 1 else "?", True, 0.0)
                     for l in locs]), cv2.COLOR_BGR2RGB)
                st.image(cv2.resize(vis, (VIS_W, int(VIS_W * vis.shape[0]
                                                       / max(1, vis.shape[1])))),
                         width="stretch")
                st.progress(min(n / cfg.n_samples, 1.0),
                            text=f"{n}/{cfg.n_samples} samples captured")

            with side_slot.container():
                st.markdown("**Captured samples**")
                if cap["thumbs"]:
                    for i, tmb in enumerate(cap["thumbs"]):
                        c1, c2 = st.columns([3, 1])
                        c1.image(tmb, width="stretch")
                        if c2.button("✕", key=f"sx::{pid}::{i}",
                                     help="Delete this sample"):
                            cap["samples"].pop(i)
                            cap["thumbs"].pop(i)
                            st.rerun(scope="fragment")
                else:
                    st.caption("Thumbnails appear here as quality frames are "
                               "auto-captured.")
                with st.container(border=True):
                    st.markdown("**Tips** 📸")
                    st.markdown("• Vary angle & distance between captures\n"
                                "• Keep a single face, good lighting\n"
                                f"• Auto-capture stops at "
                                f"{cfg.n_samples}/{cfg.n_samples}")

        _auto()
    else:
        left, right = st.columns([3, 2])
        with left:
            shot = st.camera_input("Capture a sample — vary angle & lighting",
                                   key=f"man::{pid}::{len(cap['samples'])}")
            if shot is not None:
                rgb = to_rgb(shot)
                locs, encs = detect_and_encode(rgb, scale=1.0, num_jitters=1)
                box = largest_face(locs)
                if box is None:
                    st.error("No face detected — improve lighting and retake.")
                else:
                    cap["samples"].append(encs[locs.index(box)])
                    cap["thumbs"].append(face_thumb(rgb, box, 96))
                    st.rerun()
        with right:
            if cap["thumbs"]:
                for i, tmb in enumerate(cap["thumbs"]):
                    c1, c2 = st.columns([3, 1])
                    c1.image(tmb, width="stretch")
                    if c2.button("✕", key=f"mx::{pid}::{i}",
                                 help="Delete this sample"):
                        cap["samples"].pop(i)
                        cap["thumbs"].pop(i)
                        st.rerun()

    if cap["samples"]:
        st.progress(min(len(cap["samples"]) / cfg.n_samples, 1.0),
                    text=f"{len(cap['samples'])}/{cfg.n_samples} samples captured")
        if st.button(f"💾 Save {len(cap['samples'])} sample(s) for {name or pid}",
                     type="primary", width="stretch"):
            store.enroll(pid, name or pid, cap["samples"])
            added = [t for t in class_targets if store.add_to_class(t, pid)]
            for k in (f"cap::{pid}", f"autodone::{pid}"):
                st.session_state.pop(k, None)
            msg = f"Saved **{name or pid}** with {len(cap['samples'])} samples."
            if added:
                msg += f" Added to: {', '.join(added)}."
            flash("success" if added else "warning", msg)
            if on_saved:
                on_saved(pid)
            st.rerun()


# ================= INVITE CLASS PORTAL (pre-login) =================
_invite = st.query_params.get("invite")
if _invite and "user" not in st.session_state:
    dark = _dark_toggle(right=False)
    _inject_css(hide_sidebar=True, dark=dark)
    icfg, istore, _ = resources(MAIN_DB)
    icid = istore.class_by_invite(_invite)
    render_flash()
    if not icid:
        empty_state("🔗", "Invalid invite link",
                    "It may have been regenerated — ask your teacher for a "
                    "fresh link.")
    else:
        icls = istore.load_classes()[icid]
        if st.session_state.pop("invite_done", False):
            st.title(f"🪪 {icls['name']}")
            st.success("🎉 You're enrolled! Your teacher can now see you in "
                       "the class list.")
            st.caption("You can close this page.")
            st.stop()
        st.title(f"🪪 Join {icls['name']}")
        st.caption("Enter your details and capture your face samples — after "
                   "saving, you're automatically on the class list.")
        c1, c2 = st.columns(2)
        pid = c1.text_input("Student ID (if you don't have one, make up a "
                            "unique one, e.g. abebe-k)", key="inv_pid")
        name = c2.text_input("Full name", key="inv_name")
        if pid.strip() and name.strip():
            if pid.strip() in istore.load_people():
                st.info("You're already registered — new samples will be added "
                        "to your profile.")
            capture_section(pid.strip(), name.strip(), istore, icfg,
                            class_targets=[icid],
                            on_saved=lambda p: st.session_state.__setitem__(
                                "invite_done", True))
    st.stop()


# ================= LOGIN =================
if "user" not in st.session_state:
    dark = _dark_toggle(right=False)
    _inject_css(hide_sidebar=True, dark=dark)
    _, _, auth0 = resources(MAIN_DB)
    c1, c2, c3 = st.columns([1, 2.2, 1])
    with c2:
        st.write("")
        hero()
        with st.container(border=True):
            st.markdown("**Sign in**")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Sign in", type="primary", width="stretch"):
                usr = auth0.verify(u, p)
                if usr:
                    st.session_state.user = usr
                    st.rerun()
                st.error("Wrong username or password.")
            if os.environ.get("ATT_GUEST_MODE") == "1":
                if st.button("👀 Continue as guest (demo sandbox)", width="stretch"):
                    st.session_state.user = {"username": "guest", "role": "guest"}
                    st.rerun()
        st.caption("Teachers get accounts from the admin. Students never sign in.")
    st.stop()

# ================= SESSION =================
user = st.session_state.user
guest = user["role"] == "guest"
dark = _dark_toggle()
_inject_css(dark=dark)
cfg, store, auth = resources(GUEST_DB if guest else MAIN_DB)
ops = ops_resource(GUEST_DB if guest else MAIN_DB)
render_flash()
store.actor = user["username"]

if guest and "DEMO" not in store.load_classes():
    store.create_class("DEMO", "Guest demo", "guest")

classes = (store.load_classes() if user["role"] in ("admin", "guest")
           else store.classes_of(user["username"]))
people = store.load_people()
enc_all = store.load_encodings()
ACTOR = user["username"]


def scope_records() -> pd.DataFrame:
    df = store.records_df()
    return df[df.Class.isin(classes)] if (classes and not df.empty) else df


def rate_table(cid: str) -> pd.DataFrame:
    r = store.attendance_rates(cid)
    if not r.empty:
        r["Rate"] = pd.to_numeric(r["Rate"].astype(str).str.rstrip("%"), errors="coerce")
    return r


# ================= DASHBOARD =================
def page_dashboard():
    section("📊", "Dashboard")
    if guest:
        st.info("Demo sandbox — everything resets when the app sleeps.", icon="🧪")

    if not people and not classes and scope_records().empty:
        empty_data_hero(
            "Welcome — let's set up your first class",
            ["Create a class (🏫 Classes → ➕ Create class).",
             "Share its invite link, or enroll students yourself (➕ Enroll).",
             "Start a live session in ✅ Take attendance — students walk in "
             "and get marked automatically."])
        return

    @st.fragment(run_every="60s")
    def live_panel():
        df = scope_records()
        t = df[df.Date == store._today()] if not df.empty else df
        pres = t[t.Status.isin(["Present", "Late"])]["ID"].nunique() if not t.empty else 0
        late = t[t.Status == "Late"]["ID"].nunique() if not t.empty else 0
        m = st.columns(4)
        m[0].metric("Enrolled faces", len(people))
        m[1].metric("Classes", len(classes))
        m[2].metric("Present today", pres)
        m[3].metric("Late today", late)
        st.caption("⟳ auto-refreshes every 60s")

    live_panel()

    if classes:
        cid0 = st.selectbox("Class schedule", list(classes),
                            format_func=lambda c: classes[c]["name"],
                            label_visibility="collapsed", key="dash_sched")
        now_s, next_s = ops.now_and_next(cid0)
        n1, n2, n3 = st.columns(3)
        n1.metric("Room", ops.room_of(cid0))
        n2.metric("Now", f"{now_s['start']}–{now_s['end']}" if now_s else "—")
        n3.metric("Next", f"{next_s['day']} {next_s['start']}" if next_s else "—")

    d = scope_records()
    dates = sorted(d.Date.unique())[-14:] if not d.empty else []
    cl, ch = st.columns([2, 3])
    with cl:
        st.markdown("**Today at a glance**")
        if classes:
            cid = st.selectbox("Class", list(classes),
                               format_func=lambda c: classes[c]["name"],
                               label_visibility="collapsed")
            t = store.today_df(cid)
            smap = {r.ID: (r.Status, r.Time) for r in t.itertuples()} if not t.empty else {}
            rows = [{"ID": p, "Name": people.get(p, {}).get("name", p),
                     "Status": STATUS_EMOJI.get(smap.get(p, ("—",))[0], "· Not scanned"),
                     "Time": smap.get(p, ("", ""))[1] or "—"}
                    for p in classes[cid].get("students", [])]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            empty_state("🏫", "No classes yet", "Create one under Classes.")
    with ch:
        st.markdown("**Last 14 days**")
        if dates:
            dd = d[d.Date.isin(dates)]
            p = dd.pivot_table(index="Date", columns="Status", values="ID",
                               aggfunc="count", fill_value=0).reindex(dates, fill_value=0)
            m = p.reset_index().melt("Date", var_name="Status", value_name="Students")
            colors = {"Present": "#16A34A", "Late": "#F59E0B", "Absent": "#EF4444"}
            domain = [s for s in ["Present", "Late", "Absent"] if s in m.Status.unique()]
            chart = (alt.Chart(m).mark_bar(cornerRadius=3)
                     .encode(x=alt.X("Date:O", axis=alt.Axis(labelAngle=-40, title=None,
                                                             labelColor="#64748B")),
                             y=alt.Y("Students:Q", axis=alt.Axis(title=None,
                                                                 labelColor="#64748B")),
                             color=alt.Color("Status:N", scale=alt.Scale(
                                 domain=domain, range=[colors[s] for s in domain])),
                             tooltip=["Date", "Status", "Students"])
                     .properties(height=280))
            st.altair_chart(chart, width="stretch")
        else:
            empty_state("📉", "No records yet",
                        "Run your first live session — the chart fills itself.")


# ================= ATTENDANCE =================
def _class_readiness_card(cid: str, cls: dict):
    students = cls.get("students", [])
    if not students:
        with st.container(border=True):
            st.warning("No students in this class yet.")
            candidates = [p for p in people if p not in students]
            if candidates:
                pick = st.multiselect(
                    "Add existing students to this class", candidates,
                    format_func=lambda p: f"{people[p].get('name', p)} ({p})",
                    key=f"quickadd::{cid}")
                if pick and st.button("Add to class",
                                      key=f"quickaddb::{cid}", type="primary"):
                    for p in pick:
                        store.add_to_class(cid, p)
                        store._audit(ACTOR, "add_to_class", f"{cid}/{p}")
                    st.rerun()
            if PAGES.get("Enroll student"):
                st.page_link(PAGES["Enroll student"],
                             label="➕ …or enroll a new student (select this class)",
                             icon="➕")
            if PAGES.get("Classes"):
                st.page_link(PAGES["Classes" if user["role"] == "admin"
                                   else "My classes"],
                             label="🔗 …or share the class invite link", icon="🔗")
        return
    missing = [p for p in students if p not in enc_all]
    if missing:
        names = [people[p].get("name", p) for p in missing]
        shown = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        with st.container(border=True):
            st.warning(f"**{len(missing)} of {len(students)}** students don't "
                       f"have face samples yet: {shown}")
            if PAGES.get("Students"):
                st.page_link(PAGES["Students"],
                             label="📷 Capture their face samples now", icon="📷")
    else:
        st.caption(f"👥 **{len(students)}** students ready for recognition.")


def page_attendance():
    section("✅", "Take attendance")
    if not classes:
        empty_data_hero(
            "No classes yet",
            ["Create a class (🏫 Classes → ➕ Create class).",
             "Add students: share the invite link, enroll manually, or "
             "import a roster (👥 Students).",
             "Come back here and start a live session."])
        return
    cid = st.selectbox("Class", list(classes),
                       format_func=lambda c: f"{classes[c]['name']} · {c}")
    cls = classes[cid]

    cL1, cL2, cL3 = st.columns([2, 2, 3])
    cur = cls.get("late_after") or ""
    hh, mm = (cur.split(":") + ["00"])[:2] if cur else ("09", "00")
    try:
        late_val = cL2.time_input("Late after (marks from this time = Late)",
                                  value=_dt.time(int(hh), int(mm)),
                                  key=f"late::{cid}")
        late_str = late_val.strftime("%H:%M")
    except Exception:
        late_str = cur
    if late_str != cur and cL1.button("💾 Save late time", width="stretch"):
        store.update_class(cid, {"late_after": late_str})
        cls["late_after"] = late_str
        store._audit(ACTOR, "set_late", f"{cid}={late_str}")
        flash("success", f"Late time set to **{late_str}**.")
        st.rerun()
    cL3.caption(f"Current: **{cur or 'off (all marks = Present)'}** · Room: "
                f"**{ops.room_of(cid)}**")

    enc = {p: enc_all[p] for p in cls.get("students", []) if p in enc_all}
    _class_readiness_card(cid, cls)
    if not enc:
        return

    with st.popover("⚙️ Scan settings"):
        tolerance = st.slider("Match tolerance (lower = stricter)", 0.60, 1.40,
                              cfg.tolerance, 0.01)
        liveness = st.slider("Liveness strictness (0 = off)", 0.0, 1.0, 0.35, 0.05,
                             help="Requires visible motion between frames — "
                             "defeats holding up a still photo.")

    mode = st.radio("Scan mode", ["🎥 Live auto-scan (hands-free)", "📷 Single photo"],
                    horizontal=True, key="scan_mode")

    if mode.startswith("🎥"):
        _live_session(cid, cls, enc, tolerance, liveness)
    else:
        _single_photo(cid, cls, enc, tolerance)

    st.divider()
    st.markdown("**✏️ Today's statuses — edit or delete marks**")
    t = store.today_df(cid)
    smap = {r.ID: r.Status for r in t.itertuples()} if not t.empty else {}
    base = pd.DataFrame([{"ID": p, "Name": people.get(p, {}).get("name", p),
                          "Status": smap.get(p, "—")}
                         for p in cls.get("students", [])])
    if base.empty:
        st.caption("This class has no students yet.")
    else:
        st.caption("Set a student to **—** to delete today's mark.")
        ed = st.data_editor(base, disabled=["ID", "Name"], hide_index=True,
                            width="stretch", key=f"ed::{cid}",
                            column_config={"Status": st.column_config.SelectboxColumn(
                                "Status", options=STATUS_OPTS)})
        if st.button("💾 Apply changes"):
            n = 0
            for (_, a), (_, b) in zip(base.iterrows(), ed.iterrows()):
                if a["Status"] != b["Status"]:
                    store.set_status(a["ID"], a["Name"],
                                     None if b["Status"] == "—" else b["Status"], cid)
                    store._audit(ACTOR, "set_status",
                                 f"{cid}/{a['ID']}={b['Status']}")
                    n += 1
            flash("success" if n else "info",
                  f"Updated {n} record(s)." if n else "No changes to apply.")
            st.rerun()


def _run_recognition(frame_bgr, enc: dict, tolerance: float):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    work, inv = to_working(rgb)
    results = recognize(work, enc, tolerance=tolerance, scale=1.0,
                        prob_threshold=LIVE_PROB)
    vis = cv2.cvtColor(annotate(cv2.cvtColor(work, cv2.COLOR_RGB2BGR), results),
                       cv2.COLOR_BGR2RGB)
    return vis, results, inv


def _apply_marks(results, cid: str, sess: dict, live_ok: bool):
    status = store.status_now(classes[cid].get("late_after"))
    new_marks, any_unknown = [], False
    for box, label, known, dist in results:
        if known:
            if not live_ok:
                continue
            nm = people.get(label, {}).get("name", label)
            if store.mark(label, nm, status, cid):
                new_marks.append(nm)
                store._audit(ACTOR, "mark_live", f"{cid}/{label}={status}")
            if not any(r["ID"] == label for r in sess["log"]):
                sess["log"].insert(0, {"Name": nm, "ID": label,
                                       "Status": STATUS_EMOJI.get(status, status),
                                       "Score": round(dist, 2)})
        else:
            any_unknown = True
    sess["unknown"] += int(any_unknown)
    return new_marks


def _live_session(cid: str, cls: dict, enc: dict, tolerance: float,
                  liveness: float):
    key = f"session::{cid}"
    sess = st.session_state.setdefault(key, {"started": False, "log": [],
                                             "captures": 0, "unknown": 0,
                                             "last_emb": None, "vis": None})
    summary_key = f"summary::{cid}"
    status = store.status_now(cls.get("late_after"))
    if not enc:
        return

    left, right = st.columns([5, 4])

    with left:
        if not sess["started"]:
            if summary_key in st.session_state:
                _render_session_summary(summary_key, cid)
            with st.container(border=True):
                st.markdown("**🎥 Live auto-scan**")
                st.caption(f"{len(enc)} students ready · marks as **{status}**"
                           + (" · 🛡️ liveness ON" if liveness > 0 else "")
                           + f" · Room: **{ops.room_of(cid)}**")
                sess["interval"] = st.select_slider(
                    "Capture interval", options=[3, 5, 8],
                    value=sess.get("interval", 5),
                    format_func=lambda v: f"every {v}s")
                if st.button("▶️ Start live session", type="primary",
                             width="stretch"):
                    sess.update(started=True, log=[], captures=0, unknown=0,
                                last_emb=None, vis=None)
                    st.session_state.pop(summary_key, None)
                    st.rerun()

        @st.fragment(run_every=float(sess.get("interval", 5)))
        def _loop():
            if not sess["started"]:
                st.caption("🔴 Camera off — press ▶️ Start live session.")
                return
            try:
                frame = camera_frame_bgr()
            except Exception as e:
                st.error(f"Camera error: {str(e)[:160]}")
                st.info("If the camera is blocked, use **📷 Single photo** mode.")
                return
            if frame is None:
                st.info("Starting camera… allow the permission pop-up (once).")
                return
            tick = sess["captures"]
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                work, _ = to_working(rgb)
                locs, encs = detect_and_encode(work, scale=1.0,
                                               prob_threshold=LIVE_PROB)
                live_ok = True
                if liveness > 0 and encs:
                    ref = sess.get("last_emb")
                    if ref is not None:
                        d = float(np.linalg.norm(
                            np.asarray(encs[0]) - np.asarray(ref)))
                        live_ok = d >= (0.02 + 0.10 * (1.0 - liveness))
                    sess["last_emb"] = encs[0]
                results = []
                for box, e in zip(locs, encs):
                    pid, dist = best_match(e, enc, tolerance=tolerance)
                    results.append((box, pid, True, dist) if pid
                                   else (box, f"Unknown ({dist:.2f})", False, dist))
                sess["captures"] += 1
            except Exception as e:
                st.error(f"Recognition error: {str(e)[:160]}")
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            suggestion_box(live_suggestion(len(results), float(gray.mean()),
                                           float(cv2.Laplacian(gray, cv2.CV_64F).var())))
            if liveness > 0 and not live_ok:
                st.warning("🛡️ Still frame detected — move naturally; not "
                           "marking from frozen images.")

            has_content = any(r[2] for r in results) or \
                any(not r[2] for r in results)
            if has_content or sess["vis"] is None or tick % 3 == 0:
                vis = cv2.cvtColor(
                    annotate(cv2.cvtColor(work, cv2.COLOR_RGB2BGR), results),
                    cv2.COLOR_BGR2RGB)
                sess["vis"] = cv2.resize(
                    vis, (VIS_W, int(VIS_W * vis.shape[0] / max(1, vis.shape[1]))))
            if sess["vis"] is not None:
                st.image(sess["vis"], width="stretch")

            new_marks = _apply_marks(results, cid, sess, live_ok)
            if new_marks:
                st.toast(f"{', '.join(new_marks)} — marked", icon="🪪")
            if sess["log"]:
                st.dataframe(pd.DataFrame(sess["log"][:6]), hide_index=True,
                             width="stretch")

        _loop()

    with right:
        st.markdown("**Session log**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Marked", len(sess["log"]))
        c2.metric("Unknown", sess["unknown"])
        c3.metric("Captures", sess["captures"])
        unmarked = [p for p in cls.get("students", [])
                    if p in enc and not any(r["ID"] == p for r in sess["log"])]
        if unmarked:
            with st.container(border=True):
                st.markdown("**🙋 Not recognized?** Mark manually (supervised):")
                pick = st.selectbox("Student", unmarked,
                                    format_func=lambda p:
                                    f"{people[p].get('name', p)} ({p})",
                                    key=f"manual::{cid}")
                if st.button("✅ Mark present (manual)", width="stretch",
                             key=f"manualb::{cid}"):
                    confirm_manual(cid, pick, sess)
        if sess["unknown"] >= 3:
            st.warning("Repeated unknown faces — they may not be enrolled, or "
                       "not added to this class.")
        if sess["started"] and st.button("⏹️ End session", type="primary",
                                         width="stretch"):
            st.toast("Ending session…", icon="⏹️")
            st.session_state[summary_key] = {
                "log": list(sess["log"]), "unknown": sess["unknown"],
                "captures": sess["captures"],
                "ended": store.cfg.now().strftime("%H:%M:%S")}
            sess.update(started=False, log=[], captures=0, unknown=0)
            st.rerun()


@st.dialog("Mark this student present?")
def confirm_manual(cid: str, pid: str, sess: dict):
    nm = people.get(pid, {}).get("name", pid)
    status = store.status_now(classes[cid].get("late_after"))
    st.write(f"Teacher-confirmed mark for **{nm}** — recorded as "
             f"**{status}** and logged in the audit trail.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Confirm", type="primary", width="stretch"):
        if store.mark(pid, nm, status, cid):
            sess["log"].insert(0, {"Name": nm, "ID": pid,
                                   "Status": STATUS_EMOJI.get(status, status),
                                   "Score": "manual"})
            store._audit(ACTOR, "mark_manual", f"{cid}/{pid}={status}")
            st.toast(f"{nm} — marked (manual)", icon="🙋")
        st.rerun()


def _render_session_summary(summary_key: str, cid: str):
    s = st.session_state[summary_key]
    with st.container(border=True):
        st.markdown(f"**📋 Session ended** · {s['ended']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Marked this session", len(s["log"]))
        c2.metric("Unknown glimpses", s["unknown"])
        c3.metric("Captures", s["captures"])
        if s["log"]:
            st.dataframe(pd.DataFrame(s["log"]), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        if c1.button("🚫 Mark everyone else Absent", width="stretch",
                     key=f"absent::{summary_key}"):
            n = store.mark_absent_all(cid)
            store._audit(ACTOR, "mark_absent_all", f"{cid} n={n}")
            st.success(f"{n} student(s) marked Absent.")
        if c2.button("✅ Done — hide summary", width="stretch"):
            st.session_state.pop(summary_key, None)
            st.rerun()


@st.dialog("End this class?")
def confirm_end_class(cid: str, sess_key: str = None):
    st.write(f"Everyone in **{classes[cid]['name']}** without a record will be "
             "marked **Absent**. Students marked **Excused** keep their status.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Mark absent", type="primary", width="stretch"):
        n = store.mark_absent_all(cid)
        store._audit(ACTOR, "mark_absent_all", f"{cid} n={n}")
        if sess_key:
            st.session_state.pop(sess_key, None)
        flash("success", f"{n} student(s) marked Absent in {classes[cid]['name']}.")
        st.rerun()


def _single_photo(cid: str, cls: dict, enc: dict, tolerance: float):
    left, right = st.columns([5, 4])
    with left:
        snap = st.camera_input("Scan the room (assisted mode — no liveness)")
        results, rgb, inv = [], None, 1.0
        if snap is not None:
            rgb = to_rgb(snap)
            with st.spinner("Recognizing…"):
                vis, results, inv = _run_recognition(rgb, enc, tolerance)
            st.image(vis, width="stretch")
        if st.button("🚫 End class — mark everyone else absent", width="stretch"):
            confirm_end_class(cid)
    with right:
        st.markdown("**Results**")
        if snap is not None and not results:
            empty_state("😶", "No faces detected", "Move closer / improve lighting.")
        status = store.status_now(cls.get("late_after"))
        fresh = 0
        for box, label, known, dist in results:
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                if rgb is not None:
                    t, r, b, l = [int(v * inv) for v in box]
                    c1.image(face_thumb(rgb, (t, r, b, l)), width="stretch")
                if known:
                    nm = people.get(label, {}).get("name", label)
                    is_new = store.mark(label, nm, status, cid)
                    fresh += is_new
                    if is_new:
                        store._audit(ACTOR, "mark_photo", f"{cid}/{label}={status}")
                    c2.markdown(f"**{nm}**  \n`{label}` · score {dist:.2f}")
                    c2.markdown(f"<span class='pill pill-{status.lower()}'>{status}</span>"
                                + (" <span class='pill pill-muted'>NEW</span>" if is_new else ""),
                                unsafe_allow_html=True)
                else:
                    c2.markdown(f"**Unknown face**  \nscore {dist:.2f}")
        if fresh:
            st.toast(f"{fresh} new mark(s) in {cls['name']}", icon="🪪")


# ================= ENROLL =================
def page_enroll():
    section("➕", "Enroll student")
    if not classes:
        empty_data_hero(
            "Before enrolling: create a class",
            ["Classes group students for recognition — a student is only "
             "recognized in classes they belong to.",
             "Create one in 🏫 Classes (a single click), then come back."])
        return
    c1, c2, c3 = st.columns(3)
    id_mode = c1.radio("ID mode", ["🤖 Automatic", "✍️ Custom"], horizontal=True)
    nxt = f"STU-{len(people) + 1:03d}"
    while nxt in people:
        nxt = f"STU-{int(nxt.split('-')[1]) + 1:03d}"
    if id_mode.startswith("🤖"):
        pid = c2.text_input("Student ID (auto)", value=nxt, disabled=True,
                            key="pid_auto")
    else:
        pid = c2.text_input("Student ID", key="pid")
    name = c3.text_input("Full name", key="pname")
    targets = st.multiselect(
        "Add to class(es) — students are only recognized in classes they belong to",
        list(classes), format_func=lambda c: f"{classes[c]['name']} ({c})",
        key="enroll_targets")
    if not pid.strip() or not name.strip():
        st.caption("Enter an ID and name to start capturing.")
        return
    if st.session_state.pop(f"saved::{pid.strip()}", False):
        st.success(f"**{name.strip()}** saved. Form cleared for the next "
                   "student.")
    capture_section(pid.strip(), name.strip(), store, cfg, class_targets=targets,
                    on_saved=lambda p: st.session_state.__setitem__(
                        f"saved::{p}", True))


# ================= STUDENTS =================
def parse_roster(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"".join(ch for ch in str(c).lower() if ch.isalnum()): c for c in df.columns}
    name_col = next((o for n, o in cols.items() if "name" in n), None)
    id_col = next((o for n, o in cols.items()
                   if n in ("id", "studentid", "sid", "roll", "rollno", "no", "number")
                   or n.endswith("id")), None)
    if name_col is None:
        raise ValueError("No name column found (expected a header containing 'Name').")
    rows, i = [], 0
    for _, row in df.iterrows():
        nm = str(row[name_col]).strip()
        if not nm or nm.lower() == "nan":
            continue
        pid = str(row[id_col]).strip() if id_col is not None else ""
        if not pid or pid.lower() == "nan":
            pid = f"STU-{len(people) + 1 + i:03d}"
            while pid in people or any(r["ID"] == pid for r in rows):
                pid = f"STU-{int(pid.split('-')[1]) + 1:03d}"
        rows.append({"ID": pid, "Name": nm})
        i += 1
    return pd.DataFrame(rows)


def page_students():
    section("👥", "Students")
    tab_import, tab_photos, tab_edit = st.tabs(
        ["📥 Import from file", "📷 Missing face samples", "✏️ Manage students"])

    with tab_import:
        f = st.file_uploader("Upload CSV or Excel with columns **Name** "
                             "(and optional **ID**)",
                             type=["csv", "xlsx", "xls"])
        if f is None:
            st.caption("Tip: edit IDs/names in the preview before importing.")
        else:
            try:
                roster = parse_roster(pd.read_csv(f) if f.name.endswith("csv")
                                      else pd.read_excel(f))
            except Exception as e:
                st.error(f"Could not read file: {e}")
                roster = None
            if roster is not None:
                preview = st.data_editor(roster, num_rows="dynamic",
                                         hide_index=True, width="stretch",
                                         key="import_prev")
                targets = st.multiselect(
                    "Add imported students to class(es)", list(classes),
                    format_func=lambda c: f"{classes[c]['name']} ({c})")
                if st.button(f"📥 Import {len(preview)} student(s)",
                             type="primary"):
                    new = skip = 0
                    for _, r in preview.iterrows():
                        if store.ensure_person(str(r["ID"]).strip(),
                                               str(r["Name"]).strip()):
                            new += 1
                            for tgt in targets:
                                store.add_to_class(tgt, str(r["ID"]).strip())
                            store._audit(ACTOR, "import", f"{tgt}/{r['ID']}")
                        else:
                            skip += 1
                    flash("success", f"Imported {new} new student(s)"
                                     + (f", skipped {skip} existing."
                                        if skip else "."))
                    st.rerun()

    with tab_photos:
        pending = store.pending_samples()
        if not pending:
            empty_state("✅", "Everyone has face samples",
                        f"All students have at least {cfg.n_samples} samples.")
        else:
            pick = st.selectbox(
                "Student", list(pending),
                format_func=lambda p: f"{pending[p].get('name', p)} · {p}")
            ms_targets = st.multiselect(
                "Add this student to class(es) on save", list(classes),
                format_func=lambda c: f"{classes[c]['name']} ({c})",
                key=f"mstargets::{pick}")
            capture_section(pick, pending[pick].get("name", pick), store, cfg,
                            class_targets=ms_targets)

    with tab_edit:
        counts = store.sample_counts()
        rows = []
        for p, info in people.items():
            n = counts.get(p, 0)
            status = ("✅ Ready" if n >= cfg.n_samples else
                      "🟡 Partial" if n > 0 else
                      "🔴 Needs recapture" if info.get("backend") else
                      "⚪ No samples")
            rows.append({"ID": p, "Name": info.get("name", p),
                         "Samples": n, "Status": status})
        base = pd.DataFrame(rows)
        if base.empty:
            empty_state("👥", "No students yet",
                        "Enroll, import, or share a class invite link.")
        else:
            ed = st.data_editor(base, disabled=["ID", "Samples", "Status"],
                                hide_index=True, width="stretch",
                                num_rows="fixed", key="edit_people")
            if st.button("💾 Save name changes"):
                n = 0
                for _, row in ed.iterrows():
                    orig = base.loc[base.ID == row["ID"], "Name"]
                    if not orig.empty and row["Name"] != orig.iloc[0]:
                        store.rename_person(row["ID"], row["Name"])
                        store._audit(ACTOR, "rename", row["ID"])
                        n += 1
                flash("success" if n else "info",
                      f"Renamed {n} student(s)." if n else "No changes.")
                st.rerun()
            st.markdown("**🔄 Re-enroll** — capture fresh samples for an "
                        "existing student.")
            target = st.selectbox(
                "Student", [""] + [f"{r['Name']} · {r['ID']}"
                                   for _, r in base.iterrows()],
                key="reenroll_pick")
            if target and st.button("🔄 Open re-enroll", type="primary"):
                nm, pidx = target.rsplit(" · ", 1)
                reenroll_dialog(pidx)


@st.dialog("🔄 Re-enroll samples")
def reenroll_dialog(pid: str):
    name = people.get(pid, {}).get("name", pid)
    st.caption(f"New samples for **{name}** (`{pid}`) — added on top of "
               "existing ones.")
    capture_section(pid, name, store, cfg,
                    on_saved=lambda p: st.session_state.__setitem__(
                        f"reenroll_done::{p}", True))
    if st.session_state.pop(f"reenroll_done::{pid}", False):
        st.success("Samples saved.")
        st.rerun()


# ================= OPS: OVERSIGHT / TIMETABLES / POLICY =================
def page_oversight():
    section("🏛️", "School oversight")
    if not classes:
        empty_data_hero(
            "Nothing to oversee yet",
            ["Classes appear here once created — with sessions, rates, "
             "rooms and automatic flags."])
        return
    odf = ops.oversight(store, classes, people)
    flagged = odf[~odf.Flags.str.startswith("✅")] if not odf.empty else odf
    c1, c2, c3 = st.columns(3)
    c1.metric("Classes", len(odf))
    c2.metric("Flagged", len(flagged))
    c3.metric("Sessions total", int(odf.Sessions.sum()) if not odf.empty else 0)
    if not flagged.empty:
        st.warning(f"**{len(flagged)} class(es) need attention** — see Flags.")
    st.dataframe(odf, hide_index=True, width="stretch")

    st.divider()
    st.markdown("**⚙️ Attendance policy**")
    pol = ops.policy()
    p1, p2, p3 = st.columns(3)
    min_rate = p1.number_input("Minimum attendance %", 0, 100,
                               int(pol["min_rate"]), key="pol_min")
    stale = p2.number_input("Flag class if no session for (days)", 1, 60,
                            pol["stale_days"], key="pol_stale")
    manual = p3.number_input("Flag student after N manual marks", 1, 50,
                             pol["manual_flag"], key="pol_manual")
    if st.button("💾 Save policy"):
        ops.set_policy(ACTOR, min_rate=min_rate, stale_days=stale,
                       manual_flag=manual)
        flash("success", "Policy saved.")
        st.rerun()

    st.divider()
    st.markdown("**⚠️ Defaulter report** (below the policy minimum)")
    cid_d = st.selectbox("Class", list(classes),
                         format_func=lambda c: f"{classes[c]['name']} · {c}",
                         key="def_class")
    d = ops.defaulters(store, cid_d, people)
    if d.empty:
        st.success("No defaulters in this class.")
    else:
        st.dataframe(d, hide_index=True, width="stretch")

    st.divider()
    st.markdown("**🔎 Anomalies — students frequently marked manually** "
                "(recognition failing for them, or proxy attempts)")
    an = ops.anomalies(store, store.audit_df())
    if an.empty:
        st.caption("No anomaly pattern detected.")
    else:
        st.dataframe(an, hide_index=True, width="stretch")

    st.divider()
    st.markdown("**📄 Institutional report (exportable)**")
    df_all = store.records_df()
    if df_all.empty:
        st.caption("No records yet.")
    else:
        months = sorted({d[:7] for d in df_all.Date.unique()}, reverse=True)
        month = st.selectbox("Month", months, key="ops_month")
        rep = ops.institutional_report(store, classes, people, month)
        if rep.empty:
            st.caption("No sessions in that month.")
        else:
            st.dataframe(rep, hide_index=True, width="stretch")
            st.download_button("⬇️ Download report (CSV)",
                               rep.to_csv(index=False).encode(),
                               f"institutional-report-{month}.csv",
                               "text/csv", width="stretch")


def page_timetables():
    section("🗓️", "Timetables & rooms")
    if not classes:
        empty_data_hero("No classes yet", ["Create classes first (🏫 Classes).",
                                           "Then set each class's room and "
                                           "weekly schedule here."])
        return
    cid = st.selectbox("Class", list(classes),
                       format_func=lambda c: f"{classes[c]['name']} · {c}",
                       key="tt_class")
    room = st.text_input("Room / location", value=ops.room_of(cid),
                         placeholder="e.g. Room 204, Block B")
    if st.button("💾 Save room"):
        ops.set_room(cid, room)
        store._audit(ACTOR, "set_room", f"{cid}={room}")
        st.rerun()

    st.markdown("**Weekly schedule**")
    slots = ops.slots_of(cid)
    with st.form("slot", border=False):
        c1, c2, c3 = st.columns(3)
        day = c1.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
                                   "Sun"])
        s1, s2 = c2.columns(2)
        start = s1.time_input("Start", value=_dt.time(9, 0), key="slot_s")
        end = s2.time_input("End", value=_dt.time(10, 30), key="slot_e")
        if st.form_submit_button("➕ Add slot") and end > start:
            slots = slots + [{"day": day,
                              "start": start.strftime("%H:%M"),
                              "end": end.strftime("%H:%M")}]
            ops.set_slots(ACTOR, cid, slots)
            st.rerun()
    if not slots:
        st.caption("No slots yet — add the weekly meeting times above. "
                   "The dashboard shows 'Now / Next' from these.")
    else:
        st.dataframe(pd.DataFrame(slots), hide_index=True, width="stretch")
        del_i = st.selectbox("Remove slot #", [""] +
                             [str(i + 1) for i in range(len(slots))],
                             key="slot_del")
        if del_i and st.button("🗑️ Remove slot"):
            slots.pop(int(del_i) - 1)
            ops.set_slots(ACTOR, cid, slots)
            st.rerun()


# ================= CLASSES / TEACHERS / RECORDS / AUDIT =================
@st.dialog("Delete class?")
def confirm_delete_class(cid: str):
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Delete", type="primary", width="stretch"):
        store.delete_class(cid)
        store._audit(ACTOR, "delete_class", cid)
        flash("success", f"Deleted class `{cid}`.")
        st.rerun()


def class_card(cid: str, cls: dict, show_teacher: bool):
    students = cls.get("students", [])
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown(f"**{cls['name']}** &nbsp;<span class='pill pill-muted'>{cid}</span>",
                    unsafe_allow_html=True)
        with h2.popover("⚙️"):
            pick = st.multiselect("Add students",
                                  [p for p in people if p not in students],
                                  format_func=lambda p:
                                  f"{people[p].get('name', p)} ({p})",
                                  key=f"add{cid}")
            if pick and st.button("Add", key=f"addb{cid}", type="primary",
                                  width="stretch"):
                for p in pick:
                    store.add_to_class(cid, p)
                st.rerun()
            new_late = st.text_input("Late after (HH:MM)",
                                     value=cls.get("late_after") or "",
                                     key=f"latep{cid}")
            if st.button("Save settings", key=f"save{cid}", width="stretch"):
                store.update_class(cid, {"late_after": new_late.strip()})
                st.rerun()
            if st.button("🗑️ Delete class", key=f"del{cid}", width="stretch"):
                confirm_delete_class(cid)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students", len(students))
        c2.metric("Ready", sum(1 for p in students if p in enc_all))
        c3.metric("Late after", cls.get("late_after") or "—")
        c4.metric("Room", ops.room_of(cid))
        if show_teacher:
            st.caption(f"Teacher: **{cls['teacher']}**")

        with st.expander("🔗 Invite link (students self-enroll)"):
            code = store.ensure_class_invite(cid)
            st.code(code or "—", language=None)
            base_url = _app_base_url()
            if base_url and code:
                st.code(f"{base_url}/?invite={code}", language=None)
            if st.button("♻️ Regenerate code", key=f"regen::{cid}"):
                store.regenerate_invite(cid)
                store._audit(ACTOR, "regen_invite", cid)
                st.rerun()

        with st.expander("📈 Attendance rates"):
            r = rate_table(cid)
            if r.empty:
                st.caption("No sessions recorded yet.")
            else:
                r = r.dropna(subset=["Rate"])
                st.dataframe(r, hide_index=True, width="stretch",
                             column_config={"Rate": st.column_config.ProgressColumn(
                                 "Attendance", min_value=0, max_value=100,
                                 format="%.0f%%")})

        with st.expander("🗓️ Monthly heatmap"):
            months = sorted({d[:7] for d in store.records_df(cid).Date.unique()},
                            reverse=True)
            if not months:
                st.caption("No sessions recorded yet — the heatmap appears "
                           "after your first session.")
            else:
                _render_heatmap(cid, st.selectbox("Month", months,
                                                  key=f"hm::{cid}"))


def page_classes():
    section("🏫", "My classes" if user["role"] == "teacher" else "Classes")
    with st.form("newclass", border=False):
        c1, c2, c3 = st.columns([2, 1.4, 1])
        name = c1.text_input("Class name", placeholder="e.g. CS Year 2 — Section A")
        owner = (c2.selectbox("Teacher", sorted(auth.list_users()))
                 if user["role"] == "admin" else c2.empty())
        late = c3.text_input("Late after (HH:MM)")
        if st.form_submit_button("➕ Create class", type="primary") and name.strip():
            cid = ("".join(ch for ch in name.strip().lower()
                           if ch.isalnum() or ch in "-_")[:24]
                   or f"class-{len(classes)+1}")
            if store.create_class(cid, name.strip(),
                                  owner if user["role"] == "admin"
                                  else user["username"],
                                  late.strip() or None):
                store._audit(ACTOR, "create_class", f"{cid} ({name.strip()})")
                flash("success", f"Created **{name.strip()}**.")
                st.rerun()
            st.error(f"`{cid}` already exists.")
    if not classes:
        empty_data_hero(
            "Create your first class",
            ["A class groups students and holds its own late rule and "
             "invite link.",
             "After creating it, share the invite link — students enroll "
             "themselves."])
        return
    for cid, cls in classes.items():
        class_card(cid, cls, show_teacher=(user["role"] == "admin"))


def page_teachers():
    section("👩‍🏫", "Teacher accounts")
    users = auth.list_users()
    c1, c2 = st.columns([2, 1])
    with c1:
        with st.form("newteacher", border=False):
            a1, a2 = st.columns(2)
            nu = a1.text_input("Username")
            npw = a2.text_input("Password (min 6 chars)", type="password")
            if st.form_submit_button("➕ Create account", type="primary"):
                ok = auth.add_user(nu, npw, role="teacher")
                store._audit(ACTOR, "create_teacher",
                             nu.strip().lower() if ok else "failed")
                flash("success" if ok else "error",
                      f"Created **{nu.strip().lower()}**." if ok else
                      "Invalid: empty/duplicate username, or password < 6 chars.")
                st.rerun()
    with c2:
        with st.container(border=True):
            target = st.selectbox("Account", [""] + sorted(users))
            if target:
                newpw = st.text_input("New password", type="password", key="npw")
                if st.button("Reset password", width="stretch") and newpw:
                    ok = auth.set_password(target, newpw)
                    store._audit(ACTOR, "reset_password",
                                 target if ok else "failed")
                    flash("success" if ok else "error",
                          "Password updated." if ok else "Too short (min 6).")
                    st.rerun()
                if st.button("🗑️ Remove account", width="stretch"):
                    ok = auth.remove_user(target)
                    store._audit(ACTOR, "remove_teacher", f"{target} ok={ok}")
                    flash("success" if ok else "error",
                          "Removed." if ok else "Cannot remove the last admin.")
                    st.rerun()
    st.dataframe(pd.DataFrame([{"Username": k,
                                "Role": "🛡️ " + v if v == "admin" else "👩‍🏫 " + v}
                               for k, v in sorted(users.items())]),
                 hide_index=True, width="stretch")


@st.dialog("Delete records?")
def confirm_delete_records(n: int, rows_idx):
    st.write(f"Permanently delete **{n}** record(s)? This can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Delete", type="primary", width="stretch"):
        store.delete_records(rows_idx)
        store._audit(ACTOR, "delete_records", f"n={n}")
        flash("success", f"Deleted {n} record(s).")
        st.rerun()


def page_records():
    section("📋", "Records")
    df_all = scope_records()
    if df_all.empty:
        empty_data_hero(
            "No records yet",
            ["Create a class and add students (invite links make this fast).",
             "Run a live session in ✅ Take attendance.",
             "Records land here automatically — filter, export, or delete."])
        return
    f1, f2, f3 = st.columns(3)
    cid = f1.selectbox("Class", ["All"] + list(classes),
                       format_func=lambda c: "All classes" if c == "All"
                       else f"{classes[c]['name']}")
    date = f2.selectbox("Date", ["All"] + sorted(df_all.Date.unique(), reverse=True))
    statuses = f3.multiselect("Status", ["Present", "Late", "Excused", "Absent"],
                              default=["Present", "Late", "Excused", "Absent"])
    view = df_all[df_all.Class == cid] if cid != "All" else df_all
    if date != "All":
        view = view[view.Date == date]
    view = view[view.Status.isin(statuses)].copy().reset_index().rename(
        columns={"index": "_row"})
    k = st.columns(5)
    k[0].metric("Records", len(view))
    k[1].metric("Present", (view.Status == "Present").sum())
    k[2].metric("Late", (view.Status == "Late").sum())
    k[3].metric("Excused", (view.Status == "Excused").sum())
    k[4].metric("Absent", (view.Status == "Absent").sum())
    v = view.drop(columns=["_row"]).copy()
    v["Status"] = v.Status.map(lambda s: STATUS_EMOJI.get(s, s))
    v.insert(0, "🗑️", False)
    ed = st.data_editor(v, hide_index=True, width="stretch", key="rec_editor",
                        column_config={"🗑️": st.column_config.CheckboxColumn(
                            "Delete", help="Tick, then Delete below")})
    del_idx = view.loc[ed["🗑️"], "_row"].tolist() if "🗑️" in ed else []
    b1, b2, b3 = st.columns(3)
    if del_idx and b1.button(f"🗑️ Delete {len(del_idx)} record(s)",
                             type="primary"):
        confirm_delete_records(len(del_idx), del_idx)
    b2.download_button("⬇️ CSV",
                       view.drop(columns=["_row"]).to_csv(index=False).encode(),
                       "attendance.csv", "text/csv", width="stretch")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        view.drop(columns=["_row"]).to_excel(xw, index=False)
    b3.download_button("⬇️ Excel", buf.getvalue(), "attendance.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch")


def page_audit():
    section("🗒️", "Audit log")
    st.caption("Append-only trail of every mark, edit, deletion, enrollment "
               "and account change.")
    adf = store.audit_df()
    if adf.empty:
        empty_state("🗒️", "No audit entries yet",
                    "Actions are logged automatically as they happen.")
        return
    c1, c2, c3 = st.columns(3)
    actor = c1.selectbox("Actor", ["All"] + sorted(adf.Actor.unique()))
    action = c2.selectbox("Action", ["All"] + sorted(adf.Action.unique()))
    day = c3.selectbox("Day", ["All"] + sorted({str(t)[:10] for t in adf.Time},
                                               reverse=True))
    view = adf
    if actor != "All":
        view = view[view.Actor == actor]
    if action != "All":
        view = view[view.Action == action]
    if day != "All":
        view = view[view.Time.str.startswith(day)]
    st.dataframe(view.iloc[::-1], hide_index=True, width="stretch")
    st.download_button("⬇️ Download audit log", view.to_csv(index=False).encode(),
                       "audit.csv", "text/csv", width="stretch")


# ================= NAV =================
def build_nav(role: str):
    pages = [st.Page(page_dashboard, title="Dashboard", icon="📊", default=True)]
    if role in ("teacher", "guest"):
        pages += [st.Page(page_attendance, title="Take attendance", icon="✅"),
                  st.Page(page_enroll, title="Enroll student", icon="➕"),
                  st.Page(page_students, title="Students", icon="👥")]
    if role == "teacher":
        pages += [st.Page(page_classes, title="My classes", icon="🏫"),
                  st.Page(page_timetables, title="Timetables", icon="🗓️")]
    if role == "admin":
        pages += [st.Page(page_teachers, title="Teachers", icon="👩‍🏫"),
                  st.Page(page_classes, title="Classes", icon="🏫"),
                  st.Page(page_students, title="Students", icon="👥"),
                  st.Page(page_oversight, title="Oversight", icon="🏛️"),
                  st.Page(page_timetables, title="Timetables", icon="🗓️"),
                  st.Page(page_audit, title="Audit log", icon="🗒️")]
    pages += [st.Page(page_records, title="Records", icon="📋")]
    return pages


PAGES = {p.title: p for p in build_nav(user["role"])}
pg = st.navigation(list(PAGES.values()))

with st.sidebar:
    m1, m2 = st.columns(2)
    m1.metric("Enrolled", len(people))
    _t = store.today_df()
    m2.metric("Today", _t[_t.Status.isin(["Present", "Late"])]["ID"].nunique()
              if not _t.empty else 0)
    user_chip(user)
    with st.popover("⚙️ Settings"):
        from zoneinfo import available_timezones
        tz_list = sorted(t for t in available_timezones() if "/" in t) + ["UTC"]
        cur = cfg.timezone or "UTC"
        tz = st.selectbox("Time zone", tz_list,
                          index=tz_list.index(cur) if cur in tz_list
                          else len(tz_list) - 1)
        if tz != cur:
            store.set_setting("timezone", tz)
            cfg.timezone = tz
            st.toast(f"Time zone set to {tz}")
        st.divider()
        blob = store.backup_zip()
        st.download_button("⬇️ Download full backup (.zip)", blob,
                           f"attendance-backup-{store._today()}.zip",
                           "application/zip", width="stretch")
        up = st.file_uploader("⬆️ Restore from backup (admin overwrites all)",
                              type=["zip"])
        if up and user["role"] == "admin" and not guest and \
                st.button("⚠️ Restore now (overwrites everything)"):
            ok = store.restore_zip(up.getvalue(), actor=ACTOR)
            flash("success" if ok else "error",
                  "Backup restored." if ok else "Invalid backup file.")
            st.rerun()
    if user["role"] == "admin" and not guest and auth.default_admin:
        st.error("⚠️ You're using the default admin password — change it under "
                 "Teachers.")
    if st.button("Log out", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.caption("v2.13.8 · self-hosted · data stays local")
pg.run()
