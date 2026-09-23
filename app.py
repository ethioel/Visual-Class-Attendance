import base64
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
    """Silence orphaned-fragment polling warnings (harmless stale-tab timers).
    Covers both message variants; attaches to every existing handler."""
    class _F(logging.Filter):
        def filter(self, record):
            m = record.getMessage()
            return ("does not exist anymore" not in m
                    and "Couldn't find fragment" not in m)

    seen = set()

    def attach(lg: logging.Logger) -> None:
        if id(lg) in seen:
            return
        seen.add(id(lg))
        for h in lg.handlers:
            h.addFilter(_F())

    names = ["", "streamlit"] + list(logging.root.manager.loggerDict)
    for name in names:
        try:
            attach(logging.getLogger(name))
        except Exception:
            pass


_install_fragment_noise_filter()

from camera_input_live import camera_input_live

from attendance.auth import Auth
from attendance.config import Config
from attendance.engine import annotate, detect_and_encode, largest_face, recognize
from attendance.store import Store
from attendance.ui import (STATUS_EMOJI, empty_state, face_thumb, flash, hero,
                           inject_css, render_flash, section, suggestion_box,
                           theme_is_dark, user_chip)

st.set_page_config(page_title="Visual Attendance", page_icon="🪪", layout="wide",
                   initial_sidebar_state="collapsed")

MAIN_DB = os.environ.get("ATT_DB_DIR", "attendance_db")
GUEST_DB = "demo_db"
STATUS_OPTS = ["—", "Present", "Late", "Excused", "Absent"]
WORK_WIDTH = 960          # recognition working width (px)
LIVE_PROB = 0.85          # detector score threshold for camera frames


@st.cache_resource
def resources(db_dir: str):
    cfg = Config(db_dir=db_dir)
    store, auth = Store(cfg), Auth(cfg)
    tz = store.get_settings().get("timezone")          # UI setting > env > system
    if tz:
        cfg.timezone = tz
    return cfg, store, auth


def to_rgb(upload) -> np.ndarray:
    arr = np.frombuffer(upload.getvalue(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode image — retake.")
        st.stop()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def camera_frame_bgr() -> np.ndarray | None:
    """Latest frame from the live camera element as BGR ndarray (None if not
    ready). Handles every return type the library has used across versions
    (BytesIO / data-URL string / PIL Image)."""
    raw = camera_input_live()
    if raw is None:
        return None
    if isinstance(raw, str):                       # data URL
        raw = BytesIO(base64.b64decode(raw.split(",", 1)[1]))
    elif isinstance(raw, (bytes, bytearray)):
        raw = BytesIO(raw)
    img = raw if isinstance(raw, Image.Image) else Image.open(raw)
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def to_working(rgb, max_w: int = WORK_WIDTH):
    """Resize to a fixed working width; return (image, inv) where
    original = image * inv."""
    h, w = rgb.shape[:2]
    if w > max_w:
        s = max_w / w
        return cv2.resize(rgb, (max_w, int(h * s))), 1.0 / s
    return rgb, 1.0


def live_suggestion(n_faces: int, brightness: float, sharpness: float) -> str:
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
    """Best-effort public URL of this app (for invite links)."""
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


def _render_heatmap(cid: str, month: str):
    """Students × days grid for one month, color-coded by status."""
    df = store.heatmap(cid, month)
    if df.empty:
        st.caption("No sessions recorded in this month.")
        return
    dates = sorted(df.Date.unique())
    day_of = {d: d[-2:] for d in dates}
    df["Day"] = df.Date.map(day_of)
    names = df.Name.drop_duplicates().tolist()
    no_rec = "#334155" if theme_is_dark() else "#E5E7EB"
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
                    class_targets: list | None = None,
                    on_saved=None):
    """Capture widget for one student. class_targets = classes added to on
    save; on_saved = optional callback(pid) fired once after a successful save."""
    class_targets = class_targets or []
    cap = st.session_state.setdefault(f"cap::{pid}", {"samples": [], "thumbs": []})
    run_key = f"auto::{pid}"
    done_key = f"autodone::{pid}"
    mode = st.radio("Capture mode",
                    ["🎥 Auto-capture (hands-free)", "📷 Manual snapshots"],
                    horizontal=True, key=f"cmode::{pid}")

    if mode.startswith("🎥"):
        # controls (outside fragment — full reruns are safe, fragment stays mounted)
        if st.session_state.get(run_key):
            if st.button("⏹️ Stop camera", width="stretch"):
                st.session_state[run_key] = False
                st.session_state[done_key] = False
                st.rerun()
        else:
            if st.button("▶️ Start auto-capture", type="primary", width="stretch"):
                st.session_state[run_key] = True
                st.session_state[done_key] = False
                st.rerun()

        # fragment ALWAYS mounted in this mode; body gates on flags → no orphan timers
        @st.fragment(run_every=3.0)
        def _auto():
            if st.session_state.get(done_key):
                suggestion_box("✅ All samples captured — press Save below.")
                return
            if not st.session_state.get(run_key):
                st.caption("📷 Camera off — press ▶️ Start auto-capture.")
                return
            try:
                frame = camera_frame_bgr()
            except Exception as e:
                st.error(f"Camera error: {str(e)[:160]}")
                return
            if frame is None:
                st.info("Starting camera… allow the permission pop-up (once).")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright, sharp = float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())
            try:
                locs, encs = detect_and_encode(rgb, scale=1.0,
                                               num_jitters=cfg.enroll_jitters)
            except Exception as e:
                st.error(f"Recognition error: {str(e)[:160]}")
                return
            n = len(cap["samples"])
            if n < cfg.n_samples:
                good = (len(locs) == 1 and 55 <= bright <= 210 and sharp >= 30)
                if good and time.time() - st.session_state.get(f"lc::{pid}", 0) > 2.5:
                    cap["samples"].append(encs[0])
                    cap["thumbs"].append(face_thumb(rgb, locs[0], 72))
                    st.session_state[f"lc::{pid}"] = time.time()
                    n += 1
                    st.toast(f"Sample {n}/{cfg.n_samples} captured", icon="📸")
                suggestion_box(live_suggestion(len(locs), bright, sharp))
            else:
                # done: stop camera inside the fragment, flip flags
                st.session_state[run_key] = False
                st.session_state[done_key] = True
                st.rerun()
            st.progress(min(n / cfg.n_samples, 1.0),
                        text=f"{n}/{cfg.n_samples} samples captured")
            th = st.columns(5)
            for i, tmb in enumerate(cap["thumbs"]):
                th[i % 5].image(tmb, width="stretch")

        _auto()
    else:
        left, right = st.columns([3, 2])
        with left:
            shot = st.camera_input("Capture a sample — vary angle & lighting",
                                   key=f"man::{pid}::{len(cap['samples'])}")
            if shot is not None:
                rgb = to_rgb(shot)
                locs, encs = detect_and_encode(rgb, scale=1.0,
                                               num_jitters=cfg.enroll_jitters)
                box = largest_face(locs)
                if box is None:
                    st.error("No face detected — improve lighting and retake.")
                else:
                    cap["samples"].append(encs[locs.index(box)])
                    cap["thumbs"].append(face_thumb(rgb, box, 72))
                    st.rerun()
        with right:
            th = st.columns(5)
            for i, tmb in enumerate(cap["thumbs"]):
                th[i % 5].image(tmb, width="stretch")

    if cap["samples"]:
        st.progress(min(len(cap["samples"]) / cfg.n_samples, 1.0),
                    text=f"{len(cap['samples'])}/{cfg.n_samples} samples captured")
        if st.button(f"💾 Save {len(cap['samples'])} sample(s) for {name or pid}",
                     type="primary", width="stretch"):
            store.enroll(pid, name or pid, cap["samples"])
            added = [t for t in class_targets if store.add_to_class(t, pid)]
            st.session_state.pop(f"cap::{pid}", None)
            st.session_state.pop(run_key, None)
            st.session_state.pop(done_key, None)
            msg = f"Saved **{name or pid}** with {len(cap['samples'])} samples."
            if added:
                msg += f" Added to: {', '.join(added)}."
            elif class_targets:
                msg += " (Class selection was cleared — re-select if needed.)"
            flash("success" if added else "warning", msg)
            if on_saved:
                on_saved(pid)
            st.rerun()


# ================= INVITE SELF-ENROLL (pre-login) =================
_invite = st.query_params.get("invite")
if _invite and "user" not in st.session_state:
    inject_css(hide_sidebar=True)
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
    inject_css(hide_sidebar=True)
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
        st.caption("Teachers get accounts from the admin. Students never sign in. "
                   "🌓 Dark mode: ⋮ menu → Settings → Theme.")
    st.stop()

# ================= SESSION =================
user = st.session_state.user
guest = user["role"] == "guest"
inject_css()
cfg, store, auth = resources(GUEST_DB if guest else MAIN_DB)
render_flash()

if guest and "DEMO" not in store.load_classes():
    store.create_class("DEMO", "Guest demo", "guest")

classes = (store.load_classes() if user["role"] in ("admin", "guest")
           else store.classes_of(user["username"]))
people = store.load_people()
enc_all = store.load_encodings()


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
            empty_state("📉", "No records yet", "Take attendance to populate the chart.")


# ================= ATTENDANCE =================
def _class_readiness_card(cid: str, cls: dict):
    """Plain-language readiness: who's in the class, who still needs face
    samples — with one-click fixes for both."""
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
                if pick and st.button("Add to class", key=f"quickaddb::{cid}",
                                      type="primary"):
                    for p in pick:
                        store.add_to_class(cid, p)
                    st.rerun()
            PAGES.get("Enroll student") and st.page_link(
                PAGES["Enroll student"],
                label="➕ …or enroll a new student (select this class)", icon="➕")
            PAGES.get("Classes") and st.page_link(
                PAGES["Classes" if user["role"] == "admin" else "My classes"],
                label="🔗 …or share the class invite link", icon="🔗")
        return
    missing = [p for p in students if p not in enc_all]
    if missing:
        names = [people[p].get("name", p) for p in missing]
        shown = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        with st.container(border=True):
            st.warning(f"**{len(missing)} of {len(students)}** students don't have "
                       f"face samples yet: {shown}")
            PAGES.get("Students") and st.page_link(
                PAGES["Students"], label="📷 Capture their face samples now",
                icon="📷")
    else:
        st.caption(f"👥 **{len(students)}** students ready for face recognition.")


def page_attendance():
    section("✅", "Take attendance")
    if not classes:
        empty_state("🏫", "No classes yet", "Create a class first, then scan its students.")
        return
    cid = st.selectbox("Class", list(classes),
                       format_func=lambda c: f"{classes[c]['name']} · {c}")
    cls = classes[cid]
    enc = {p: enc_all[p] for p in cls.get("students", []) if p in enc_all}
    _class_readiness_card(cid, cls)
    tolerance = st.slider("Match tolerance (lower = stricter)", 0.60, 1.40,
                          cfg.tolerance, 0.01)
    mode = st.radio("Scan mode", ["🎥 Live auto-scan (hands-free)", "📷 Single photo"],
                    horizontal=True, key="scan_mode")

    if mode.startswith("🎥"):
        _live_session(cid, cls, enc, tolerance)
    else:
        _single_photo(cid, cls, enc, tolerance)

    # ---- editable statuses (incl. Excused) ----
    st.divider()
    st.markdown("**✏️ Today's statuses — edit manually**")
    t = store.today_df(cid)
    smap = {r.ID: r.Status for r in t.itertuples()} if not t.empty else {}
    base = pd.DataFrame([{"ID": p, "Name": people.get(p, {}).get("name", p),
                          "Status": smap.get(p, "—")}
                         for p in cls.get("students", [])])
    if base.empty:
        st.caption("This class has no students yet.")
    else:
        ed = st.data_editor(base, disabled=["ID", "Name"], hide_index=True,
                            width="stretch", key=f"ed::{cid}",
                            column_config={"Status": st.column_config.SelectboxColumn(
                                "Status", options=STATUS_OPTS)})
        if st.button("💾 Apply status changes"):
            n = 0
            for (_, a), (_, b) in zip(base.iterrows(), ed.iterrows()):
                if a["Status"] != b["Status"]:
                    store.set_status(a["ID"], a["Name"],
                                     None if b["Status"] == "—" else b["Status"], cid)
                    n += 1
            flash("success" if n else "info",
                  f"Updated {n} record(s)." if n else "No changes to apply.")
            st.rerun()


def _run_recognition(frame_bgr, enc: dict, tolerance: float):
    """Shared recognition path. Returns (annotated_rgb, results, inv) —
    boxes are in WORK coords; multiply by inv for original-image coords."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    work, inv = to_working(rgb)
    results = recognize(work, enc, tolerance=tolerance, scale=1.0,
                        prob_threshold=LIVE_PROB)
    vis = cv2.cvtColor(annotate(cv2.cvtColor(work, cv2.COLOR_RGB2BGR), results),
                       cv2.COLOR_BGR2RGB)
    return vis, results, inv


def _apply_marks(results, cid: str, sess: dict):
    """Mark known faces, update the session log. Returns new names."""
    status = store.status_now(classes[cid].get("late_after"))
    new_marks, any_unknown = [], False
    for box, label, known, dist in results:
        if known:
            nm = people.get(label, {}).get("name", label)
            if store.mark(label, nm, status, cid):
                new_marks.append(nm)
            if not any(r["ID"] == label for r in sess["log"]):
                sess["log"].insert(0, {"Name": nm, "ID": label,
                                       "Status": STATUS_EMOJI.get(status, status),
                                       "Score": round(dist, 2)})
        else:
            any_unknown = True
    sess["unknown"] += int(any_unknown)
    return new_marks


def _live_session(cid: str, cls: dict, enc: dict, tolerance: float):
    """Start/Stop live session. The recognition fragment is ALWAYS mounted
    while this section renders; its body gates on sess['started'] — so timers
    never outlive the fragment (no orphan warnings)."""
    key = f"session::{cid}"
    sess = st.session_state.setdefault(key, {"started": False, "log": [],
                                             "captures": 0, "unknown": 0,
                                             "vis": None, "interval": 5})
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
                st.caption(f"Walk-through marking · {len(enc)} students ready · "
                           f"new scans mark as **{status}**.")
                sess["interval"] = st.select_slider(
                    "Capture interval", options=[3, 5, 8],
                    value=sess.get("interval", 5),
                    format_func=lambda v: f"every {v}s")
                if st.button("▶️ Start live session", type="primary",
                             width="stretch"):
                    sess.update(started=True, log=[], captures=0,
                                unknown=0, vis=None)
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
            try:
                vis, results, _ = _run_recognition(frame, enc, tolerance)
            except Exception as e:
                st.error(f"Recognition error: {str(e)[:160]}")
                return
            sess["vis"] = vis
            sess["captures"] += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            suggestion_box(live_suggestion(len(results), float(gray.mean()),
                                           float(cv2.Laplacian(gray, cv2.CV_64F).var())))
            st.image(vis, width="stretch")
            new_marks = _apply_marks(results, cid, sess)
            if new_marks:
                st.toast(f"{', '.join(new_marks)} — marked", icon="🪪")

        _loop()
        if sess["started"] and sess["vis"] is not None:
            st.image(sess["vis"], width="stretch")

    with right:
        st.markdown("**Session log**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Marked", len(sess["log"]))
        c2.metric("Unknown", sess["unknown"])
        c3.metric("Captures", sess["captures"])
        if sess["log"]:
            st.dataframe(pd.DataFrame(sess["log"]), hide_index=True, width="stretch")
        elif sess["started"]:
            st.caption("Recognized students appear here automatically.")
        if sess["unknown"] >= 3:
            st.warning("Repeated unknown faces — they may not be enrolled, or "
                       "not added to this class.")
        if sess["started"] and st.button("⏹️ End session", type="primary",
                                         width="stretch"):
            st.session_state[summary_key] = {
                "log": list(sess["log"]), "unknown": sess["unknown"],
                "captures": sess["captures"],
                "ended": store.cfg.now().strftime("%H:%M:%S")}
            sess.update(started=False, log=[], captures=0, unknown=0, vis=None)
            st.rerun()


def _render_session_summary(summary_key: str, cid: str):
    """Post-session results card."""
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
            st.success(f"{n} student(s) marked Absent. Excused students kept "
                       "their status.")
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
        if sess_key:
            st.session_state.pop(sess_key, None)
        flash("success", f"{n} student(s) marked Absent in {classes[cid]['name']}.")
        st.rerun()


def _single_photo(cid: str, cls: dict, enc: dict, tolerance: float):
    """One-shot flow — fallback, and the phone rear-camera mode."""
    left, right = st.columns([5, 4])
    with left:
        st.caption(f"👥 {len(cls.get('students', []))} students in this class · "
                   f"{len(enc)} ready for recognition"
                   + (f" · late after **{cls['late_after']}**"
                      if cls.get("late_after") else ""))
        snap = st.camera_input("Scan the room")
        results, rgb, inv = [], None, 1.0
        if snap is not None:
            if not enc:
                st.warning("No one in this class has face samples yet — see the "
                           "readiness card above.")
            else:
                rgb = to_rgb(snap)
                with st.spinner("Recognizing…"):
                    vis, results, inv = _run_recognition(rgb, enc, tolerance)
                st.image(vis, width="stretch")
        if st.button("🚫 End class — mark everyone else absent", width="stretch"):
            confirm_end_class(cid)
    with right:
        st.markdown("**Results**")
        if snap is not None and not results and rgb is not None:
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
                    c2.markdown(f"**{nm}**  \n`{label}` · score {dist:.2f}")
                    c2.markdown(f"<span class='pill pill-{status.lower()}'>{status}</span>"
                                + (" <span class='pill pill-muted'>NEW</span>" if is_new else ""),
                                unsafe_allow_html=True)
                else:
                    c2.markdown(f"**Unknown face**  \nscore {dist:.2f} — not on "
                                "this class list")
        if fresh:
            st.toast(f"{fresh} new mark(s) in {cls['name']}", icon="🪪")


# ================= ENROLL =================
def page_enroll():
    section("➕", "Enroll student")
    c1, c2, c3 = st.columns(3)
    id_mode = c1.radio("ID mode", ["🤖 Automatic", "✍️ Custom"], horizontal=True)
    nxt = f"STU-{len(people) + 1:03d}"
    while nxt in people:
        nxt = f"STU-{int(nxt.split('-')[1]) + 1:03d}"
    if id_mode.startswith("🤖"):
        pid = c2.text_input("Student ID (auto)", value=nxt, disabled=True)
    else:
        pid = c2.text_input("Student ID", key="pid")
    name = c3.text_input("Full name", key="pname")
    targets = st.multiselect(
        "Add to class(es) — students are only recognized in classes they belong to",
        list(classes), format_func=lambda c: f"{classes[c]['name']} ({c})",
        key="enroll_targets")
    if pid in people:
        st.warning(f"`{pid}` exists — new captures will be **added** to their "
                   "existing samples.")
    if not pid.strip() or not name.strip():
        st.caption("Enter an ID and name to start capturing.")
        return
    capture_section(pid.strip(), name.strip(), store, cfg, class_targets=targets)


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
        ["📥 Import from file", "📷 Missing face samples", "✏️ Edit students"])

    # NOTE: no early returns inside tabs — a `return` here would abort the
    # whole page and blank every tab defined after it (v2.11.0 bug).

    with tab_import:
        f = st.file_uploader("Upload CSV or Excel with columns **Name** "
                             "(and optional **ID**)",
                             type=["csv", "xlsx", "xls"])
        if f is None:
            st.caption("Tip: you can edit IDs/names in the preview before "
                       "importing.")
        else:
            try:
                roster = parse_roster(pd.read_csv(f) if f.name.endswith("csv")
                                      else pd.read_excel(f))
            except Exception as e:
                st.error(f"Could not read file: {e}")
                roster = None
            if roster is not None:
                st.caption("Edit IDs/names below before importing "
                           "(add/remove rows freely).")
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
            st.caption(f"**{len(pending)}** student(s) still need face samples — "
                       "imported from a file or saved without photos.")
            pick = st.selectbox(
                "Student", list(pending),
                format_func=lambda p: f"{pending[p].get('name', p)} · {p}")
            ms_targets = st.multiselect(
                "Add this student to class(es) on save", list(classes),
                format_func=lambda c: f"{classes[c]['name']} ({c})",
                key=f"mstargets::{pick}")
            ups = st.file_uploader(
                "Upload photo files (JPG/PNG — one face per photo)",
                type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                key=f"up::{pick}")
            if ups and st.button(f"➕ Add {len(ups)} uploaded photo(s) as samples"):
                added = 0
                for upf in ups:
                    arr = np.frombuffer(upf.getvalue(), np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    locs, encs = detect_and_encode(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                        scale=1.0, num_jitters=3)
                    if locs:
                        store.enroll(pick, pending[pick].get("name", pick),
                                     [encs[0]])
                        added += 1
                for _t in ms_targets:
                    store.add_to_class(_t, pick)
                flash("success" if added else "error",
                      f"Added {added} sample(s) from photos." if added
                      else "No usable faces found in the uploaded photos.")
                st.rerun()
            st.markdown("**…or capture with the camera**")
            capture_section(pick, pending[pick].get("name", pick), store, cfg,
                            class_targets=ms_targets)

    with tab_edit:
        counts = store.sample_counts()
        rows = []
        for p, info in people.items():
            n = counts.get(p, 0)
            if n >= cfg.n_samples:
                status = "✅ Ready"
            elif n > 0:
                status = "🟡 Partial — add more samples"
            elif info.get("backend"):
                status = "🔴 Needs recapture (model updated)"
            else:
                status = "⚪ No face samples"
            rows.append({"ID": p, "Name": info.get("name", p),
                         "Samples": n, "Status": status})
        base = pd.DataFrame(rows)
        if base.empty:
            empty_state("👥", "No students yet",
                        "Enroll, import, or share a class invite link.")
        else:
            st.caption(f"**{len(base)}** student(s) total · "
                       f"{(base.Samples >= cfg.n_samples).sum()} ready for "
                       "recognition")
            show = st.selectbox("Show",
                                ["All students", "Only those needing samples"],
                                key="edit_filter")
            view = (base if show == "All students"
                    else base[base.Samples < cfg.n_samples])
            ed = st.data_editor(view, disabled=["ID", "Samples", "Status"],
                                hide_index=True, width="stretch",
                                num_rows="fixed", key="edit_people")
            if st.button("💾 Save name changes"):
                n = 0
                for _, row in ed.iterrows():
                    orig = base.loc[base.ID == row["ID"], "Name"]
                    if not orig.empty and row["Name"] != orig.iloc[0]:
                        store.rename_person(row["ID"], row["Name"])
                        n += 1
                flash("success" if n else "info",
                      f"Renamed {n} student(s)." if n
                      else "No changes to apply.")
                st.rerun()


# ================= CLASSES / TEACHERS / RECORDS =================
@st.dialog("Delete class?")
def confirm_delete_class(cid: str):
    st.write(f"Delete **{classes[cid]['name']}**? Attendance history stays in "
             "records.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Delete", type="primary", width="stretch"):
        store.delete_class(cid)
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
                                     key=f"late{cid}")
            if st.button("Save settings", key=f"save{cid}", width="stretch"):
                store.update_class(cid, {"late_after": new_late.strip()})
                st.rerun()
            if st.button("🗑️ Delete class", key=f"del{cid}", width="stretch"):
                confirm_delete_class(cid)
        c1, c2, c3 = st.columns(3)
        c1.metric("Students", len(students))
        c2.metric("Ready for recognition", sum(1 for p in students if p in enc_all))
        c3.metric("Late after", cls.get("late_after") or "—")
        if show_teacher:
            st.caption(f"Teacher: **{cls['teacher']}**")

        with st.expander("🔗 Invite link (students self-enroll)"):
            code = store.ensure_class_invite(cid)
            st.code(code or "—", language=None)
            base = _app_base_url()
            if base and code:
                st.code(f"{base}/?invite={code}", language=None)
            st.caption("Students open the link (or add `?invite=CODE` to your "
                       "app URL) and enroll themselves into this class — no "
                       "account needed. Regenerate to invalidate old links.")
            if st.button("♻️ Regenerate code", key=f"regen::{cid}"):
                store.regenerate_invite(cid)
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
                st.caption("No sessions recorded yet.")
            else:
                month = st.selectbox("Month", months, key=f"hm::{cid}")
                _render_heatmap(cid, month)


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
            ok = store.create_class(cid, name.strip(),
                                    owner if user["role"] == "admin"
                                    else user["username"],
                                    late.strip() or None)
            flash("success" if ok else "error",
                  f"Created **{name.strip()}**." if ok
                  else f"`{cid}` already exists.")
            st.rerun()
    if not classes:
        empty_state("🏫", "No classes yet", "Create your first class above.")
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
                flash("success" if ok else "error",
                      f"Created **{nu.strip().lower()}**." if ok else
                      "Invalid: empty/duplicate username, or password < 6 chars.")
                st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown("**Manage an account**")
            target = st.selectbox("Account", [""] + sorted(users))
            if target:
                newpw = st.text_input("New password", type="password", key="npw")
                if st.button("Reset password", width="stretch") and newpw:
                    ok = auth.set_password(target, newpw)
                    flash("success" if ok else "error",
                          "Password updated." if ok
                          else "Password too short (min 6).")
                    st.rerun()
                if st.button("🗑️ Remove account", width="stretch"):
                    ok = auth.remove_user(target)
                    flash("success" if ok else "error",
                          "Removed." if ok else "Cannot remove the last admin.")
                    st.rerun()
    st.dataframe(pd.DataFrame([{"Username": k,
                                "Role": "🛡️ " + v if v == "admin" else "👩‍🏫 " + v}
                               for k, v in sorted(users.items())]),
                 hide_index=True, width="stretch")


def page_records():
    section("📋", "Records")
    df = scope_records()
    if df.empty:
        empty_state("📋", "No records yet", "Take attendance to create records.")
        return
    f1, f2, f3 = st.columns(3)
    cid = f1.selectbox("Class", ["All"] + list(classes),
                       format_func=lambda c: "All classes" if c == "All"
                       else f"{classes[c]['name']}")
    date = f2.selectbox("Date", ["All"] + sorted(df.Date.unique(), reverse=True))
    statuses = f3.multiselect("Status", ["Present", "Late", "Excused", "Absent"],
                              default=["Present", "Late", "Excused", "Absent"])
    view = df[df.Class == cid] if cid != "All" else df
    if date != "All":
        view = view[view.Date == date]
    view = view[view.Status.isin(statuses)]
    k = st.columns(5)
    k[0].metric("Records", len(view))
    k[1].metric("Present", (view.Status == "Present").sum())
    k[2].metric("Late", (view.Status == "Late").sum())
    k[3].metric("Excused", (view.Status == "Excused").sum())
    k[4].metric("Absent", (view.Status == "Absent").sum())
    v = view.copy()
    v["Status"] = v.Status.map(lambda s: STATUS_EMOJI.get(s, s))
    st.dataframe(v, hide_index=True, width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV", view.to_csv(index=False).encode(),
                       "attendance.csv", "text/csv", width="stretch")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        view.to_excel(xw, index=False)
    d2.download_button("⬇️ Download Excel", buf.getvalue(), "attendance.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch")
    if cid != "All":
        st.markdown("**Attendance rate (all sessions in this class)**")
        r = rate_table(cid).dropna(subset=["Rate"])
        if not r.empty:
            r2 = r.sort_values("Rate", ascending=False)
            chart = (alt.Chart(r2).mark_bar(cornerRadius=4, color="#4F46E5")
                     .encode(x=alt.X("Rate:Q", scale=alt.Scale(domain=[0, 100]),
                                     title=None),
                             y=alt.Y("Name:N", sort=r2["Name"].tolist(), title=None),
                             tooltip=["Name", "Rate"])
                     .properties(height=min(34 * len(r2) + 40, 420)))
            st.altair_chart(chart, width="stretch")


# ================= NAV =================
def build_nav(role: str):
    pages = [st.Page(page_dashboard, title="Dashboard", icon="📊", default=True)]
    if role in ("teacher", "guest"):
        pages += [st.Page(page_attendance, title="Take attendance", icon="✅"),
                  st.Page(page_enroll, title="Enroll student", icon="➕"),
                  st.Page(page_students, title="Students", icon="👥")]
    if role == "teacher":
        pages += [st.Page(page_classes, title="My classes", icon="🏫")]
    if role == "admin":
        pages += [st.Page(page_teachers, title="Teachers", icon="👩‍🏫"),
                  st.Page(page_classes, title="Classes", icon="🏫"),
                  st.Page(page_students, title="Students", icon="👥")]
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
    if user["role"] == "admin" and not guest and auth.default_admin:
        st.error("⚠️ You're using the default admin password — change it under "
                 "Teachers.")
    if st.button("Log out", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.caption("v2.11.1 · self-hosted · data stays local")
pg.run()
