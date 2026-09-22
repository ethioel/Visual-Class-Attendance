import logging
import os
import threading
import time
from io import BytesIO

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    for _k, _v in st.secrets.get("env", {}).items():
        os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass


class _IceRetryNoise(logging.Filter):
    """Silence aioice STUN retry tracebacks (dead transport after widget teardown)."""
    def filter(self, record):
        msg = record.getMessage()
        return "Transaction.__retry" not in msg and "sendto" not in msg


logging.getLogger("asyncio").addFilter(_IceRetryNoise())

from attendance.auth import Auth
from attendance.config import Config
from attendance.engine import (_get_backend, annotate, detect_and_encode,
                               largest_face, recognize)
from attendance.store import Store
from attendance.ui import (STATUS_EMOJI, empty_state, face_thumb, flash, hero,
                           inject_css, render_flash, section, suggestion_box,
                           user_chip)

st.set_page_config(page_title="Visual Attendance", page_icon="🪪", layout="wide",
                   initial_sidebar_state="collapsed")

MAIN_DB = os.environ.get("ATT_DB_DIR", "attendance_db")
GUEST_DB = "demo_db"
STATUS_OPTS = ["—", "Present", "Late", "Excused", "Absent"]


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


# ---------------- shared live-camera machinery ----------------
_CAM = {"lock": threading.Lock(), "frame": None, "m": {}, "i": 0}


def _on_frame(frame):
    """aiortc thread: keep latest frame + quality metrics (cheap, every 3rd frame).
    Boxes are stored in FULL-FRAME coordinates for direct preview drawing."""
    try:
        img = frame.to_ndarray(format="bgr24")
        with _CAM["lock"]:
            _CAM["frame"] = img
            _CAM["i"] += 1
            skip = _CAM["i"] % 3
        if skip:
            return frame
        small = cv2.resize(img, None, fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = {"brightness": float(gray.mean()),
             "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
             "boxes": [], "shape": img.shape}
        try:
            mtcnn, _, _ = _get_backend()
            boxes, probs = mtcnn.detect(rgb)
            m["boxes"] = [(np.asarray(b) * 2).tolist()
                          for b, p in zip(boxes or [], probs or [])
                          if b is not None and p is not None and p >= 0.90]
        except Exception:
            pass
        with _CAM["lock"]:
            _CAM["m"] = m
    except Exception:
        pass
    return frame


def live_suggestion(m: dict) -> str:
    if not m:
        return "📷 Starting camera…"
    boxes = m.get("boxes")
    if boxes is None:
        return "📷 Analyzing…"
    if len(boxes) == 0:
        return "🙂 No face — look into the camera"
    if len(boxes) > 1:
        return "👥 Multiple faces — one person per sample"
    x1, y1, x2, y2 = boxes[0]
    h, w = m["shape"][:2]
    bw, cx, cy = (x2 - x1) / w, (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
    msgs = []
    if bw < 0.16:
        msgs.append("↔️ Move closer")
    elif bw > 0.80:
        msgs.append("↔️ Move back a bit")
    if abs(cx - 0.5) > 0.18:
        msgs.append("⬅️ shift left" if cx > 0.5 else "➡️ shift right")
    if cy < 0.28:
        msgs.append("⬇️ move down")
    elif cy > 0.78:
        msgs.append("⬆️ move up")
    br = m.get("brightness", 128)
    if br < 55:
        msgs.append("☀️ Too dark — face a light")
    elif br > 205:
        msgs.append("🌤️ Too bright — avoid direct light")
    if m.get("sharpness", 100) < 35:
        msgs.append("📸 Hold still — blurry")
    return " · ".join(msgs) if msgs else "✅ Perfect — hold still"


_ICE = {"iceServers": [
    {"urls": "stun:stun.l.google.com:19302"},
    {"urls": "turn:openrelay.metered.ca:80",
     "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": "turn:openrelay.metered.ca:443?transport=tcp",
     "username": "openrelayproject", "credential": "openrelayproject"},
]}


def _webrtc():
    from streamlit_webrtc import webrtc_streamer
    webrtc_streamer(key="livecam", video_frame_callback=_on_frame,
                    rtc_configuration=_ICE,
                    media_stream_constraints={"video": {"width": {"ideal": 1280},
                                                        "height": {"ideal": 720},
                                                        "facingMode": "user"},
                                              "audio": False},
                    async_processing=True)


def capture_section(pid: str, name: str, store: Store, cfg: Config):
    """Manual capture widget (snapshots) for one student — used in Enroll."""
    cap = st.session_state.setdefault(f"cap::{pid}", {"samples": [], "thumbs": []})
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
        for i, t in enumerate(cap["thumbs"]):
            th[i % 5].image(t, width="stretch")
    if cap["samples"]:
        if st.button(f"💾 Save {len(cap['samples'])} sample(s) for {name or pid}",
                     type="primary", width="stretch"):
            store.enroll(pid, name or pid, cap["samples"])
            st.session_state.pop(f"cap::{pid}", None)
            flash("success", f"Saved **{name or pid}** with {len(cap['samples'])} samples.")
            st.rerun()


# ================= LOGIN =================
if "user" not in st.session_state:
    inject_css(hide_sidebar=True)                      # ← no sidebar on login
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
def page_attendance():
    section("✅", "Take attendance")
    if not classes:
        empty_state("🏫", "No classes yet", "Create a class first, then scan its roster.")
        return
    cid = st.selectbox("Class", list(classes),
                       format_func=lambda c: f"{classes[c]['name']} · {c}")
    cls = classes[cid]
    enc = {p: enc_all[p] for p in cls.get("students", []) if p in enc_all}
    tolerance = st.slider("Match tolerance (lower = stricter)", 0.60, 1.40,
                          cfg.tolerance, 0.01)
    mode = st.radio("Scan mode", ["🎥 Live auto-scan (hands-free)", "📷 Single photo"],
                    horizontal=True)

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


def _live_session(cid: str, cls: dict, enc: dict, tolerance: float):
    """Hands-free: webrtc runs continuously; a fragment recognizes the latest
    frame every ~2.5s and auto-marks known faces. A faster fragment shows a
    live preview + setup suggestions."""
    key = f"session::{cid}"
    sess = st.session_state.setdefault(key, {"log": [], "last_i": -1, "unknown": 0})
    left, right = st.columns([5, 4])

    with left:
        if not enc:
            st.warning("Nobody on this roster has face data yet — enroll first.")

        @st.fragment(run_every=1.0)
        def _preview():
            with _CAM["lock"]:
                m, frame = dict(_CAM["m"]), _CAM["frame"]
            suggestion_box(live_suggestion(m))
            if frame is None:
                empty_state("🎥", "Waiting for camera…",
                            "Allow camera access — recognition starts automatically.")
                return
            vis = frame.copy()
            for b in (m.get("boxes") or []):
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="stretch")

        _webrtc()
        status = store.status_now(cls.get("late_after"))
        st.caption(f"Auto-marking as **{status}** · {len(enc)} face(s) on roster · "
                   "scans ~every 2.5s — just walk through the frame")
        _preview()

    with right:
        st.markdown("**Session log**")

        @st.fragment(run_every=2.5)
        def _loop():
            with _CAM["lock"]:
                frame, i = _CAM["frame"], _CAM["i"]
            if frame is None or i == sess["last_i"] or not enc:
                return
            if i - sess["last_i"] < 5:          # ignore stale frames
                return
            sess["last_i"] = i
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = recognize(rgb, enc, tolerance=tolerance,
                                    scale=cfg.detect_scale)
            except Exception:
                return
            status = store.status_now(cls.get("late_after"))
            new_marks, any_unknown = [], False
            for box, label, known, dist in results:
                if known:
                    nm = people.get(label, {}).get("name", label)
                    if store.mark(label, nm, status, cid):
                        new_marks.append(nm)
                    if not any(r for r in sess["log"] if r["ID"] == label):
                        sess["log"].insert(0, {"Name": nm, "ID": label,
                                               "Status": STATUS_EMOJI.get(status, status),
                                               "Score": round(dist, 2)})
                else:
                    any_unknown = True
            sess["unknown"] += int(any_unknown)
            if new_marks:
                st.toast(f"{', '.join(new_marks)} — marked", icon="🪪")

        _loop()
        lg = sess["log"]
        c1, c2 = st.columns(2)
        c1.metric("Marked this session", len(lg))
        c2.metric("Unknown glimpses", sess["unknown"])
        if lg:
            st.dataframe(pd.DataFrame(lg), hide_index=True, width="stretch")
        else:
            st.caption("Recognized students will appear here automatically.")
        if sess["unknown"] >= 3:
            st.warning("Repeated unknown faces — enroll them or add them to this class "
                       "(➕ Enroll / 👥 Students).")
        if st.button("⏹️ End session", width="stretch"):
            confirm_end_class(cid, key)


@st.dialog("End this class?")
def confirm_end_class(cid: str, sess_key: str = None):
    st.write(f"Everyone in **{classes[cid]['name']}** without a record will be marked "
             "**Absent**. Students marked **Excused** keep their status.")
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
        st.caption(f"Roster **{len(cls.get('students', []))}** · face data **{len(enc)}**"
                   + (f" · late after **{cls['late_after']}**" if cls.get("late_after") else ""))
        snap = st.camera_input("Scan the room")
        results = []
        if snap is not None:
            if not enc:
                st.warning("Nobody on this roster has face data yet — enroll first.")
            else:
                rgb = to_rgb(snap)
                with st.spinner("Recognizing…"):
                    results = recognize(rgb, enc, tolerance=tolerance,
                                        scale=cfg.detect_scale)
                st.session_state["_last_rgb"] = rgb
                st.image(cv2.cvtColor(annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                               results), cv2.COLOR_BGR2RGB), width="stretch")
        if st.button("🚫 End class — mark remaining absent", width="stretch"):
            confirm_end_class(cid)
    with right:
        st.markdown("**Results**")
        if snap is not None and not results and enc:
            empty_state("😶", "No faces detected", "Move closer / improve lighting.")
        status = store.status_now(cls.get("late_after"))
        fresh = 0
        for box, label, known, dist in results:
            rgb = st.session_state.get("_last_rgb")
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                if rgb is not None:
                    c1.image(face_thumb(rgb, box), width="stretch")
                if known:
                    nm = people.get(label, {}).get("name", label)
                    is_new = store.mark(label, nm, status, cid)
                    fresh += is_new
                    c2.markdown(f"**{nm}**  \n`{label}` · score {dist:.2f}")
                    c2.markdown(f"<span class='pill pill-{status.lower()}'>{status}</span>"
                                + (" <span class='pill pill-muted'>NEW</span>" if is_new else ""),
                                unsafe_allow_html=True)
                else:
                    c2.markdown(f"**Unknown face**  \nscore {dist:.2f} — not on this roster")
        if fresh:
            st.toast(f"{fresh} new mark(s) in {cls['name']}", icon="🪪")


# ================= ENROLL (single, manual capture) =================
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
    targets = st.multiselect("Add to class(es)", list(classes),
                             format_func=lambda c: f"{classes[c]['name']} ({c})")
    if pid in people:
        st.warning(f"`{pid}` exists — new captures will be **added** to their samples.")
    if not pid.strip() or not name.strip():
        st.caption("Enter an ID and name to start capturing.")
        return
    capture_section(pid.strip(), name.strip(), store, cfg)


# ================= STUDENTS (batch + edit + pending photos) =================
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
        ["📥 Import from file", "📷 Pending photos", "✏️ Edit students"])

    with tab_import:
        f = st.file_uploader("Upload CSV or Excel with columns **Name** (and optional **ID**)",
                             type=["csv", "xlsx", "xls"])
        if f is not None:
            try:
                roster = parse_roster(pd.read_csv(f) if f.name.endswith("csv")
                                      else pd.read_excel(f))
            except Exception as e:
                st.error(f"Could not read file: {e}")
                return
            st.caption("Edit IDs/names below before importing (add/remove rows freely).")
            preview = st.data_editor(roster, num_rows="dynamic", hide_index=True,
                                     width="stretch", key="import_prev")
            targets = st.multiselect("Add imported students to class(es)", list(classes),
                                     format_func=lambda c: f"{classes[c]['name']} ({c})")
            if st.button(f"📥 Import {len(preview)} student(s)", type="primary"):
                new = skip = 0
                for _, r in preview.iterrows():
                    if store.ensure_person(str(r["ID"]).strip(), str(r["Name"]).strip()):
                        new += 1
                        for tgt in targets:
                            store.add_to_class(tgt, str(r["ID"]).strip())
                    else:
                        skip += 1
                flash("success", f"Imported {new} new student(s)"
                                 + (f", skipped {skip} existing." if skip else "."))
                st.rerun()

    with tab_photos:
        pending = store.pending_samples()
        if not pending:
            empty_state("✅", "All students have photo samples",
                        f"Everyone has at least {cfg.n_samples} samples.")
            return
        st.caption(f"{len(pending)} student(s) still need photo samples — "
                   "imported from file or saved without photos.")
        pick = st.selectbox("Student", list(pending),
                            format_func=lambda p: f"{pending[p].get('name', p)} · {p}")
        ups = st.file_uploader("Upload photo files (JPG/PNG — one face per photo)",
                               type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                               key=f"up::{pick}")
        if ups and st.button(f"➕ Add {len(ups)} uploaded photo(s) as samples"):
            added = 0
            for upf in ups:
                arr = np.frombuffer(upf.getvalue(), np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                locs, encs = detect_and_encode(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                               scale=1.0, num_jitters=3)
                if locs:
                    store.enroll(pick, pending[pick].get("name", pick), [encs[0]])
                    added += 1
            flash("success" if added else "error",
                  f"Added {added} sample(s) from photos." if added
                  else "No usable faces found in the uploaded photos.")
            st.rerun()
        st.markdown("**…or capture with the camera**")
        capture_section(pick, pending[pick].get("name", pick), store, cfg)

    with tab_edit:
        counts = store.sample_counts()
        base = pd.DataFrame([{"ID": p, "Name": info.get("name", p),
                              "Samples": counts.get(p, 0)} for p, info in people.items()])
        if base.empty:
            empty_state("👥", "No students yet", "Enroll or import students first.")
            return
        ed = st.data_editor(base, disabled=["ID", "Samples"], hide_index=True,
                            width="stretch", num_rows="fixed", key="edit_people")
        if st.button("💾 Save name changes"):
            n = sum(1 for (_, a), (_, b) in zip(base.iterrows(), ed.iterrows())
                    if a["Name"] != b["Name"])
            for (_, a), (_, b) in zip(base.iterrows(), ed.iterrows()):
                if a["Name"] != b["Name"]:
                    store.rename_person(a["ID"], b["Name"])
            flash("success" if n else "info",
                  f"Renamed {n} student(s)." if n else "No changes to apply.")
            st.rerun()


# ================= CLASSES / TEACHERS / RECORDS =================
@st.dialog("Delete class?")
def confirm_delete_class(cid: str):
    st.write(f"Delete **{classes[cid]['name']}**? Attendance history stays in records.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Delete", type="primary", width="stretch"):
        store.delete_class(cid)
        flash("success", f"Deleted class `{cid}`.")
        st.rerun()


def class_card(cid: str, cls: dict, show_teacher: bool):
    roster = cls.get("students", [])
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown(f"**{cls['name']}** &nbsp;<span class='pill pill-muted'>{cid}</span>",
                    unsafe_allow_html=True)
        with h2.popover("⚙️"):
            pick = st.multiselect("Add students",
                                  [p for p in people if p not in roster],
                                  format_func=lambda p: f"{people[p].get('name', p)} ({p})",
                                  key=f"add{cid}")
            if pick and st.button("Add", key=f"addb{cid}", type="primary", width="stretch"):
                for p in pick:
                    store.add_to_class(cid, p)
                st.rerun()
            new_late = st.text_input("Late after (HH:MM)",
                                     value=cls.get("late_after") or "", key=f"late{cid}")
            if st.button("Save settings", key=f"save{cid}", width="stretch"):
                store.update_class(cid, {"late_after": new_late.strip()})
                st.rerun()
            if st.button("🗑️ Delete class", key=f"del{cid}", width="stretch"):
                confirm_delete_class(cid)
        c1, c2, c3 = st.columns(3)
        c1.metric("Students", len(roster))
        c2.metric("With face data", sum(1 for p in roster if p in enc_all))
        c3.metric("Late after", cls.get("late_after") or "—")
        if show_teacher:
            st.caption(f"Teacher: **{cls['teacher']}**")
        with st.expander("📈 Attendance rates"):
            r = rate_table(cid)
            if r.empty:
                st.caption("No sessions recorded yet.")
            else:
                r = r.dropna(subset=["Rate"])
                st.dataframe(r, hide_index=True, width="stretch",
                             column_config={"Rate": st.column_config.ProgressColumn(
                                 "Attendance", min_value=0, max_value=100, format="%.0f%%")})


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
                           if ch.isalnum() or ch in "-_")[:24] or f"class-{len(classes)+1}")
            ok = store.create_class(cid, name.strip(),
                                    owner if user["role"] == "admin" else user["username"],
                                    late.strip() or None)
            flash("success" if ok else "error",
                  f"Created **{name.strip()}**." if ok else f"`{cid}` already exists.")
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
                          "Password updated." if ok else "Password too short (min 6).")
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
                     .encode(x=alt.X("Rate:Q", scale=alt.Scale(domain=[0, 100]), title=None),
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


pg = st.navigation(build_nav(user["role"]))

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
                          index=tz_list.index(cur) if cur in tz_list else len(tz_list) - 1)
        if tz != cur:
            store.set_setting("timezone", tz)
            cfg.timezone = tz
            st.toast(f"Time zone set to {tz}")
    if not guest and auth.default_admin:
        st.error("⚠️ Admin still uses the default password — change it under Teachers.")
    if st.button("Log out", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.caption("v2.4 · self-hosted · data stays local")
pg.run()
