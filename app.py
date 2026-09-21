import os
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Community Cloud secrets → env (HF Spaces injects secrets as env vars already)
try:
    for _k, _v in st.secrets.get("env", {}).items():
        os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass

from attendance.auth import Auth
from attendance.config import Config
from attendance.engine import annotate, detect_and_encode, largest_face, recognize
from attendance.store import Store

st.set_page_config(page_title="Visual Attendance", page_icon="🪪", layout="wide")

MAIN_DB = os.environ.get("ATT_DB_DIR", "attendance_db")
GUEST_DB = "demo_db"


@st.cache_resource
def resources(db_dir: str):
    cfg = Config(db_dir=db_dir)
    return cfg, Store(cfg), Auth(cfg)


def to_rgb(upload) -> np.ndarray:
    arr = np.frombuffer(upload.getvalue(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode image — retake.")
        st.stop()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------- login gate ----------------
if "user" not in st.session_state:
    _, _, auth0 = resources(MAIN_DB)
    st.title("🪪 Visual Attendance")
    st.subheader("Sign in")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    b1, b2 = st.columns(2)
    if b1.button("Sign in", type="primary"):
        usr = auth0.verify(u, p)
        if usr:
            st.session_state.user = usr
            st.rerun()
        st.error("Wrong username or password.")
    if os.environ.get("ATT_GUEST_MODE") == "1" and b2.button("👀 Try guest demo"):
        st.session_state.user = {"username": "guest", "role": "guest"}
        st.rerun()
    st.caption("Teacher accounts are created by the admin. Students never sign in.")
    st.stop()

user = st.session_state.user
guest = user["role"] == "guest"
cfg, store, auth = resources(GUEST_DB if guest else MAIN_DB)
st.session_state.setdefault("samples", [])
st.session_state.setdefault("shot", 0)

if guest and "DEMO" not in store.load_classes():
    store.create_class("DEMO", "Guest demo", "guest")

classes = (store.load_classes() if user["role"] in ("admin", "guest")
           else store.classes_of(user["username"]))
people = store.load_people()

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🪪 Visual Attendance")
    st.caption(f"**{user['username']}** · {user['role']}")
    if user["role"] == "admin":
        page = st.radio("Menu", ["👩‍🏫 Teachers", "🏫 Classes", "📋 Records"])
    elif user["role"] == "teacher":
        page = st.radio("Menu", ["✅ Take attendance", "➕ Enroll student",
                                 "🏫 My classes", "📋 Records"])
    else:
        page = st.radio("Menu", ["✅ Take attendance", "➕ Enroll student", "📋 Records"])
    st.divider()
    st.metric("Enrolled faces", len(people))
    tdf = store.today_df()
    n_marked = tdf[tdf.Status.isin(["Present", "Late"])]["ID"].nunique() if not tdf.empty else 0
    st.metric("Marked today", n_marked)
    if user["role"] == "admin" and not guest and auth.default_admin:
        st.error("Admin still uses the default password — change it on 👩‍🏫 Teachers!")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()
    st.caption("Space disk is ephemeral — export records regularly, or set "
               "HF_TOKEN + HF_DATASET_REPO secrets for auto-backup.")


# ---------------- pages ----------------
def page_attendance():
    st.header("✅ Take attendance")
    if not classes:
        st.info("No classes yet — create one under 🏫 first.")
        return
    cid = st.selectbox("Class", list(classes),
                       format_func=lambda c: f"{classes[c]['name']} · {c}")
    cls = classes[cid]
    if cls.get("late_after"):
        st.caption(f"Late after **{cls['late_after']}**")
    tolerance = st.slider("Match tolerance (lower = stricter)", 0.35, 0.70,
                          cfg.tolerance, 0.01)
    enc_all = store.load_encodings()
    roster = cls.get("students", [])
    enc = {p: enc_all[p] for p in roster if p in enc_all}   # roster-filtered gallery
    st.caption(f"Roster {len(roster)} · with face data {len(enc)} — "
               "anyone not on this roster shows as Unknown.")
    snap = st.camera_input("Scan the room (phone rear camera works)")
    if snap is not None:
        if not enc:
            st.warning("Nobody on this roster has face data yet — enroll them first.")
        else:
            rgb = to_rgb(snap)
            with st.spinner("Recognizing…"):
                results = recognize(rgb, enc, tolerance=tolerance, scale=cfg.detect_scale)
            bgr = annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results)
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            rows, fresh = [], 0
            status = store.status_now(cls.get("late_after"))
            for box, label, known, dist in results:
                if known:
                    nm = people.get(label, {}).get("name", label)
                    is_new = store.mark(label, nm, status, cid)
                    fresh += is_new
                    rows.append({"Face": nm, "ID": label, "Status": status,
                                 "Distance": round(dist, 3), "New": "✅" if is_new else ""})
                else:
                    rows.append({"Face": f"Unknown ({dist:.2f})", "Status": "—",
                                 "Distance": round(dist, 3)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if fresh:
                st.toast(f"{fresh} new mark(s) in {cls['name']}", icon="🪪")
            if any(not r[2] for r in results):
                st.warning("Unknown face(s): not enrolled, or not added to this class.")
    st.divider()
    if st.button("🚫 Mark unmarked roster absent"):
        st.success(f"{store.mark_absent_all(cid)} student(s) marked Absent.")
    st.subheader(f"Today · {cls['name']}")
    st.dataframe(store.today_df(cid), hide_index=True, use_container_width=True)


def page_enroll():
    st.header("➕ Enroll student")
    c1, c2 = st.columns(2)
    pid = c1.text_input("Student ID", value=f"STU-{len(people)+1:03d}", key="pid")
    name = c2.text_input("Full name", key="pname")
    targets = st.multiselect("Add to class(es)", list(classes),
                             format_func=lambda c: f"{classes[c]['name']} ({c})")
    if pid in people:
        st.warning(f"`{pid}` exists — new captures will be **added** to their samples.")
    needed = cfg.n_samples
    st.progress(min(len(st.session_state.samples) / needed, 1.0),
                text=f"{len(st.session_state.samples)}/{needed} samples captured")
    shot = st.camera_input("Capture a sample — vary angle/lighting between shots",
                           key=f"cam{st.session_state.shot}")
    if shot is not None:
        rgb = to_rgb(shot)
        locs, encs = detect_and_encode(rgb, scale=1.0, num_jitters=cfg.enroll_jitters)
        box = largest_face(locs)
        if box is None:
            st.error("No face detected — improve lighting, then retake.")
        else:
            st.session_state.samples.append(encs[locs.index(box)])
            st.session_state.shot += 1
            st.rerun()
    if st.session_state.samples and st.button("💾 Save student", type="primary",
                                              disabled=not (pid.strip() and name.strip())):
        store.enroll(pid.strip(), name.strip(), st.session_state.samples)
        for t in targets:
            store.add_to_class(t, pid.strip())
        st.success(f"Saved **{name}** ({len(st.session_state.samples)} samples)"
                   + (f" → {', '.join(targets)}" if targets else ""))
        st.session_state.samples, st.session_state.shot = [], st.session_state.shot + 1
        st.rerun()


def page_my_classes():
    st.header("🏫 My classes")
    with st.form("newclass"):
        c1, c2 = st.columns([2, 1])
        name = c1.text_input("Class name")
        late = c2.text_input("Late after (HH:MM, blank = off)", value=cfg.late_after or "")
        if st.form_submit_button("Create class") and name.strip():
            cid = ("".join(ch for ch in name.strip().lower()
                           if ch.isalnum() or ch in "-_")[:24]
                   or f"class-{len(classes)+1}")
            if store.create_class(cid, name.strip(), user["username"], late.strip() or None):
                st.rerun()
            st.error(f"Class id `{cid}` already exists.")
    if not classes:
        st.info("No classes yet.")
        return
    everyone = {p: info.get("name", p) for p, info in people.items()}
    for cid, cls in classes.items():
        roster = cls.get("students", [])
        with st.expander(f"**{cls['name']}** · `{cid}` · {len(roster)} students"):
            if roster:
                st.dataframe(pd.DataFrame(
                    [{"ID": p, "Name": everyone.get(p, "?")} for p in roster]),
                    hide_index=True, use_container_width=True)
            else:
                st.caption("Empty roster — add students below or via ➕ Enroll.")
            pick = st.multiselect("Add existing students",
                                  [p for p in everyone if p not in roster],
                                  format_func=lambda p: f"{everyone[p]} ({p})",
                                  key=f"add{cid}")
            if pick and st.button("Add to roster", key=f"addb{cid}"):
                for p in pick:
                    store.add_to_class(cid, p)
                st.rerun()
            c1, c2 = st.columns(2)
            new_late = c1.text_input("Late after", value=cls.get("late_after") or "",
                                     key=f"late{cid}")
            if c1.button("Save settings", key=f"save{cid}"):
                store.update_class(cid, {"late_after": new_late.strip()})
                st.rerun()
            if (c2.checkbox("Confirm delete", key=f"delc{cid}")
                    and c2.button("🗑️ Delete class", key=f"delb{cid}")):
                store.delete_class(cid)
                st.rerun()


def page_teachers():
    st.header("👩‍🏫 Teacher accounts")
    with st.form("newteacher"):
        c1, c2 = st.columns(2)
        nu = c1.text_input("Username")
        npw = c2.text_input("Password (min 6 chars)", type="password")
        if st.form_submit_button("Create teacher account"):
            if auth.add_user(nu, npw, role="teacher"):
                st.success(f"Created **{nu.strip().lower()}**.")
            else:
                st.error("Invalid: empty/duplicate username, or password < 6 chars.")
    st.dataframe(pd.DataFrame([{"Username": k, "Role": v}
                               for k, v in sorted(auth.list_users().items())]),
                 hide_index=True, use_container_width=True)
    with st.expander("Manage an account"):
        target = st.selectbox("Account", [""] + sorted(auth.list_users()))
        if target:
            newpw = st.text_input("New password", type="password", key="npw")
            c1, c2 = st.columns(2)
            if c1.button("Reset password") and newpw:
                ok = auth.set_password(target, newpw)
                st.success("Password updated." if ok else "Password too short (min 6).")
            if c2.button("🗑️ Remove account"):
                st.success("Removed." if auth.remove_user(target)
                           else "Cannot remove the last admin.")


def page_classes():
    st.header("🏫 All classes")
    teachers = auth.list_users("teacher") or auth.list_users()
    with st.form("adminclass"):
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input("Class name")
        owner = c2.selectbox("Teacher", sorted(teachers))
        late = c3.text_input("Late after (HH:MM)")
        if st.form_submit_button("Create") and name.strip():
            cid = ("".join(ch for ch in name.strip().lower()
                           if ch.isalnum() or ch in "-_")[:24]
                   or f"class-{len(classes)+1}")
            if not store.create_class(cid, name.strip(), owner, late.strip() or None):
                st.error(f"`{cid}` already exists.")
    for cid, cls in classes.items():
        c1, c2 = st.columns([5, 1])
        c1.write(f"**{cls['name']}** · `{cid}` · teacher **{cls['teacher']}** · "
                 f"{len(cls['students'])} students")
        if c2.button("🗑️", key=f"del{cid}"):
            store.delete_class(cid)
            st.rerun()


def page_records():
    st.header("📋 Records")
    df_all = store.records_df()
    if df_all.empty:
        st.info("No records yet.")
        return
    c1, c2 = st.columns(2)
    cid = c1.selectbox("Class", ["All"] + list(classes),
                       format_func=lambda c: "All classes" if c == "All"
                       else f"{classes[c]['name']} ({c})")
    d = c2.selectbox("Date", ["All"] + sorted(df_all.Date.unique(), reverse=True))
    df = df_all[df_all.Class == cid] if cid != "All" else df_all
    if d != "All":
        df = df[df.Date == d]
    st.dataframe(df, hide_index=True, use_container_width=True)
    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇️ CSV", df.to_csv(index=False).encode(),
                        "attendance.csv", "text/csv")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False)
    dl2.download_button("⬇️ Excel", buf.getvalue(), "attendance.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if cid != "All":
        st.subheader("Attendance rate (all dates in this class)")
        st.dataframe(store.attendance_rates(cid), hide_index=True,
                     use_container_width=True)


if page == "✅ Take attendance":
    page_attendance()
elif page == "➕ Enroll student":
    page_enroll()
elif page == "🏫 My classes":
    page_my_classes()
elif page == "👩‍🏫 Teachers":
    page_teachers()
elif page == "🏫 Classes":
    page_classes()
elif page == "📋 Records":
    page_records()
