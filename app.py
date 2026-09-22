import os
from io import BytesIO

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Cloud secrets → env (HF Spaces injects env vars directly)
try:
    for _k, _v in st.secrets.get("env", {}).items():
        os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass

from attendance.auth import Auth
from attendance.config import Config
from attendance.engine import annotate, detect_and_encode, largest_face, recognize
from attendance.store import Store
from attendance.ui import (CSS, STATUS_EMOJI, empty_state, face_thumb, flash, hero,
                           inject_css, render_flash, section, user_chip)

st.set_page_config(page_title="Visual Attendance", page_icon="🪪", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

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


# ================= LOGIN =================
if "user" not in st.session_state:
    _, _, auth0 = resources(MAIN_DB)
    c1, c2, c3 = st.columns([1, 2.2, 1])
    with c2:
        st.write("")
        hero()
        with st.container(border=True):
            st.markdown("**Sign in**")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Sign in", type="primary", use_container_width=True):
                usr = auth0.verify(u, p)
                if usr:
                    st.session_state.user = usr
                    st.rerun()
                st.error("Wrong username or password.")
            if os.environ.get("ATT_GUEST_MODE") == "1":
                if st.button("👀 Continue as guest (demo sandbox)", use_container_width=True):
                    st.session_state.user = {"username": "guest", "role": "guest"}
                    st.rerun()
        st.caption("Teachers get accounts from the admin. Students never sign in. "
                   "Self-hosted — biometric data stays on this machine.")
    st.stop()

# ================= SESSION =================
user = st.session_state.user
guest = user["role"] == "guest"
cfg, store, auth = resources(GUEST_DB if guest else MAIN_DB)
st.session_state.setdefault("samples", [])
st.session_state.setdefault("thumbs", [])
st.session_state.setdefault("shot", 0)
render_flash()

if guest and "DEMO" not in store.load_classes():
    store.create_class("DEMO", "Guest demo", "guest")

classes = (store.load_classes() if user["role"] in ("admin", "guest")
           else store.classes_of(user["username"]))
people = store.load_people()
enc_all = store.load_encodings()


def rate_table(cid: str) -> pd.DataFrame:
    r = store.attendance_rates(cid)
    if not r.empty:
        r["Rate"] = pd.to_numeric(r["Rate"].astype(str).str.rstrip("%"))
    return r


def scope_records() -> pd.DataFrame:
    df = store.records_df()
    return df[df.Class.isin(classes)] if (classes and not df.empty) else df


def status_map_for(cid: str) -> dict:
    t = store.today_df(cid)
    return {row.ID: (row.Status, row.Time) for row in t.itertuples()} if not t.empty else {}


# ================= PAGES =================
def page_dashboard():
    section("📊", "Dashboard")
    if guest:
        st.info("Demo sandbox — everything resets when the app sleeps.", icon="🧪")

    @st.fragment(run_every="60s")
    def live_panel():
        df = scope_records()
        t = df[df.Date == store._today()] if not df.empty else df
        present = t[t.Status.isin(["Present", "Late"])]["ID"].nunique() if not t.empty else 0
        late = t[t.Status == "Late"]["ID"].nunique() if not t.empty else 0
        m = st.columns(4)
        m[0].metric("Enrolled faces", len(people))
        m[1].metric("Classes", len(classes))
        m[2].metric("Present today", present)
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
                               format_func=lambda c: f"{classes[c]['name']}",
                               label_visibility="collapsed")
            smap = status_map_for(cid)
            roster = classes[cid].get("students", [])
            rows = [{"ID": p, "Name": people.get(p, {}).get("name", p),
                     "Status": STATUS_EMOJI.get(smap.get(p, ("—",))[0], "· Not scanned"),
                     "Time": smap.get(p, ("", ""))[1] or "—"} for p in roster]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
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
            chart = (alt.Chart(m).mark_bar(corner_radius=3)
                     .encode(x=alt.X("Date:O", axis=alt.Axis(labelAngle=-40, title=None,
                                                             labelColor="#64748B")),
                             y=alt.Y("Students:Q", axis=alt.Axis(title=None,
                                                                 labelColor="#64748B")),
                             color=alt.Color("Status:N", scale=alt.Scale(
                                 domain=domain, range=[colors[s] for s in domain])),
                             tooltip=["Date", "Status", "Students"])
                     .properties(height=280))
            st.altair_chart(chart, use_container_width=True)
        else:
            empty_state("📉", "No records yet", "Take attendance to populate the chart.")


def page_attendance():
    section("✅", "Take attendance")
    if not classes:
        empty_state("🏫", "No classes yet", "Create a class first, then scan its roster.")
        return
    cid = st.selectbox("Class", list(classes),
                       format_func=lambda c: f"{classes[c]['name']} · {c}")
    cls = classes[cid]
    left, right = st.columns([5, 4])
    with left:
        with st.popover("⚙️ Scan settings"):
            tolerance = st.slider("Match tolerance (lower = stricter)", 0.60, 1.40,
                                  cfg.tolerance, 0.01)
        enc = {p: enc_all[p] for p in cls.get("students", []) if p in enc_all}
        st.caption(f"Roster **{len(cls.get('students', []))}** · with face data **{len(enc)}**"
                   + (f" · late after **{cls['late_after']}**" if cls.get("late_after") else ""))
        snap = st.camera_input("Scan the room (phone rear camera works)")
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
                bgr = annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results)
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        if st.button("🚫 End class — mark remaining absent", use_container_width=True):
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
                    c1.image(face_thumb(rgb, box), use_container_width=True)
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
                    c2.markdown("<span class='pill pill-muted'>skipped</span>",
                                unsafe_allow_html=True)
        if fresh:
            st.toast(f"{fresh} new mark(s) in {cls['name']}", icon="🪪")
        st.markdown("**Today**")
        t = store.today_df(cid)
        if t.empty:
            st.caption("No records for this class yet.")
        else:
            t = t.copy()
            t["Status"] = t.Status.map(lambda s: STATUS_EMOJI.get(s, s))
            st.dataframe(t, hide_index=True, use_container_width=True)


@st.dialog("End this class?")
def confirm_end_class(cid: str):
    st.write(f"Everyone in **{classes[cid]['name']}** without a record will be marked "
             "**Absent**. This can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True):
        st.rerun()
    if c2.button("Mark absent", type="primary", use_container_width=True):
        n = store.mark_absent_all(cid)
        flash("success", f"{n} student(s) marked Absent in {classes[cid]['name']}.")
        st.rerun()


def page_enroll():
    section("➕", "Enroll student")
    flash_on_save = st.session_state.pop("flash", None)
    if flash_on_save:
        st.success(flash_on_save[1])
    c1, c2, c3 = st.columns(3)
    pid = c1.text_input("Student ID", value=f"STU-{len(people)+1:03d}", key="pid")
    name = c2.text_input("Full name", key="pname")
    targets = c3.multiselect("Add to class(es)", list(classes),
                             format_func=lambda c: f"{classes[c]['name']} ({c})")
    if pid in people:
        st.warning(f"`{pid}` exists — new captures will be **added** to their samples.")
    needed = cfg.n_samples
    n = len(st.session_state.samples)
    st.progress(min(n / needed, 1.0), text=f"{n}/{needed} samples captured")

    lc, rc = st.columns([3, 2])
    with lc:
        shot = st.camera_input("Capture a sample — vary angle & lighting between shots",
                               key=f"cam{st.session_state.shot}")
        if shot is not None:
            rgb = to_rgb(shot)
            locs, encs = detect_and_encode(rgb, scale=1.0, num_jitters=cfg.enroll_jitters)
            box = largest_face(locs)
            if box is None:
                st.session_state.flash = ("error", "No face detected — improve lighting, "
                                          "then retake.")
            else:
                st.session_state.samples.append(encs[locs.index(box)])
                st.session_state.thumbs.append(face_thumb(rgb, box, 72))
                st.session_state.shot += 1
            st.rerun()
    with rc:
        st.markdown("**Captured samples**")
        if st.session_state.thumbs:
            cols = st.columns(4)
            for i, th in enumerate(st.session_state.thumbs):
                cols[i % 4].image(th, use_container_width=True)
            if st.button("🗑️ Remove last"):
                for k in ("samples", "thumbs"):
                    st.session_state[k] = st.session_state[k][:-1]
                st.rerun()
        else:
            st.caption("Thumbnails appear here as you capture.")
        with st.container(border=True):
            st.markdown("**Tips for good samples** 📸")
            st.markdown("• Vary angle, distance, expression  \n"
                        "• Consistent, front lighting  \n"
                        "• One face per shot · no sunglasses")
    if st.session_state.samples and st.button("💾 Save student", type="primary",
                                              disabled=not (pid.strip() and name.strip())):
        store.enroll(pid.strip(), name.strip(), st.session_state.samples)
        for tgt in targets:
            store.add_to_class(tgt, pid.strip())
        msg = (f"Saved **{name.strip()}** with {len(st.session_state.samples)} samples"
               + (f" → {', '.join(targets)}" if targets else "") + ".")
        st.session_state.samples, st.session_state.thumbs = [], []
        st.session_state.shot += 1
        st.session_state.flash = ("success", msg)
        st.rerun()


def class_card(cid: str, cls: dict, show_teacher: bool):
    roster = cls.get("students", [])
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown(f"**{cls['name']}** &nbsp;<span class='pill pill-muted'>{cid}</span>",
                    unsafe_allow_html=True)
        if h2.popover("⚙️", use_container_width=True):
            pick = st.multiselect("Add students",
                                  [p for p in people if p not in roster],
                                  format_func=lambda p: f"{people[p].get('name', p)} ({p})",
                                  key=f"add{cid}")
            if pick and st.button("Add", key=f"addb{cid}", type="primary",
                                  use_container_width=True):
                for p in pick:
                    store.add_to_class(cid, p)
                st.rerun()
            new_late = st.text_input("Late after (HH:MM)", value=cls.get("late_after") or "",
                                     key=f"late{cid}")
            if st.button("Save settings", key=f"save{cid}", use_container_width=True):
                store.update_class(cid, {"late_after": new_late.strip()})
                st.rerun()
            if st.button("🗑️ Delete class", key=f"del{cid}", use_container_width=True):
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
                st.dataframe(r, hide_index=True, use_container_width=True,
                             column_config={"Rate": st.column_config.ProgressColumn(
                                 "Attendance", min_value=0, max_value=100, format="%.0f%%")})


@st.dialog("Delete class?")
def confirm_delete_class(cid: str):
    st.write(f"Delete **{classes[cid]['name']}**? Attendance history stays in records.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True):
        st.rerun()
    if c2.button("Delete", type="primary", use_container_width=True):
        store.delete_class(cid)
        flash("success", f"Deleted class `{cid}`.")
        st.rerun()


def page_classes():
    section("🏫", "My classes" if user["role"] == "teacher" else "Classes")
    with st.form("newclass", border=False):
        c1, c2, c3 = st.columns([2, 1.4, 1])
        name = c1.text_input("Class name", placeholder="e.g. CS Year 2 — Section A")
        owner = c2.selectbox("Teacher", sorted(auth.list_users())) \
            if user["role"] == "admin" else c2.empty()
        late = c3.text_input("Late after (HH:MM)")
        if st.form_submit_button("➕ Create class", type="primary") and name.strip():
            cid = ("".join(ch for ch in name.strip().lower() if ch.isalnum() or ch in "-_")[:24]
                   or f"class-{len(classes)+1}")
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
                if st.button("Reset password", use_container_width=True) and newpw:
                    flash("success" if auth.set_password(target, newpw) else "error",
                          "Password updated." if auth.set_password(target, newpw) or True
                          else "Too short.")
                    st.rerun()
                if st.button("🗑️ Remove account", use_container_width=True):
                    ok = auth.remove_user(target)
                    flash("success" if ok else "error",
                          "Removed." if ok else "Cannot remove the last admin.")
                    st.rerun()
    st.dataframe(pd.DataFrame([{"Username": k,
                                "Role": "🛡️ " + v if v == "admin" else "👩‍🏫 " + v}
                               for k, v in sorted(users.items())]),
                 hide_index=True, use_container_width=True)


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
    statuses = f3.multiselect("Status", ["Present", "Late", "Absent"],
                              default=["Present", "Late", "Absent"])
    view = df[df.Class == cid] if cid != "All" else df
    if date != "All":
        view = view[view.Date == date]
    view = view[view.Status.isin(statuses)]
    k = st.columns(4)
    k[0].metric("Records", len(view))
    k[1].metric("Present", (view.Status == "Present").sum())
    k[2].metric("Late", (view.Status == "Late").sum())
    k[3].metric("Absent", (view.Status == "Absent").sum())
    v = view.copy()
    v["Status"] = v.Status.map(lambda s: STATUS_EMOJI.get(s, s))
    st.dataframe(v, hide_index=True, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV", view.to_csv(index=False).encode(),
                       "attendance.csv", "text/csv", use_container_width=True)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        view.to_excel(xw, index=False)
    d2.download_button("⬇️ Download Excel", buf.getvalue(), "attendance.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    if cid != "All":
        st.markdown("**Attendance rate (all sessions in this class)**")
        r = rate_table(cid)
        if not r.empty:
            r2 = r.sort_values("Rate", ascending=False)
            chart = (alt.Chart(r2).mark_bar(corner_radius=4, color="#4F46E5")
                     .encode(x=alt.X("Rate:Q", scale=alt.Scale(domain=[0, 100]), title=None),
                             y=alt.Y("Name:N", sort=r2["Name"].tolist(), title=None),
                             tooltip=["Name", "Rate"])
                     .properties(height=min(34 * len(r2) + 40, 420)))
            st.altair_chart(chart, use_container_width=True)


# ================= NAV =================
def build_nav(role: str):
    pages = [st.Page(page_dashboard, title="Dashboard", icon="📊", default=True)]
    if role in ("teacher", "guest"):
        pages += [st.Page(page_attendance, title="Take attendance", icon="✅"),
                  st.Page(page_enroll, title="Enroll student", icon="➕")]
    if role == "teacher":
        pages += [st.Page(page_classes, title="My classes", icon="🏫")]
    if role == "admin":
        pages += [st.Page(page_teachers, title="Teachers", icon="👩‍🏫"),
                  st.Page(page_classes, title="Classes", icon="🏫")]
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
    if not guest and auth.default_admin:
        st.error("⚠️ Admin still uses the default password — change it under Teachers.")
    if st.button("Log out", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.caption("v2.2 · self-hosted · data stays local")
pg.run()
