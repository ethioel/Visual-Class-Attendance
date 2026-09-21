from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from attendance.config import Config
from attendance.engine import annotate, detect_and_encode, largest_face, recognize
from attendance.store import Store

st.set_page_config(page_title="Visual Class Attendance", page_icon="🪪", layout="wide")


@st.cache_resource
def get_store() -> Store:
    return Store(Config())


store, cfg = get_store(), get_store().cfg
st.session_state.setdefault("samples", [])
st.session_state.setdefault("shot", 0)


def to_rgb(upload) -> np.ndarray:
    arr = np.frombuffer(upload.getvalue(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode image — retake.")
        st.stop()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.title("🪪 Visual Attendance")
    mode = st.radio("Mode", ["✅ Take attendance", "➕ Enroll person", "📋 Records"])
    tolerance = st.slider("Match tolerance (lower = stricter)", 0.35, 0.70, cfg.tolerance, 0.01)
    late = st.text_input("Late after (HH:MM, blank = off)", value=cfg.late_after or "")
    cfg.late_after = late.strip() or None
    st.divider()
    people = store.load_people()
    st.metric("Enrolled", len(people))
    tdf = store.today_df()
    n_present = tdf[tdf.Status.isin(["Present", "Late"])]["ID"].nunique() if not tdf.empty else 0
    st.metric("Present today", n_present)
    st.caption("Records are CSV-crash-safe. On HF Spaces the disk resets on rebuild — "
               "download data from **Records** and/or set HF_TOKEN + HF_DATASET_REPO "
               "secrets to auto-backup every mark to a Dataset repo.")

# ---------------- ENROLL ----------------
if mode == "➕ Enroll person":
    st.header("Enroll a person")
    c1, c2 = st.columns(2)
    pid = c1.text_input("Person ID", value=f"STU-{len(store.load_people())+1:03d}", key="pid")
    name = c2.text_input("Full name", key="pname")
    if pid in store.load_people():
        st.warning(f"`{pid}` exists — new captures will be **added** to their samples.")

    needed = cfg.n_samples
    st.progress(min(len(st.session_state.samples) / needed, 1.0),
                text=f"{len(st.session_state.samples)}/{needed} samples captured")

    shot = st.camera_input("Capture a sample (vary angle/lighting between shots)",
                           key=f"cam{st.session_state.shot}")
    if shot is not None:
        rgb = to_rgb(shot)
        locs, encs = detect_and_encode(rgb, scale=1.0, num_jitters=cfg.enroll_jitters)     # enroll at full resolution
        box = largest_face(locs)
        if box is None:
            st.error("No face detected — move closer / improve lighting, then retake.")
        else:
            st.session_state.samples.append(encs[locs.index(box)])
            st.session_state.shot += 1                     # fresh camera widget
            st.rerun()

    if st.session_state.samples and st.button("💾 Save person", type="primary",
                                              disabled=not (pid.strip() and name.strip())):
        store.enroll(pid.strip(), name.strip(), st.session_state.samples)
        st.success(f"Saved **{name}** with {len(st.session_state.samples)} samples.")
        st.session_state.samples, st.session_state.shot = [], st.session_state.shot + 1
        st.rerun()

    with st.expander("🗑️ Remove a person"):
        ppl = store.load_people()
        target = st.selectbox("Person", [""] + [f"{k} — {v['name']}" for k, v in ppl.items()])
        if target and st.button("Delete permanently"):
            store.remove_person(target.split(" — ")[0])
            st.rerun()

# ---------------- ATTENDANCE ----------------
elif mode == "✅ Take attendance":
    st.header("Take attendance")
    encodings, people = store.load_encodings(), store.load_people()
    if not encodings:
        st.info("Nobody enrolled yet — switch to **➕ Enroll person** first.")
    else:
        snap = st.camera_input("Scan the room (works on phone rear camera too)")
        if snap is not None:
            rgb = to_rgb(snap)
            with st.spinner("Recognizing…"):
                results = recognize(rgb, encodings, tolerance=tolerance, scale=cfg.detect_scale)
            st.image(cv2.cvtColor(annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results),
                                  cv2.COLOR_BGR2RGB), use_container_width=True)
            rows, fresh = [], 0
            for box, label, known, dist in results:
                if known:
                    nm = people.get(label, {}).get("name", label)
                    if store.mark(label, nm):
                        fresh += 1
                    rows.append({"ID": label, "Name": nm, "Status": store.status_now(),
                                 "Distance": round(dist, 3),
                                 "New": "✅" if fresh and rows == [] or True else ""})
                else:
                    rows.append({"Name": label, "Status": "—", "Distance": round(dist, 3)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if fresh:
                st.toast(f"{fresh} new mark(s) recorded", icon="🪪")
            if any(not r[2] for r in results):
                st.warning("Unknown face(s) detected — enroll them to track automatically.")

        st.divider()
        if st.button("🚫 Mark everyone else Absent (end of class)"):
            st.success(f"Marked {store.mark_absent_all()} absent.")
        st.subheader("Today")
        st.dataframe(store.today_df(), use_container_width=True, hide_index=True)

# ---------------- RECORDS ----------------
else:
    st.header("Records")
    df = store.records_df()
    if df.empty:
        st.info("No records yet.")
    else:
        sel = st.selectbox("Date", ["All"] + sorted(df.Date.unique(), reverse=True))
        st.dataframe(df if sel == "All" else df[df.Date == sel],
                     use_container_width=True, hide_index=True)
        st.divider()
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(),
                           "attendance_log.csv", "text/csv")
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df.to_excel(xw, index=False)
        c2.download_button("⬇️ Download Excel", buf.getvalue(), "attendance.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")