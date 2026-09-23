<div align="center">

# 🪪 Visual Class Attendance

**Face-recognition attendance for classrooms — enroll once, scan the room, done.**

[![VISIT HERE](https://img.shields.io/badge/▶_VISIT-Streamlit_Cloud-FF4B4B?logo=streamlit)](https://visual-attendance.streamlit.app)
[![CI](https://github.com/ethioel/Visual-Class-Attendance/actions/workflows/ci.yml/badge.svg)](https://github.com/ethioel/Visual-Class-Attendance/actions)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-16A34A)
![ML](https://img.shields.io/badge/Backend-ArcFace_ONNX-8B5CF6)

<!-- 
![demo](docs/demo.gif)
-->

</div>

---

## Why

Paper roll-calls and manual registers cost 5–10 minutes per session and are
trivially spoofed. Commercial biometric systems are closed, priced per seat,
and send face data to third parties. This project is the open middle ground:

| | Manual roll call | Commercial systems | **Visual Class Attendance** |
|---|---|---|---|
| Time per session | 5–10 min | ~30 s | **~30 s, hands-free** |
| Spoofable by proxy | Trivially | No | **Hard** (top-k multi-sample matching) |
| Face data leaves the room | — | Usually yes | **Never** (in-process CPU inference) |
| Runs on | Paper | Vendor hardware | **Any laptop / phone browser** |
| Cost | Free | Per-seat licenses | **Free, MIT** |

## How it works

```text
┌────────┐   ┌─────────────┐   ┌──────────────────┐   ┌─────────────┐
│ Camera │ → │ YuNet detect │ → │ 5-pt alignment   │ → │ ArcFace     │
│ stream │   │ (ONNX)      │   │ warp → 112×112   │   │ 512-d embed │
└────────┘   └─────────────┘   └──────────────────┘   └──────┬──────┘
                                                             ↓
                              ┌──────────────────────────────────────┐
                              │ top-3 mean distance vs class gallery │
                              │ ≤ tolerance → Present / Late         │
                              └──────────────────────────────────────┘
```

```python
# Enrollment: 5 jittered samples per person (flip-TTA denoises the embedding)
locs, encodings = detect_and_encode(rgb, scale=1.0, num_jitters=10)
store.enroll("STU-001", "Abebe Kebede", encodings)

# Recognition: mean of the 3 closest samples — one bad photo can neither
# fake a match nor break one.
results = recognize(rgb, class_gallery, tolerance=1.0)

# Marking: duplicate-safe per person/class/day, honors the class late rule.
store.mark("STU-001", "Abebe Kebede", status="Present", class_id="cs2a")
```

**Design decisions worth stealing:**

- **Top-k mean matching**, not single-nearest — robust to one corrupted
  enrollment sample and one lucky impostor frame.
- **Class-scoped galleries** — recognition only considers the selected class;
  outsiders and other-class students fall through to *Unknown*.
- **CSV as source of truth**, Excel as a derived view — append + `fsync` is
  crash-safe by construction; atomic-replace JSON/pickle everywhere else.
- **Backend-stamped embeddings** — stored samples carry a `BACKEND_ID`; a
  model upgrade cleanly invalidates old vectors instead of silently
  cross-matching incompatible spaces.

## Key capabilities

| Area | What you get |
|---|---|
| **Live sessions** | ▶️ Start → camera opens in-browser → frames recognized every few seconds → auto-marked with toasts. ⏹️ End → results card + one-click bulk absent. |
| **Enrollment** | Hands-free auto-capture (quality-gated: single face, lighting, sharpness) or manual snapshots; 5 samples per person. |
| **Self-enrollment** | Per-class invite code + link (`?invite=CODE`). Students enroll without accounts; regenerate to revoke. |
| **Attendance states** | Present / Late (per-class threshold) / **Excused** (excluded from rate denominator, survives bulk-absent) / Absent. Editable in-app per day. |
| **Reporting** | Dashboard with live metrics + 14-day chart · per-student attendance rates · **monthly heatmap** (students × days) · CSV/Excel export. |
| **Operations** | Roles (admin creates teachers; optional guest sandbox), batch roster import from CSV/Excel, timezone setting, dark mode. |

## Quickstart

```bash
git clone https://github.com/ethioel/Visual-Class-Attendance
cd Visual-Class-Attendance
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Calibration

Pick the threshold from your own population instead of guessing:

```bash
python scripts/calibrate.py
```

```text
genuine  max: 0.87    ← same person, different samples
impostor min: 1.18    ← different people
Good tolerance sits between these two numbers.
```

ArcFace reference points: genuine ≈ 0.3–0.9, impostor ≈ 1.1–1.5. If the two
distributions overlap, enrollment quality is the problem — recapture with
varied angle, distance, and lighting.

## Deployment

**Streamlit Community Cloud**:


## Privacy & security

- Faces are stored as **512-d embeddings, never images**.
- All inference is **in-process on CPU** — no third-party API sees a face.
- Passwords: PBKDF2-HMAC-SHA256, 200k iterations, per-user salt, uniform
  verify delay (brute-force throttle + no user enumeration).
- Invite links are **bearer credentials** — share with the class only;
  ♻️ Regenerate invalidates leaks.
- `attendance_db/`, `demo_db/`, `models/` are gitignored; real deployments
  belong on self-hosted hardware with institutional consent.

## Testing

```bash
python -m pytest -q
```

Covers matching logic, store flows (duplicate guard, Excused semantics,
legacy migration), auth rules, and invite codes. The ML backend is lazily
imported — CI runs in seconds with no ONNX/torch.

## Contributing

Issues and PRs welcome. Run `python -m pytest -q` before submitting; keep the
ML backend lazily imported so the test suite stays dependency-light.

## License

[MIT](LICENSE)

## Acknowledgments

- [ArcFace w600k_r50](https://github.com/deepinsight/insightface) — recognition ONNX
- [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo) — face detection
- [Streamlit](https://streamlit.io) and the
  [streamlit-camera-input-live](https://github.com/whitphx/streamlit-camera-input-live) component

<div align="center">
<sub>Built for real classrooms. Faces leave the frame; only numbers stay :)</sub>
</div>
