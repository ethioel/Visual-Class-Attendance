"""Local webcam attendance. Run: python cli.py   (enroll via `streamlit run app.py`)"""
import os
import time
from datetime import datetime

import cv2

from attendance.config import Config
from attendance.engine import annotate, recognize
from attendance.store import Store


def main() -> None:
    cfg = Config()
    store = Store(cfg)
    encodings, people = store.load_encodings(), store.load_people()
    if not encodings:
        raise SystemExit("Nobody enrolled — run `streamlit run app.py` → Enroll first.")

    cam = cv2.VideoCapture(cfg.camera_index)
    if not cam.isOpened():
        raise SystemExit("Could not open webcam.")

    print("SPACE=mark absent · S=snapshot · Q=quit")
    frame_i, last, t0, last_unk = 0, [], time.time(), 0.0
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % cfg.frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last = recognize(rgb, encodings, tolerance=cfg.tolerance, scale=cfg.detect_scale)
            for box, label, known, dist in last:
                if known:
                    store.mark(label, people.get(label, {}).get("name", label))
            if any(not r[2] for r in last) and time.time() - last_unk > 60:
                last_unk = time.time()
                cv2.imwrite(os.path.join(cfg.unknown_dir,
                            f"unknown_{datetime.now():%H%M%S}.jpg"), frame)
        out = annotate(frame.copy(), last)
        fps = 1.0 / max(1e-3, time.time() - t0); t0 = time.time()
        cv2.putText(out, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Attendance", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord(" "): print("absent marked:", store.mark_absent_all())
        if key == ord("s"):
            cv2.imwrite(os.path.join(cfg.snapshot_dir, f"snap_{datetime.now():%H%M%S}.jpg"), frame)
    store.mark_absent_all()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()