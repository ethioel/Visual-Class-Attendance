from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

Face = Tuple[int, int, int, int]                 # (top, right, bottom, left), source scale
Recognition = Tuple[Face, str, bool, float]      # box, label, is_known, distance


def detect_and_encode(rgb, scale: float = 1.0, model: str = "hog", num_jitters: int = 1):
    """Detect + encode. num_jitters>1 averages perturbed crops (slower, more stable).
    Coordinates are returned in the ORIGINAL image scale."""
    import cv2
    import face_recognition  # lazy: keeps tests and CI dlib-free

    work = rgb if scale >= 1.0 else cv2.resize(rgb, None, fx=scale, fy=scale)
    locs = face_recognition.face_locations(work, model=model)
    encs = face_recognition.face_encodings(work, locs, num_jitters=num_jitters)
    if scale != 1.0:
        inv = 1.0 / scale
        locs = [(int(t * inv), int(r * inv), int(b * inv), int(l * inv)) for t, r, b, l in locs]
    return locs, encs


def best_match(encoding: np.ndarray, encodings_db: Dict[str, List[np.ndarray]],
               tolerance: float = 0.55, top_k: int = 3) -> Tuple[Optional[str], float]:
    """Mean of the k closest samples per person — one lucky sample can't fake a
    match, one bad sample can't break one. Pure NumPy → testable without dlib."""
    best_pid, best_score = None, float("inf")
    target = np.asarray(encoding, dtype=np.float32)
    for pid, samples in encodings_db.items():
        mat = np.asarray(samples, dtype=np.float32)
        d = np.linalg.norm(mat - target, axis=1)
        score = float(np.sort(d)[: min(top_k, d.size)].mean())
        if score < best_score:
            best_pid, best_score = pid, score
    if best_pid is not None and best_score <= tolerance:
        return best_pid, best_score
    return None, best_score


def recognize(rgb, encodings_db, tolerance: float = 0.55, scale: float = 0.5,
              model: str = "hog") -> List[Recognition]:
    locs, encs = detect_and_encode(rgb, scale=scale, model=model)
    out: List[Recognition] = []
    for box, enc in zip(locs, encs):
        pid, dist = best_match(enc, encodings_db, tolerance=tolerance)
        out.append((box, pid, True, dist) if pid else (box, f"Unknown ({dist:.2f})", False, dist))
    return out


def largest_face(locs: List[Face]) -> Optional[Face]:
    return max(locs, key=lambda b: (b[1] - b[3]) * (b[2] - b[0])) if locs else None


def annotate(frame_bgr, results: List[Recognition]):
    import cv2
    for (t, r, b, l), label, known, _ in results:
        color = (0, 200, 0) if known else (0, 0, 230)
        cv2.rectangle(frame_bgr, (l, t), (r, b), color, 2)
        cv2.putText(frame_bgr, label, (l, max(22, t - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame_bgr