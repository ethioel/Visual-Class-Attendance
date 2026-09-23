from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

Face = Tuple[int, int, int, int]                 # (top, right, bottom, left), source scale
Recognition = Tuple[Face, str, bool, float]      # box, label, is_known, distance

BACKEND_ID = "arcface-w600k-r50"   # stamps stored encodings (migration guard)

# Canonical ArcFace 5-point template on 112×112 (insightface standard).
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

_YUNET_URLS = ["https://github.com/opencv/opencv_zoo/raw/main/models/"
               "face_detection_yunet/face_detection_yunet_2023mar.onnx"]
_ARCFACE_URLS = [
    "https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx",
]

_BACKEND = None  # lazy singleton: {"cv2", "yunet", "sess", "in", "out"}


def _download(urls: List[str], dest: str, min_size: int) -> str:
    """First URL that yields a file of plausible size wins. Atomic replace."""
    if os.path.isfile(dest) and os.path.getsize(dest) >= min_size:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".", suffix=".dl")
    os.close(fd)
    last: Optional[Exception] = None
    for url in urls:
        if not url:
            continue
        try:
            with urllib.request.urlopen(url, timeout=90) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if os.path.getsize(tmp) >= min_size:
                os.replace(tmp, dest)
                return dest
        except Exception as e:                       # try next mirror
            last = e
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"Could not download model ({dest}): {last}")


def _get_backend():
    """YuNet detector (OpenCV) + ArcFace recognition ONNX (onnxruntime), CPU.
    Weights download once to ATT_MODELS_DIR (default ./models). Lazy import →
    tests/CI never touch onnxruntime."""
    global _BACKEND
    if _BACKEND is None:
        import cv2
        import onnxruntime as ort
        mdir = os.environ.get("ATT_MODELS_DIR", "models")
        yunet = _download(
            [os.environ.get("ATT_YUNET_URL"), *_YUNET_URLS],
            os.path.join(mdir, "face_detection_yunet_2023mar.onnx"), 100_000)
        arc = _download(
            [os.environ.get("ATT_ARCFACE_URL"), *_ARCFACE_URLS],
            os.path.join(mdir, "w600k_r50.onnx"), 100_000_000)
        sess = ort.InferenceSession(arc, providers=["CPUExecutionProvider"])
        _BACKEND = {"cv2": cv2, "yunet": yunet, "sess": sess,
                    "in": sess.get_inputs()[0].name,
                    "out": sess.get_outputs()[0].name}
    return _BACKEND


def _embed(be, rgb112: np.ndarray, num_jitters: int) -> np.ndarray:
    x = (rgb112.astype(np.float32) - 127.5) / 127.5
    x = x.transpose(2, 0, 1)[None]                   # NCHW
    emb = be["sess"].run([be["out"]], {be["in"]: x})[0][0].astype(np.float32)
    if num_jitters > 1:                              # flip-TTA
        xf = x[:, :, :, ::-1]
        emb = emb + be["sess"].run([be["out"]], {be["in"]: xf})[0][0]
    return emb / max(float(np.linalg.norm(emb)), 1e-6)


def detect_and_encode(rgb, scale: float = 1.0, model: str = "arcface",
                      num_jitters: int = 1, prob_threshold: float = 0.90):
    """Detect (YuNet) + align (5-point similarity warp to 112×112) + encode
    (ArcFace 512-d, L2-normalized). Returns boxes in ORIGINAL scale.
    num_jitters>1 adds horizontal-flip TTA."""
    be = _get_backend()
    cv2 = be["cv2"]
    work = rgb if scale >= 1.0 else cv2.resize(rgb, None, fx=scale, fy=scale)
    bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(be["yunet"], "", (w, h),
                                    score_threshold=float(prob_threshold))
    _, faces = det.detect(bgr)
    if faces is None:
        return [], []

    locs, aligned = [], []
    for f in faces:
        x, y, fw, fh = [float(v) for v in f[0:4]]
        if fw < 8 or fh < 8:
            continue
        pts = np.asarray(f[4:14], dtype=np.float32).reshape(5, 2)
        M, _ = cv2.estimateAffinePartial2D(pts, _ARCFACE_DST)
        if M is None:
            continue
        face112 = cv2.warpAffine(bgr, M, (112, 112))
        x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
        x2, y2 = min(w - 1, int(round(x + fw))), min(h - 1, int(round(y + fh)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        locs.append((y1, x2, y2, x1))
        aligned.append(face112)
    if not aligned:
        return [], []

    inv = 1.0 / scale if scale < 1.0 else 1.0
    embs = [_embed(be, cv2.cvtColor(f112, cv2.COLOR_BGR2RGB), num_jitters)
            for f112 in aligned]
    return locs, embs


def best_match(encoding, encodings_db: Dict[str, list],
               tolerance: float = 1.0, top_k: int = 3) -> Tuple[Optional[str], float]:
    """Mean of the k closest samples per person. Skips samples with a different
    dimension so mixed databases never crash."""
    target = np.asarray(encoding, dtype=np.float32)
    best_pid, best_score = None, float("inf")
    for pid, samples in encodings_db.items():
        mats = [np.asarray(s, dtype=np.float32) for s in samples
                if np.asarray(s).shape == target.shape]
        if not mats:
            continue
        mat = np.stack(mats)
        d = np.linalg.norm(mat - target, axis=1)
        score = float(np.sort(d)[: min(top_k, d.size)].mean())
        if score < best_score:
            best_pid, best_score = pid, score
    if best_pid is not None and best_score <= tolerance:
        return best_pid, best_score
    return None, best_score


def recognize(rgb, encodings_db, tolerance: float = 1.0, scale: float = 1.0,
              model: str = "arcface",
              prob_threshold: float = 0.90) -> List[Recognition]:
    locs, encs = detect_and_encode(rgb, scale=scale, model=model,
                                   prob_threshold=prob_threshold)
    out: List[Recognition] = []
    for box, enc in zip(locs, encs):
        pid, dist = best_match(enc, encodings_db, tolerance=tolerance)
        out.append((box, pid, True, dist) if pid
                   else (box, f"Unknown ({dist:.2f})", False, dist))
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
