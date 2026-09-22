from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

Face = Tuple[int, int, int, int]                 # (top, right, bottom, left), source scale
Recognition = Tuple[Face, str, bool, float]      # box, label, is_known, distance

_BACKEND = None  # lazy singleton: (mtcnn, resnet, torch)


def _get_backend():
    """MTCNN detector + InceptionResnetV1 (VGGFace2, 512-d) on CPU.
    Weights (~100 MB) download on first use, cached per runtime.
    Lazy import → tests/CI never touch torch."""
    global _BACKEND
    if _BACKEND is None:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN
        mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device="cpu")
        resnet = InceptionResnetV1(pretrained="vggface2").eval()
        _BACKEND = (mtcnn, resnet, torch)
    return _BACKEND


def detect_and_encode(rgb, scale: float = 1.0, model: str = "facenet",
                      num_jitters: int = 1):
    """Detect + encode. Returns ((top,right,bottom,left) boxes in ORIGINAL scale,
    512-d L2-normalized embeddings). num_jitters>1 adds horizontal-flip TTA
    (embedding = mean of both views). `model` kept for API compatibility."""
    import cv2
    mtcnn, resnet, torch = _get_backend()

    work = rgb if scale >= 1.0 else cv2.resize(rgb, None, fx=scale, fy=scale)
    boxes, probs = mtcnn.detect(work)
    if boxes is None:
        return [], []

    ok = [i for i, (b, p) in enumerate(zip(boxes, probs))
          if b is not None and p is not None and p >= 0.90]
    if not ok:
        return [], []
    boxes = np.stack([boxes[i] for i in ok])

    with torch.no_grad():
        faces = mtcnn.extract(work, boxes)              # aligned (N,3,160,160)
        emb = resnet(faces)
        if num_jitters > 1:                             # cheap TTA
            emb = emb + resnet(torch.flip(faces, dims=(3,)))
        emb = torch.nn.functional.normalize(emb, dim=1)

    inv = 1.0 / scale if scale < 1.0 else 1.0
    locs = [(int(y1 * inv), int(x2 * inv), int(y2 * inv), int(x1 * inv))
            for (x1, y1, x2, y2) in boxes]
    return locs, list(emb.cpu().numpy())


def best_match(encoding, encodings_db: Dict[str, list],
               tolerance: float = 1.0, top_k: int = 3) -> Tuple[Optional[str], float]:
    """Mean of the k closest samples per person. Skips samples with a different
    dimension (e.g. legacy dlib 128-d vectors) so mixed databases never crash."""
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


def recognize(rgb, encodings_db, tolerance: float = 1.0, scale: float = 0.5,
              model: str = "facenet") -> List[Recognition]:
    locs, encs = detect_and_encode(rgb, scale=scale, model=model)
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
