"""
Face detectors
"""

import logging
from typing import Optional

import numpy as np

from .onnx_estimator import FaceBox
from .tracking import FaceTracker

log = logging.getLogger(__name__)


class UnifaceFaceBoxProvider:
    def __init__(self, detector=None, max_faces: int = 4,
                 det_width: int = 960, min_confidence: float = 0.5) -> None:
        if detector is None:
            from uniface.detection import RetinaFace
            detector = RetinaFace()
        self.det = detector
        self.max_faces = max_faces
        self.det_width = det_width
        self.min_confidence = min_confidence
        self._tracker = FaceTracker()

    def __call__(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[FaceBox]:
        import cv2
        h, w = frame_rgb.shape[:2]
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        scale = 1.0
        if self.det_width and w > self.det_width:
            scale = self.det_width / w
            bgr = cv2.resize(bgr, (self.det_width, int(round(h * scale))))
        inv = 1.0 / scale

        dets = []
        for f in self.det.detect(bgr):
            conf = float(getattr(f, "confidence", 1.0))
            if conf < self.min_confidence:
                continue
            b = np.asarray(f.bbox, dtype=float).ravel()
            x1 = max(0, int(b[0] * inv)); y1 = max(0, int(b[1] * inv))
            x2 = min(w, int(b[2] * inv)); y2 = min(h, int(b[3] * inv))
            if x2 <= x1 or y2 <= y1:
                continue

            origin = None
            lm = getattr(f, "landmarks", None)
            if lm is not None:
                lm = np.asarray(lm, dtype=float).reshape(-1, 2)
                if lm.shape[0] >= 2:
                    ex = (lm[0, 0] + lm[1, 0]) * 0.5 * inv
                    ey = (lm[0, 1] + lm[1, 1]) * 0.5 * inv
                    origin = (min(1.0, max(0.0, ex / w)), min(1.0, max(0.0, ey / h)))
            if origin is None:
                origin = ((x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h))
            
            dets.append((conf, FaceBox(x1, y1, x2, y2, origin)))

        dets.sort(key=lambda t: -t[0])
        boxes = [fb for _, fb in dets[:self.max_faces]]
        ids = self._tracker.update([((b.x1 + b.x2) / (2.0 * w), (b.y1 + b.y2) / (2.0 * h)) for b in boxes])

        for b, tid in zip(boxes, ids):
            b.track_id = tid
        
        return boxes
    