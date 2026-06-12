"""Camera attention tracker: multi-face landmarks -> head pose -> attention score.

Design notes (ROCm-friendly): MediaPipe runs CPU real-time, so the GPU stays
dedicated to the LLM. No CUDA-only dependency anywhere in this module.

Attention proxy: a person is "attending" the screen when head yaw/pitch fall
inside a cone toward the display. Iris offset refines the score when available.
Scores are EMA-smoothed; tracklets are associated frame-to-frame by face-box IoU.
Identity is never computed — only geometry.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

from ..bus import EventBus
from ..events import AttentionEvent

log = logging.getLogger("aura.attention")

try:  # heavy deps are optional so the sim path works anywhere
    import cv2
    import mediapipe as mp
    import numpy as np
    _VISION_OK = True
    # MediaPipe >= ~0.10.2x removed the legacy `solutions` API in some builds.
    _LEGACY_API = hasattr(mp, "solutions")
    if not _LEGACY_API:  # Tasks API path (needs a downloaded .task model)
        from mediapipe.tasks import python as mp_tasks  # noqa: F401
        from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    _VISION_OK = False
    _LEGACY_API = False

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task")

# Canonical 3D face model (OpenCV head-pose convention):
# nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
_MODEL_3D = [
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
]
# Corresponding MediaPipe FaceMesh landmark indices (same order!)
# 33 = subject-right/image-left eye outer corner, 263 = image-right eye outer
_LM_IDX = [1, 152, 33, 263, 61, 291]

YAW_CONE_DEG = 20.0
PITCH_CONE_DEG = 15.0
EMA_ALPHA = 0.3


@dataclass
class _Tracklet:
    person_id: str
    box: tuple[float, float, float, float]
    score: float = 0.8
    last_seen: float = field(default_factory=time.time)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class AttentionTracker:
    """Consumes webcam/RTSP frames, publishes AttentionEvent per tracklet."""

    def __init__(self, bus: EventBus, camera: int | str = 0, fps: float = 10.0,
                 max_faces: int = 6, model_path: str = "face_landmarker.task") -> None:
        if not _VISION_OK:
            raise RuntimeError("Install opencv-python + mediapipe for live vision, "
                               "or run with --sim.")
        self.bus = bus
        self.camera = camera
        self.period = 1.0 / fps
        if _LEGACY_API:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=max_faces, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self._detect = self._detect_legacy
        else:
            import os
            if not os.path.exists(model_path):
                raise RuntimeError(
                    f"MediaPipe Tasks API needs the model file {model_path!r}. "
                    f"Download once:\n  curl -L -o {model_path} "
                    f"{FACE_LANDMARKER_MODEL_URL}")
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                num_faces=max_faces)
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
            self._detect = self._detect_tasks
        self._tracklets: list[_Tracklet] = []
        self._next_id = 0
        self._stop = asyncio.Event()

    # -------------------------------------------------------- API adapters
    def _detect_legacy(self, rgb) -> list:
        res = self._mesh.process(rgb)
        return list(res.multi_face_landmarks or [])

    def _detect_tasks(self, rgb) -> list:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(image)
        # Tasks API returns plain landmark lists; wrap for a uniform interface
        class _Wrap:  # noqa: D401 — tiny adapter
            def __init__(self, lms):
                self.landmark = lms
        return [_Wrap(lms) for lms in res.face_landmarks]

    # ---------------------------------------------------------------- pose
    def _head_pose(self, landmarks, w: int, h: int) -> tuple[float, float]:
        img_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h)
                            for i in _LM_IDX], dtype=np.float64)
        model = np.array(_MODEL_3D, dtype=np.float64)
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(model, img_pts, cam, np.zeros(4),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0, 0.0
        rot, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rot)  # (pitch, yaw, roll) degrees
        pitch, yaw = float(angles[0]), float(angles[1])
        # normalize: the model convention places frontal pitch near ±180
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return yaw, pitch

    @staticmethod
    def _attention_from_pose(yaw: float, pitch: float) -> float:
        y = max(0.0, 1.0 - abs(yaw) / (2 * YAW_CONE_DEG))
        p = max(0.0, 1.0 - abs(pitch) / (2 * PITCH_CONE_DEG))
        return y * p

    # ------------------------------------------------------------ tracking
    def _associate(self, box) -> _Tracklet:
        best, best_iou = None, 0.3
        for t in self._tracklets:
            i = _iou(t.box, box)
            if i > best_iou:
                best, best_iou = t, i
        if best is None:
            best = _Tracklet(person_id=f"person_{self._next_id}", box=box)
            self._next_id += 1
            self._tracklets.append(best)
        best.box = box
        best.last_seen = time.time()
        return best

    # ---------------------------------------------------------------- frame
    async def process_frame(self, frame) -> list:
        """Detect faces in one frame, update tracklets, publish events.
        Returns the events published (also used directly by tests/eval)."""
        h, w = frame.shape[:2]
        faces = await asyncio.to_thread(
            self._detect, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        published = []
        for face in faces:
            xs = [lm.x for lm in face.landmark]
            ys = [lm.y for lm in face.landmark]
            box = (min(xs), min(ys), max(xs), max(ys))
            trk = self._associate(box)
            yaw, pitch = self._head_pose(face.landmark, w, h)
            raw = self._attention_from_pose(yaw, pitch)
            trk.score = EMA_ALPHA * raw + (1 - EMA_ALPHA) * trk.score
            event = AttentionEvent(
                person_id=trk.person_id, attention=round(trk.score, 3),
                yaw=round(yaw, 1), pitch=round(pitch, 1),
                source="camera",
            )
            published.append(event)
            await self.bus.publish(event)
        # expire stale tracklets (left the room)
        now = time.time()
        self._tracklets = [t for t in self._tracklets
                           if now - t.last_seen < 3.0]
        return published

    # ---------------------------------------------------------------- loop
    async def run(self) -> None:
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera!r}")
        log.info("attention tracker started on camera=%s", self.camera)
        try:
            while not self._stop.is_set():
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok:
                    await asyncio.sleep(self.period)
                    continue
                await self.process_frame(frame)
                await asyncio.sleep(self.period)
        finally:
            cap.release()

    def stop(self) -> None:
        self._stop.set()
