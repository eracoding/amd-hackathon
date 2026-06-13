"""
Gaze modality for handling gaze data processing.
"""
import time
import math
from typing import Callable, Optional

from ...core.modality import Modality
from .estimators import GazeEstimator, GazeResult

TargetFn = Callable[[GazeResult], "tuple[str, float]"]

_TARGET_COLORS = {
    "own_screen": (0, 200, 0),
    "left":       (0, 165, 255),
    "right":      (0, 165, 255),
    "up_away":    (0, 0, 255),
    "elsewhere":  (0, 0, 255),
    "eyes_closed": (150, 150, 150)
}
_NEUTRAL = (255, 255, 255)

def _draw_gaze_overlay(cv2, frame, r: GazeResult, target, mirror: bool) -> None:
    h, w = frame.shape[:2]

    ox = (1.0 - r.origin[0]) if mirror else r.origin[0]
    ox_px, oy_px = int(ox * w), int(r.origin[1] * h)

    length = 0.28 * w
    x_sign = 1.0 if mirror else -1.0
    ex = int(ox_px + length * math.sin(math.radians(r.yaw)) * x_sign)
    ey = int(oy_px - length * math.sin(math.radians(r.pitch)))

    color = _TARGET_COLORS.get(target[0], _NEUTRAL) if target else _NEUTRAL

    cv2.drawMarker(frame, (w // 2, h // 2), (90, 90, 90), cv2.MARKER_CROSS, 16, 1)
    cv2.arrowedLine(frame, (ox_px, oy_px), (ex, ey), color, 3, tipLength=0.25)
    cv2.circle(frame, (ox_px, oy_px), 4, color, -1)

    if target is not None:
        who = f"p{r.track_id} " if r.track_id is not None else ""
        cv2.putText(frame, f"{who}{target[0]} ({target[1]:.2f})", (ox_px - 45, oy_px - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"yaw={r.yaw:+.0f} pitch={r.pitch:+.0f} blink={r.blink:.2f}", (10, 24 + 22 * r.face_index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)


class GazeModality(Modality):
    name = "gaze"

    def __init__(self, bus, source_id: str, estimator: GazeEstimator,
                 source=0, target_fps: float = 15.0,
                 show_preview: bool = False, mirror_preview: bool = True, 
                 target_fn: Optional[TargetFn] = None) -> None:
        super().__init__(bus, source_id)
        self.estimator = estimator
        self.source = source
        self.min_dt = 1.0 / target_fps if target_fps > 0 else 0.0
        self.show_preview = show_preview
        self.mirror_preview = mirror_preview
        self.target_fn = target_fn

    def loop(self) -> None:
        import cv2
        from .capture import ThreadedCapture

        cap = ThreadedCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source {self.source}")
        t0 = time.monotonic()
        last_seq = -1
        try:
            while not self.stopping:
                frame_start = time.monotonic()
                ret, frame_bgr, seq = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                last_seq = seq
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ts_ms = int((time.monotonic() - t0) * 1000)
                results = self.estimator.estimate(rgb, ts_ms)

                disp = None
                if self.show_preview:
                    disp = cv2.flip(frame_bgr, 1) if self.mirror_preview else frame_bgr

                for r in results:
                    if r.track_id is not None:
                        sid = f"{self.source_id}#p{r.track_id}"
                    else:
                        sid = self.source_id if r.face_index == 0 else f"{self.source_id}:{r.face_index}"
                    self.emit({
                        "yaw": r.yaw, "pitch": r.pitch,
                        "head_yaw": r.head_yaw, "head_pitch": r.head_pitch,
                        "eye_h": r.eye_h, "eye_v": r.eye_v,
                        "blink": r.blink,
                    }, subject_id=sid)
                    if self.show_preview:
                        # cv2.putText(frame_bgr,
                                    # f"yaw:{r.yaw:+.0f} pitch:{r.pitch:+.0f} blink:{r.blink:.2f}",
                                    # (10, 30 + 24 * r.face_index), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        target = self.target_fn(r) if self.target_fn else None
                        _draw_gaze_overlay(cv2, disp, r, target, self.mirror_preview)
                        
                if self.show_preview:
                    cv2.imshow(f"Gaze - {self.source_id}", cv2.resize(frame_bgr, (1280, 768)))
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        self.stop()
                dt = time.monotonic() - frame_start
                if dt < self.min_dt:
                    time.sleep(self.min_dt - dt)
        finally:
            cap.release()
            if self.show_preview:
                cv2.destroyAllWindows()
