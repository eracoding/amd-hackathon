"""
Gaze estimators for processing gaze data and estimating attention based on gaze patterns.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...config import GazeConfig
from ...core.device import Accelerator


@dataclass
class GazeResult:
    face_index: int
    yaw: float
    pitch: float
    head_yaw: float
    head_pitch: float
    eye_h: float
    eye_v: float
    blink: float
    origin: tuple[float, float] = 0.5, 0.5
    track_id: Optional[int] = None

class GazeEstimator:
    def estimate(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[GazeResult]:
        raise NotImplementedError
    
class MediaPipeGazeEstimator(GazeEstimator):
    def __init__(self, model_path: str, cfg: GazeConfig, max_faces: int = 4,
                 accelerator: Optional[Accelerator]= None) -> None:
        self.cfg = cfg
        from .tracking import FaceTracker
        self._tracker = FaceTracker()

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        base_kwargs = {"model_asset_path": model_path}
        delegate = (accelerator or Accelerator()).mediapipe_delegate()
        if delegate is not None:
            base_kwargs["delegate"] = delegate

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(**base_kwargs),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def estimate(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[GazeResult]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self._landmarker.detect_for_video(image, int(timestamp_ms))

        faces = []
        for i in range(len(res.face_landmarks)):
            blend = {}
            if res.face_blendshapes:
                blend = {c.category_name: c.score for c in res.face_blendshapes[i]}
            mat = None
            if res.facial_transformation_matrixes:
                mat = np.asarray(res.facial_transformation_matrixes[i], dtype=float)
            
            head_yaw, head_pitch = self._head_angles(mat)
            eye_h, eye_v, blink = self._eye_offsets(blend)

            lms = res.face_landmarks[i]
            if len(lms) >= 478:
                origin = ((lms[468].x + lms[473].x) / 2.0, (lms[468].y + lms[473].y) / 2.0)
            elif lms:
                origin = (lms[1].x, lms[1].y)
            else:
                origin = (0.5, 0.5)

            if lms:
                cx = sum(p.x for p in lms) / len(lms)
                cy = sum(p.y for p in lms) / len(lms)
            
            else:
                cx, cy = origin

            yaw = head_yaw + self.cfg.eye_gain_deg * eye_h * self.cfg.eye_h_sign
            pitch = head_pitch + self.cfg.eye_gain_deg * eye_v * self.cfg.eye_v_sign
            faces.append((yaw, pitch, head_yaw, head_pitch, eye_h, eye_v, blink, origin, (cx, cy)))

        track_ids = self._tracker.update([f[8] for f in faces])
        results: list[GazeResult] = []
        for i, f in enumerate(faces):
            yaw, pitch, head_yaw, head_pitch, eye_h, eye_v, blink, origin, _ = f
            results.append(GazeResult(
                face_index=i, yaw=yaw, pitch=pitch,
                head_yaw=head_yaw, head_pitch=head_pitch,
                eye_h=eye_h, eye_v=eye_v, blink=blink, origin=origin,
                track_id=track_ids[i]
            ))
        
        return results
    
    def _head_angles(self, mat: Optional[np.ndarray]) -> tuple[float, float]:
        if mat is None or mat.shape != (4, 4):
            return 0.0, 0.0
        rot = mat[:3, :3]
        fwd = rot @ np.array([0.0, 0.0, self.cfg.head_forward_sign])
        yaw = math.degrees(math.atan2(fwd[0], abs(fwd[2]) + 1e-6)) * self.cfg.head_yaw_sign
        pitch = math.degrees(math.atan2(fwd[1], abs(fwd[2]) + 1e-6)) * self.cfg.head_pitch_sign
        return yaw, pitch
    
    @staticmethod
    def _eye_offsets(b: dict[str, float]) -> tuple[float, float, float]:
        def g(k: str) -> float:
            return float(b.get(k, 0.0))
        
        right = (g("eyeLookOutRight") + g("eyeLookInLeft")) / 2.0
        left = (g("eyeLookInRight") + g("eyeLookOutLeft")) / 2.0
        eye_h = right - left
        eye_v = (g("eyeLookUpLeft") + g("eyeLookUpRight") - g("eyeLookDownLeft") - g("eyeLookDownRight")) / 2.0
        blink = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2.0
        return eye_h, eye_v, blink
