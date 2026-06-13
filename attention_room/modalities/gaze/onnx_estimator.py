"""
GPU-transfer of gaze
"""
import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ...config import GazeConfig
from ...core.device import Accelerator
from .estimators import GazeEstimator, GazeResult

log = logging.getLogger(__name__)

@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    origin: tuple[float, float] # normalized x, y
    track_id: Optional[int] = None

FaceBoxProvider = Callable[[np.ndarray, int], "list[FaceBox]"]

class MediaPipeFaceBoxProvider:
    def __init__(self, model_path: str, max_faces: int = 4) -> None:
        from .tracking import FaceTracker
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._tracker = FaceTracker()
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=max_faces,
            )
        )

    def __call__(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[FaceBox]:
        h, w = frame_rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        res = self._landmarker.detect_for_video(image, int(timestamp_ms))
        boxes: list[FaceBox] = []
        for lms in res.face_landmarks:
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            x1, x2 = max(0, int(min(xs) * w)), min(w, int(max(xs) * w))
            y1, y2 = max(0, int(min(ys) * h)), min(h, int(max(ys) * h))
            if x2 <= x1 or y2 <= y1:
                continue

            if len(lms) >= 478:
                origin = ((lms[468].x + lms[473].x) / 2.0,
                          (lms[468].y + lms[473].y) / 2.0)
            else:
                origin = ((x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h))

            boxes.append(FaceBox(x1, y1, x2, y2, origin))
        ids = self._tracker.update([((b.x1 + b.x2) / (2.0 * w), (b.y1 + b.y2) / (2.0 * h)) for b in boxes])

        for b, tid in zip(boxes, ids):
            b.track_id = tid

        return boxes
    

class OnnxGazeEstimator(GazeEstimator):
    def __init__(self, model_path: str, cfg: GazeConfig,
                 accelerator: Optional[Accelerator] = None,
                 detector_model: Optional[str] = None,
                 face_box_provider: Optional[FaceBoxProvider] = None,
                 max_faces: int = 4,
                 input_size: tuple[int, int] = (448, 448),
                 mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: tuple[float, float, float] = (0.229, 0.224, 0.225),
                 bins: int = 90, binwidth: float = 4.0, angle_offset: float = 180.0,
                 output_is_radians: bool = False) -> None:
        self.cfg = cfg
        self.max_faces = max_faces
        self.input_size = input_size
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.bins = bins
        self.binwidth = binwidth
        self.angle_offset = angle_offset
        self.output_is_radians = output_is_radians
        self._idx = np.arange(bins, dtype=np.float32)

        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try: ort.preload_dlls()
            except Exception:
                pass
        providers = (accelerator or Accelerator()).onnx_providers()
        print("BEFORE PROVIDERS:", providers)
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._input = self._sess.get_inputs()[0].name
        self._output_names = [o.name for o in self._sess.get_outputs()]
        self._binned = len(self._output_names) == 2
        batch_dim = self._sess.get_inputs()[0].shape[0]
        self._dynamic_batch = not (isinstance(batch_dim, int) and batch_dim >= 1)
        log.info("OnnxGazeEstimator running on %s | %s outputs | dynamic_batch=%s", self._sess.get_providers(), "binned" if self._binned else "regression", self._dynamic_batch)

        if face_box_provider is not None:
            self._boxes = face_box_provider
        elif detector_model is not None:
            self._boxes = MediaPipeFaceBoxProvider(detector_model, max_faces)
        else:
            raise ValueError("pass detector_model=face_landmarrker.task or a face_box_provider")
        
    def estimate(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[GazeResult]:
        boxes = self._boxes(frame_rgb, timestamp_ms)[:self.max_faces]
        if not boxes:
            return []
        crops = [self._preprocess(frame_rgb, b) for b in boxes]
        if self._dynamic_batch:
            outs = self._sess.run(self._output_names, {self._input: np.concatenate(crops, axis=0)})
            yaws, pitches = self._decode(outs)
        else:
            yaws, pitches = [], []
            for c in crops:
                y, p = self._decode(self._sess.run(self._output_names, {self._input: c}))
                yaws.append(float(y[0]))
                pitches.append(float(p[0]))
            yaws, pitches = np.asarray(yaws), np.asarray(pitches)

        results = []
        for i, b in enumerate(boxes):
            yaw = float(yaws[i]) * self.cfg.head_yaw_sign
            pitch = float(pitches[i]) * self.cfg.head_pitch_sign
            results.append(GazeResult(
                face_index=i, yaw=yaw, pitch=pitch,
                head_yaw=yaw, head_pitch=pitch,
                eye_h=0.0, eye_v=0.0, blink=0.0, 
                origin=b.origin, track_id=b.track_id,
            ))
        
        return results
    
    def _preprocess(self, frame_rgb: np.ndarray, b: FaceBox) -> np.ndarray:
        import cv2
        crop = frame_rgb[b.y1: b.y2, b.x1: b.x2]
        crop = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        x = crop.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return (x - self._mean) / self._std
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    
    def _decode(self, outs: list) -> tuple[np.ndarray, np.ndarray]:
        if self._binned:
            yaw_logits, pitch_logtis = np.asarray(outs[0]), np.asarray(outs[1])
            yaw = (self._softmax(yaw_logits) * self._idx).sum(axis=1) * self.binwidth - self.angle_offset
            pitch = (self._softmax(pitch_logtis) * self._idx).sum(axis=1) * self.binwidth - self.angle_offset
            return yaw, pitch
        
        row = np.asarray(outs[0]).reshape(-1, 2)
        yaw, pitch = row[:, 0], row[:, 1]
        if self.output_is_radians:
            yaw, pitch = math.degrees(yaw), math.degrees(pitch)
        return yaw, pitch
