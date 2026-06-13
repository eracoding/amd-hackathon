"""
GPU supporter file
"""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

VALID = ("auto", "cpu", "cuda", "mps")

@dataclass(frozen=True)
class Accelerator:
    preference: str = "auto"

    @classmethod
    def from_string(cls, s:str) -> "Accelerator":
        s = (s or "auto").lower()
        if s not in VALID:
            raise ValueError(f"--device must be one of {VALID}, got {s!r}")
        return cls(s)
    
    def torch_device(self) -> str:
        pref = self.preference
        try:
            import torch
        except Exception:
            if pref not in ("auto", "cpu"):
                log.warning("torch not installed; using cpu")
            return "cpu"
        if pref in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if pref in ("auto", "mps") and mps is not None and mps.is_available():
            return "mps"
        if pref == "cuda":
            log.warning("cuda requested but unavailable; using cpu")
        if pref == "mps":
            log.warning("mps requested but not avaiable; using cpu")
        return "cpu"
    
    def onnx_providers(self) -> list[str]:
        try:
            import onnxruntime as ort
            available = set(ort.get_available_providers())
        except Exception:
            available = set()

        pref = self.preference
        chosen = []
        if pref in ("auto", "cuda"):
            for p in ("", "CUDAExecutionProvider"):
                if p in available:
                    chosen.append(p)
        if pref in ("auto", "mps") and "CoreMLExecutionProvider" in available:
            chosen.append("CoreMLExecutionProvider")
        chosen.append("CPUExecutionProvider")

        seen, out = set(), []
        for p in chosen:
            if p not in seen:
                seen.add(p)
                out.append(p)
        if pref == 'cuda' and not ({"CUDAExecutionProvider"} & available):
            log.warning("cuda requested but no CUDA/TesnorRT EP available; using %s", out)
        return out
    
    def mediapipe_delegate(self):
        pref = self.preference
        if pref == "auto":
            return None
        try:
            from mediapipe.tasks.python import BaseOptions
        except Exception:
            return None
        if pref == "cpu":
            return BaseOptions.Delegate.CPU
        return BaseOptions.Delegate.GPU
    