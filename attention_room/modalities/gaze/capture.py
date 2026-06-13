"""
OpenCV buffers incoming RTSP frames - fix for this
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np


class ThreadedCapture:
    def __init__(self, source, buffersize: int = 1) -> None:
        import cv2

        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self._cap = cv2.VideoCapture(source)

        try: self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
        except Exception:
            pass
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._stop = threading.Event()
        self._opened = self._cap.isOpened()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        if self._opened:
            self._thread.start()
        
    def isOpened(self) -> bool:
        return self._opened
    
    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
        
    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return False, None, self._seq
            return True, self._frame, self._seq
    
    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()
