"""
Plug-in-play ABC for modalities in the attention room framework.
"""

import abc
import logging
import threading
from typing import Any, Optional

from .bus import Bus
from .observation import Observation

log = logging.getLogger(__name__)

class Modality(abc.ABC):
    name: str = "modality"

    def __init__(self, bus: Bus, source_id: str) -> None:
        self.bus = bus
        self.source_id = source_id
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @abc.abstractmethod
    def loop(self) -> None:
        """The main loop of the modality, which should run until self._stop is set."""
        pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self.loop, name=self.name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()
    
    def emit(self, payload: dict[str, Any], subject_id: Optional[str] = None) -> None:
        obs = Observation(modality=self.name, source_id=self.source_id, subject_id=subject_id, payload=payload)
        self.bus.publish(obs)

    def _run(self) -> None:
        try:
            self.loop()
        except Exception as e:
            log.exception(f"Error in modality {self.name}: {e}")
