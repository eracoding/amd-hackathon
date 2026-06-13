"""
The message bus for the attention room framework, handling the flow of observations and attention states between modalities and the attention engine.
"""

import threading
from typing import Callable, Iterable, Optional

from .observation import Observation

Subcriber = Callable[[Observation], None]

class Bus:
    """Interface for the message bus, allowing modalities to publish observations and subscribe to updates."""

    def publish(self, obs: Observation) -> None:
        """Publish an observation to the bus."""
        raise NotImplementedError
    
    def subscribe(self, callback: Subcriber, modalities: Optional[Iterable[str]] = None) -> None:
        """Subscribe to receive updates from the bus."""
        raise NotImplementedError
    
class LocalBus(Bus):
    """A simple in-memory implementation of the Bus interface, using a list of subscribers and a lock for thread safety."""

    def __init__(self) -> None:
        self._subs: list[tuple[Optional[set[str]], Subcriber]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Subcriber, modalities: Optional[Iterable[str]] = None) -> None:
        flt = set(modalities) if modalities is not None else None
        with self._lock:
            self._subs.append((flt, callback))
    
    def publish(self, obs: Observation) -> None:
        with self._lock:
            subs = list(self._subs)
        for flt, callback in subs:
            if flt is None or obs.modality in flt:
                callback(obs)
