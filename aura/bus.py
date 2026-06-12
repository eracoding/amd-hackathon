"""Async in-process pub/sub event bus with session recording.

Topics are event class names. Subscribers register coroutines; publishing is
non-blocking (handlers are scheduled as tasks). A SessionRecorder subscriber
persists every event to JSONL, enabling deterministic replay for evaluation.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable

from .events import EVENT_TYPES, BaseEvent

log = logging.getLogger("aura.bus")

Handler = Callable[[BaseEvent], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = {}
        self._wildcard: list[Handler] = []
        self._tasks: set[asyncio.Task] = set()

    def subscribe(self, topic: str | type[BaseEvent], handler: Handler) -> None:
        name = topic if isinstance(topic, str) else topic.__name__
        if name == "*":
            self._wildcard.append(handler)
        else:
            self._subs.setdefault(name, []).append(handler)

    async def publish(self, event: BaseEvent) -> None:
        handlers = self._subs.get(event.topic, []) + self._wildcard
        for h in handlers:
            task = asyncio.create_task(self._safe(h, event))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def _safe(handler: Handler, event: BaseEvent) -> None:
        try:
            await handler(event)
        except Exception:  # noqa: BLE001 — a bad handler must not kill the bus
            log.exception("handler %s failed on %s", handler, event.topic)

    async def drain(self) -> None:
        """Wait for all in-flight handlers (used in tests / shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
            # gather over already-done tasks completes without yielding, which
            # would starve the call_soon discard callbacks -> prune + yield.
            self._tasks = {t for t in self._tasks if not t.done()}
            await asyncio.sleep(0)


class SessionRecorder:
    """Persists every bus event to JSONL for replay and offline evaluation."""

    def __init__(self, bus: EventBus, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        bus.subscribe("*", self._on_event)

    async def _on_event(self, event: BaseEvent) -> None:
        rec = {"topic": event.topic, **event.model_dump()}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


async def replay(bus: EventBus, path: str | Path, speed: float = 0.0,
                 rebase: bool = True) -> None:
    """Re-publish a recorded session. speed=0 → as fast as possible;
    speed=1 → real time; speed=2 → twice real time.

    rebase=True shifts (and, for speed>0, compresses) timestamps so the
    first event lands at `now` — required for fusion's wall-clock windows
    to work on recordings made in the past."""
    import time as _time
    first: float | None = None
    start = _time.time()
    prev_ts: float | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        topic = rec.pop("topic")
        cls = EVENT_TYPES.get(topic)
        if cls is None:
            continue
        event = cls(**rec)
        if first is None:
            first = event.ts
        if speed > 0 and prev_ts is not None:
            await asyncio.sleep(max(0.0, (event.ts - prev_ts) / speed))
        prev_ts = event.ts
        if rebase:
            rel = event.ts - first
            event.ts = start + (rel / speed if speed > 0 else rel)
        await bus.publish(event)
    await bus.drain()
