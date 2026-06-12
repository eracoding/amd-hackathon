"""Typed event contracts flowing through the AURA event bus.

Every perception module emits one of these; the fusion layer consumes them.
Anonymity by design: people are tracklets ("person_0"), never identities.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


def _now() -> float:
    return time.time()


def _eid() -> str:
    return uuid.uuid4().hex[:12]


class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=_eid)
    ts: float = Field(default_factory=_now)
    source: str = "unknown"

    @property
    def topic(self) -> str:
        return self.__class__.__name__


class AttentionEvent(BaseEvent):
    """Per-person attention snapshot from the camera pipeline."""
    person_id: str                      # anonymous tracklet id
    attention: float = Field(ge=0, le=1)  # 1.0 = locked on screen
    yaw: float = 0.0                    # head pose, degrees
    pitch: float = 0.0
    present: bool = True


class TranscriptSegment(BaseEvent):
    """Timestamped ASR output from the presenter microphone."""
    text: str
    t_start: float
    t_end: float
    words_per_min: Optional[float] = None


class PauseDetected(BaseEvent):
    """Presenter silence >= threshold — a natural interruption point."""
    silence_s: float


class InteractionKind(str, Enum):
    question = "question"
    comment = "comment"
    reaction = "reaction"


class InteractionEvent(BaseEvent):
    """Participant device interaction (tablet/laptop), anchored to a slide."""
    person_id: str
    kind: InteractionKind
    text: str = ""
    slide: int = 0
    resolved: bool = False


class ScreenAnnotationEvent(BaseEvent):
    """Drawing/typing detected ON the shared slide (Teams/PowerPoint Live
    annotation mode), found by diffing the captured screen against the clean
    deck page. Coordinates are normalized 0-1 relative to the slide."""
    slide: int
    bbox: list[float]                  # [x0, y0, x1, y1] normalized
    area_frac: float = 0.0             # ink area / slide area
    kind: str = "drawn"                # drawn | typed (heuristic/VLM later)
    text: str = ""                     # OCR/VLM-inferred, if available
    patch_path: str = ""               # cropped PNG of the annotation


class SlideChange(BaseEvent):
    slide: int
    title: str = ""
    content: str = ""   # slide text (from deck/VLM) so agents see what's on screen


class SessionEnd(BaseEvent):
    reason: str = "completed"


class AgentAction(BaseEvent):
    """Output of the agentic layer, consumed by the action sink."""
    agent: str
    action: Literal["nudge", "surface_questions", "insight", "debrief", "noop"]
    priority: Literal["low", "medium", "high"] = "medium"
    message: str = ""
    payload: dict = Field(default_factory=dict)


EVENT_TYPES = {
    cls.__name__: cls
    for cls in (
        AttentionEvent, TranscriptSegment, PauseDetected,
        InteractionEvent, SlideChange, ScreenAnnotationEvent,
        SessionEnd, AgentAction,
    )
}
