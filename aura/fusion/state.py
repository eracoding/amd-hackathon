"""Fusion layer: collapses heterogeneous event streams into RoomState snapshots.

Dual temporal windows:
  * tactical (10 s)  — instantaneous engagement, speech rate
  * strategic (60 s) — engagement slope, transcript context

Serialization is deliberately compact (<= ~700 tokens) because the agentic
layer's token budget is a first-class metric in the evaluation criteria.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..bus import EventBus
from ..events import (
    AttentionEvent, InteractionEvent, ScreenAnnotationEvent,
    ScreenStateEvent, SlideChange, TranscriptSegment,
)

TACTICAL_S = 10.0
STRATEGIC_S = 60.0


@dataclass
class RoomState:
    ts: float
    slide: int
    slide_title: str
    slide_content: str
    people_present: int
    engagement: float            # tactical window mean, 0..1
    engagement_60s: float        # strategic window mean
    engagement_slope: float      # per-minute delta of tactical engagement
    low_attention_ids: list[str]
    speech_wpm: float
    transcript_tail: str         # last ~60 s of presenter speech
    pending_questions: list[dict]
    recent_reactions: int
    recent_annotations: int
    screen_state: dict | None = None   # off-deck content (VLM-explained)
    annotations: dict = field(default_factory=dict)  # agent blackboard

    def to_llm_json(self) -> dict:
        """Compact, token-efficient view for agent prompts."""
        return {
            "slide": {"n": self.slide, "title": self.slide_title,
                      "content": self.slide_content[:240]},
            "people": self.people_present,
            "engagement": {
                "now": round(self.engagement, 2),
                "last_60s": round(self.engagement_60s, 2),
                "slope_per_min": round(self.engagement_slope, 2),
                "low_attention": self.low_attention_ids[:6],
            },
            "speech": {
                "wpm": round(self.speech_wpm),
                "recent": self.transcript_tail[-600:],
            },
            "pending_questions": [
                {"id": q["event_id"], "slide": q["slide"], "text": q["text"][:160]}
                for q in self.pending_questions[:8]
            ],
            "reactions_recent": self.recent_reactions,
            "annotations_recent": self.recent_annotations,
            "off_deck_screen": self.screen_state,
            "notes": self.annotations,
        }


class RoomStateBuilder:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._attn: deque[AttentionEvent] = deque()
        self._speech: deque[TranscriptSegment] = deque()
        self._reactions: deque[float] = deque()
        self._annotations: deque[float] = deque()
        self._questions: list[dict] = []      # unresolved
        self._slide, self._slide_title = 0, ""
        self._slide_content = ""
        self._screen_state: dict | None = None
        self._eng_history: deque[tuple[float, float]] = deque(maxlen=240)
        # per-slide analytics (feeds the Summarizer & debrief)
        self._slide_acc: dict[int, dict] = {}

        bus.subscribe(AttentionEvent, self._on_attn)
        bus.subscribe(TranscriptSegment, self._on_speech)
        bus.subscribe(InteractionEvent, self._on_interaction)
        bus.subscribe(SlideChange, self._on_slide)
        bus.subscribe(ScreenAnnotationEvent, self._on_annotation)
        bus.subscribe(ScreenStateEvent, self._on_screen_state)

    # ------------------------------------------------------------ handlers
    async def _on_attn(self, e: AttentionEvent) -> None:
        self._attn.append(e)
        acc = self._slide_acc.setdefault(
            self._slide, {"title": self._slide_title, "sum": 0.0, "n": 0,
                          "questions": 0, "min": 1.0})
        acc["sum"] += e.attention
        acc["n"] += 1
        acc["min"] = min(acc["min"], e.attention)

    async def _on_speech(self, e: TranscriptSegment) -> None:
        self._speech.append(e)

    async def _on_interaction(self, e: InteractionEvent) -> None:
        if e.kind.value == "question":
            d = e.model_dump()
            if d.get("slide", 0) == 0:        # voice questions carry no anchor
                d["slide"] = self._slide
            self._questions.append(d)
            e = e.model_copy(update={"slide": d["slide"]})
            acc = self._slide_acc.setdefault(
                e.slide, {"title": "", "sum": 0.0, "n": 0,
                          "questions": 0, "min": 1.0})
            acc["questions"] += 1
        elif e.kind.value == "reaction":
            self._reactions.append(e.ts)

    async def _on_annotation(self, e: ScreenAnnotationEvent) -> None:
        self._annotations.append(e.ts)
        acc = self._slide_acc.setdefault(
            e.slide, {"title": "", "sum": 0.0, "n": 0,
                      "questions": 0, "min": 1.0})
        acc["annotations"] = acc.get("annotations", 0) + 1

    async def _on_screen_state(self, e: ScreenStateEvent) -> None:
        self._screen_state = {"kind": e.kind, "summary": e.summary}

    async def _on_slide(self, e: SlideChange) -> None:
        self._screen_state = None      # back on the deck
        self._slide, self._slide_title = e.slide, e.title
        self._slide_content = e.content
        self._slide_acc.setdefault(
            e.slide, {"title": e.title, "sum": 0.0, "n": 0,
                      "questions": 0, "min": 1.0})["title"] = e.title

    def slide_stats(self) -> list[dict]:
        """Finalized per-slide engagement summary for the debrief."""
        out = []
        for slide in sorted(self._slide_acc):
            a = self._slide_acc[slide]
            if (a["n"] == 0 and a["questions"] == 0) or slide == 0:
                continue  # slide 0 = events before the first SlideChange
            out.append({
                "slide": slide,
                "title": a["title"],
                "engagement": round(a["sum"] / a["n"], 2) if a["n"] else None,
                "min_engagement": round(a["min"], 2) if a["n"] else None,
                "questions": a["questions"],
                "annotations": a.get("annotations", 0),
            })
        return out

    def resolve_questions(self, event_ids: list[str]) -> None:
        self._questions = [q for q in self._questions
                           if q["event_id"] not in event_ids]

    # ------------------------------------------------------------ snapshot
    def _trim(self, now: float) -> None:
        while self._attn and now - self._attn[0].ts > STRATEGIC_S:
            self._attn.popleft()
        while self._speech and now - self._speech[0].ts > STRATEGIC_S:
            self._speech.popleft()
        while self._reactions and now - self._reactions[0] > TACTICAL_S * 3:
            self._reactions.popleft()
        while self._annotations and now - self._annotations[0] > STRATEGIC_S:
            self._annotations.popleft()

    def snapshot(self) -> RoomState:
        now = time.time()
        self._trim(now)

        tactical = [e for e in self._attn if now - e.ts <= TACTICAL_S]
        per_person: dict[str, list[float]] = {}
        for e in tactical:
            per_person.setdefault(e.person_id, []).append(e.attention)
        person_means = {p: sum(v) / len(v) for p, v in per_person.items()}

        eng_now = (sum(person_means.values()) / len(person_means)) if person_means else 0.0
        strat_vals = [e.attention for e in self._attn]
        eng_60 = sum(strat_vals) / len(strat_vals) if strat_vals else 0.0

        if person_means:  # never record empty-room zeros — they poison the slope
            self._eng_history.append((now, eng_now))
        slope = 0.0
        recent = [(t, v) for t, v in self._eng_history if now - t <= 30.0]
        if len(recent) >= 4:
            (t0, v0), (t1, v1) = recent[0], recent[-1]
            dt_min = max(1e-6, (t1 - t0) / 60.0)
            slope = (v1 - v0) / dt_min

        recent_speech = [s for s in self._speech if now - s.ts <= STRATEGIC_S]
        wpm_vals = [s.words_per_min for s in recent_speech if s.words_per_min]
        tail = " ".join(s.text for s in recent_speech)

        return RoomState(
            ts=now, slide=self._slide, slide_title=self._slide_title,
            slide_content=self._slide_content,
            people_present=len(person_means),
            engagement=eng_now, engagement_60s=eng_60, engagement_slope=slope,
            low_attention_ids=sorted(p for p, m in person_means.items() if m < 0.4),
            speech_wpm=(sum(wpm_vals) / len(wpm_vals)) if wpm_vals else 0.0,
            transcript_tail=tail,
            pending_questions=list(self._questions),
            recent_reactions=len(self._reactions),
            recent_annotations=len(self._annotations),
            screen_state=self._screen_state,
        )
