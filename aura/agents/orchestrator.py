"""Orchestrator: decides WHICH agent runs WHEN.

Event-driven trigger policy (not naive polling):
  * EngagementAnalyst : engagement slope below threshold OR periodic (60 s)
  * Moderator         : PauseDetected with pending questions, or SlideChange
  * PresenterCoach    : Analyst reports medium/high severity (90 s cooldown)
  * Summarizer        : SessionEnd

Cooldowns + a per-minute action cap keep the presenter HUD calm — agent
over-triggering is the top UX risk in the risk register.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..bus import EventBus
from ..events import AgentAction, PauseDetected, SessionEnd, SlideChange
from ..fusion.state import RoomStateBuilder
from .crew import EngagementAnalyst, Moderator, PresenterCoach, Summarizer
from .llm import LLMClient

log = logging.getLogger("aura.orchestrator")

SLOPE_TRIGGER = -0.15        # engagement loss per minute
ANALYST_PERIOD_S = 60.0
COACH_COOLDOWN_S = 45.0
MAX_ACTIONS_PER_MIN = 4
TICK_S = 5.0


class Orchestrator:
    def __init__(self, bus: EventBus, fusion: RoomStateBuilder,
                 llm: LLMClient | None = None, tick_s: float = TICK_S) -> None:
        self.bus = bus
        self.fusion = fusion
        self.tick_s = tick_s
        self.llm = llm or LLMClient()
        self.analyst = EngagementAnalyst(self.llm)
        self.moderator = Moderator(self.llm)
        self.coach = PresenterCoach(self.llm)
        self.summarizer = Summarizer(self.llm)

        self._last_analyst = 0.0
        self._last_coach = 0.0
        self._recent_actions: list[float] = []
        self._timeline: list[str] = []
        self._stop = asyncio.Event()

        bus.subscribe(PauseDetected, self._on_pause)
        bus.subscribe(SlideChange, self._on_slide)
        bus.subscribe(SessionEnd, self._on_end)

    # ------------------------------------------------------------ emission
    def _budget_ok(self) -> bool:
        now = time.time()
        self._recent_actions = [t for t in self._recent_actions if now - t < 60]
        return len(self._recent_actions) < MAX_ACTIONS_PER_MIN

    async def _emit(self, action: AgentAction) -> None:
        if action.action == "noop":
            return
        if action.action != "debrief" and not self._budget_ok():
            log.info("action budget exhausted; dropping %s", action.agent)
            return
        self._recent_actions.append(time.time())
        self._timeline.append(
            f"[slide {self.fusion.snapshot().slide}] {action.agent}: {action.message}")
        await self.bus.publish(action)

    # ------------------------------------------------------------ triggers
    async def _on_pause(self, _e: PauseDetected) -> None:
        state = self.fusion.snapshot()
        action = await self.moderator.run(state, pause=True)
        if action.action == "surface_questions":
            self.fusion.resolve_questions(action.payload.get("question_ids", []))
        await self._emit(action)

    async def _on_slide(self, e: SlideChange) -> None:
        self._timeline.append(f"--- slide {e.slide}: {e.title}")
        state = self.fusion.snapshot()
        if state.pending_questions:
            action = await self.moderator.run(state, pause=True)
            if action.action == "surface_questions":
                self.fusion.resolve_questions(action.payload.get("question_ids", []))
            await self._emit(action)

    async def _on_end(self, _e: SessionEnd) -> None:
        state = self.fusion.snapshot()
        if not self.fusion.slide_stats() and state.people_present == 0:
            log.warning("session had NO attention and NO slide data — the "
                        "camera (ingest_room_ar) and/or screen events were "
                        "not merged into this events.jsonl. The debrief will "
                        "be near-empty. Re-merge the camera stream.")
        import json as _json
        timeline = ("SLIDE_STATS:" + _json.dumps(self.fusion.slide_stats())
                    + "\nEVENTS:\n" + "\n".join(self._timeline))
        action = await self.summarizer.run(state, timeline=timeline)
        # guarantee per-slide data in the debrief even if the LLM omits it
        action.payload.setdefault("per_slide", []) or action.payload.update(
            per_slide=[{"slide": s["slide"], "engagement": s["engagement"],
                        "note": s["title"]
                        + (f" — {s['questions']} question(s)" if s["questions"] else "")}
                       for s in self.fusion.slide_stats()])
        await self.bus.publish(action)          # debrief bypasses the budget
        self._stop.set()

    # ------------------------------------------------------------ main loop
    async def run(self) -> None:
        log.info("orchestrator started (mock_llm=%s)", self.llm.mock)
        while not self._stop.is_set():
            await asyncio.sleep(self.tick_s)
            state = self.fusion.snapshot()
            now = time.time()
            slope_alarm = (state.engagement_slope < SLOPE_TRIGGER
                           and state.people_present > 0
                           and now - self._last_analyst > self.tick_s * 10)
            periodic = now - self._last_analyst > ANALYST_PERIOD_S
            if slope_alarm or periodic:
                self._last_analyst = now
                insight = await self.analyst.run(state)
                finding = insight.payload.get("finding", "stable")
                severity = insight.payload.get("severity", "low")
                # surface the analyst's reasoning whenever it found something
                if finding != "stable" or slope_alarm:
                    await self._emit(insight)
                # coach follows the analyst on any real problem
                if (finding == "engagement_drop" or severity in ("medium", "high")) \
                        and now - self._last_coach > COACH_COOLDOWN_S:
                    self._last_coach = now
                    await self._emit(await self.coach.run(state))
        await self.bus.drain()

    @property
    def finished(self) -> asyncio.Event:
        return self._stop
