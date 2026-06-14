"""The AURA agent crew.

Each agent: a system prompt defining role + strict JSON contract, a `run`
method consuming the compact RoomState, returning a validated AgentAction.
Agents communicate through the RoomState's `annotations` blackboard — small,
auditable, token-cheap — instead of long chat histories.
"""
from __future__ import annotations

import json
import logging

from ..events import AgentAction
from ..fusion.state import RoomState
from .llm import LLMClient

log = logging.getLogger("aura.agents")


class Agent:
    name: str = "agent"
    system: str = ""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def ask(self, state: RoomState, extra: str = "") -> dict:
        user = (extra + "\nRoom state:\n"
                + json.dumps(state.to_llm_json(), ensure_ascii=False))
        return await self.llm.chat_json(self.system, user)


class EngagementAnalyst(Agent):
    name = "EngagementAnalyst"
    system = (
        "You are EngagementAnalyst inside AURA, watching a live presentation. "
        "Input: a JSON room state with per-window engagement, slope, speech "
        "rate (wpm), the recent transcript, the current slide, and any "
        "off-deck screen content. Diagnose the room's attention and explain "
        "WHY, citing concrete evidence from the transcript or pace. Respond "
        "ONLY with JSON: "
        '{"finding": "engagement_drop|stable|recovering|high_engagement", '
        '"confidence": 0..1, '
        '"probable_cause": "<specific cause, e.g. \'pace spiked to 190 wpm '
        "while covering dense architecture on slide 4'>\", "
        '"evidence": "<short quote or metric from the room state>", '
        '"severity": "low|medium|high"}'
    )

    async def run(self, state: RoomState) -> AgentAction:
        out = await self.ask(state)
        state.annotations["analyst"] = out
        cause = out.get("probable_cause", "")
        ev = out.get("evidence", "")
        msg = cause + (f"  ({ev})" if ev else "")
        return AgentAction(
            agent=self.name, action="insight",
            priority="high" if out.get("severity") == "high" else "medium",
            message=msg or out.get("finding", "?"),
            payload=out, source="agent",
        )


class Moderator(Agent):
    name = "Moderator"
    system = (
        "You are Moderator inside AURA. You hold participants' typed questions "
        "and decide WHEN to surface them: only at natural pauses or slide "
        "transitions, never mid-sentence. Cluster duplicates. Respond ONLY with "
        'JSON: {"action": "surface_questions|noop", "question_ids": [str], '
        '"summary": str}'
    )

    async def run(self, state: RoomState, pause: bool = False) -> AgentAction:
        if not state.pending_questions:
            return AgentAction(agent=self.name, action="noop", source="agent")
        extra = ("The presenter just PAUSED — this is a natural moment to "
                 "surface questions." if pause else
                 "No pause detected; surface only if questions are stale or urgent.")
        out = await self.ask(state, extra=extra)
        if out.get("action") != "surface_questions":
            return AgentAction(agent=self.name, action="noop", source="agent")
        state.annotations["moderator"] = out
        return AgentAction(
            agent=self.name, action="surface_questions", priority="high",
            message=out.get("summary", ""),
            payload={"question_ids": out.get("question_ids", [])},
            source="agent",
        )


class PresenterCoach(Agent):
    name = "PresenterCoach"
    system = (
        "You are PresenterCoach inside AURA. You see the room state and the "
        "Analyst's diagnosis (in notes.analyst). Give the presenter ONE "
        "specific, actionable piece of coaching that names WHAT went wrong "
        "and WHAT to do about it — grounded in the actual evidence, not "
        "generic advice. Good: 'You hit ~190 wpm on the architecture slide "
        "and 2 of 3 people looked down — pause, restate the key idea in one "
        "plain sentence, then ask if it landed.' Bad: 'Slow down.' Respond "
        'ONLY with JSON: {"action": "nudge|noop", '
        '"message": "<= 40 words, specific", '
        '"what_went_wrong": "<short>", "priority": "low|medium|high"}'
    )

    async def run(self, state: RoomState) -> AgentAction:
        out = await self.ask(state)
        if out.get("action") != "nudge":
            return AgentAction(agent=self.name, action="noop", source="agent")
        return AgentAction(
            agent=self.name, action="nudge",
            priority=out.get("priority", "medium"),
            message=out.get("message", ""), payload=out, source="agent",
        )


class Summarizer(Agent):
    name = "Summarizer"
    system = (
        "You are Summarizer inside AURA. Produce the end-of-session debrief "
        "from the final room state and the session timeline provided. Respond "
        'ONLY with JSON: {"summary": str, "per_slide": '
        '[{"slide": int, "engagement": float, "note": str}], '
        '"unresolved_questions": [str], "action_items": [str]}'
    )

    async def run(self, state: RoomState, timeline: str = "") -> AgentAction:
        out = await self.ask(state, extra="Session timeline:\n" + timeline[:3000])
        return AgentAction(agent=self.name, action="debrief",
                           message=out.get("summary", ""), payload=out,
                           source="agent")
