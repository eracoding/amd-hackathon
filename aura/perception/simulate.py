"""Deterministic synthetic presentation session.

Scripts a 10-minute (compressed) talk with 4 participants and two anomalies
the agents must catch:
  * engagement collapse on slide 4 (dense architecture slide)
  * question burst + presenter pause on slide 6

Time is compressed: 1 simulated minute == `tick_s` real seconds, so a full
session replays in ~30-60 s for demos and tests.
"""
from __future__ import annotations

import asyncio
import math
import random

from ..bus import EventBus
from ..events import (
    AttentionEvent, InteractionEvent, InteractionKind, PauseDetected,
    ScreenAnnotationEvent, SessionEnd, SlideChange, TranscriptSegment,
)

SLIDES = [
    (1, "Why rooms should understand intent"),
    (2, "The cost of disengaged meetings"),
    (3, "AURA system overview"),
    (4, "Fusion architecture deep-dive"),       # scripted engagement dip
    (5, "Agentic reasoning layer"),
    (6, "Privacy by design"),                   # scripted question burst
    (7, "Evaluation results"),
    (8, "Roadmap and ask"),
]

SCRIPT = {
    1: "Good morning everyone. Today I want to convince you that meeting rooms can be intelligent participants rather than passive furniture.",
    2: "Industry surveys put the cost of unproductive meetings in the billions. The core failure is feedback latency. Presenters discover disengagement weeks later, if ever.",
    3: "AURA fuses three streams. Camera based attention, presenter speech, and device interactions, into a single room state that software agents can reason over.",
    4: "Now the fusion internals. We maintain dual ring buffers per modality with ten second tactical and sixty second strategic windows, attendance weighted engagement indices, and slope estimators over exponentially smoothed per tracklet attention trajectories aligned against slide transition boundaries.",
    5: "Four agents consume that state. An analyst, a moderator, a coach, and a summarizer. Each with its own trigger policy and action authority.",
    6: "Crucially, no identities. People are anonymous tracklets and raw video never leaves the perception layer.",
    7: "On our labeled clip the attention tracker reaches the high eighties on binary attending classification, and median event to action latency stays under four seconds.",
    8: "Next steps are IP camera ingestion and slide content understanding. Thank you. Questions welcome.",
}

QUESTION_BURST = [
    ("person_1", "Where is the video stored, exactly?"),
    ("person_3", "Can this comply with GDPR for EU offices?"),
    ("person_2", "What happens if two people swap seats?"),
]


def _attention_profile(slide: int, minute_frac: float, person: int, rng: random.Random) -> float:
    base = 0.85 - 0.05 * person * 0.3
    if slide == 4:                      # scripted collapse, worsening through the slide
        base -= 0.35 + 0.25 * minute_frac
    if slide >= 7:                      # mild end-of-talk fatigue
        base -= 0.10
    return max(0.0, min(1.0, base + rng.gauss(0, 0.05)))


async def run_simulated_session(
    bus: EventBus,
    n_people: int = 4,
    tick_s: float = 0.5,
    minutes_per_slide: float = 1.0,
    seed: int = 7,
) -> None:
    rng = random.Random(seed)
    t = 0.0
    for slide, title in SLIDES:
        await bus.publish(SlideChange(slide=slide, title=title, source="sim"))
        text = SCRIPT[slide]
        words = text.split()
        # presenter speaks the slide text across the slide duration
        seg_words = max(8, len(words) // 3)
        chunks = [" ".join(words[i:i + seg_words]) for i in range(0, len(words), seg_words)]
        ticks = max(len(chunks), int(minutes_per_slide * 6))  # 6 ticks per sim-minute

        for k in range(ticks):
            frac = k / ticks
            # attention events for every participant
            for p in range(n_people):
                yaw = rng.gauss(0, 8) + (25 if _attention_profile(slide, frac, p, rng) < 0.4 else 0)
                await bus.publish(AttentionEvent(
                    person_id=f"person_{p}",
                    attention=_attention_profile(slide, frac, p, rng),
                    yaw=yaw, pitch=rng.gauss(0, 5),
                    source="sim",
                ))
            # speech chunk
            if k < len(chunks):
                wpm = 150 + (60 if slide == 4 else 0)  # presenter rushes the dense slide
                await bus.publish(TranscriptSegment(
                    text=chunks[k], t_start=t, t_end=t + tick_s,
                    words_per_min=wpm, source="sim",
                ))
            # scripted on-slide annotations during the slide-4 overload
            if slide == 4 and k == 3:
                for bbox in ([0.30, 0.25, 0.45, 0.40], [0.55, 0.60, 0.78, 0.70]):
                    await bus.publish(ScreenAnnotationEvent(
                        slide=4, bbox=bbox, area_frac=0.004,
                        kind="drawn", source="sim"))
            # scripted question burst on slide 6
            if slide == 6 and k == 2:
                for pid, q in QUESTION_BURST:
                    await bus.publish(InteractionEvent(
                        person_id=pid, kind=InteractionKind.question,
                        text=q, slide=slide, source="sim",
                    ))
            t += tick_s
            await asyncio.sleep(tick_s)

        if slide == 6:  # presenter pauses after the privacy slide
            await bus.publish(PauseDetected(silence_s=4.0, source="sim"))
            await asyncio.sleep(tick_s * 2)

    await bus.publish(SessionEnd(source="sim"))
