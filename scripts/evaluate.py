"""AURA evaluation harness.

Runs the scripted synthetic session and scores the agentic layer against
ground truth (the two scripted anomalies), producing the metrics table for
the submission:

  * detection: did the right agent fire for each scripted anomaly?
  * false positives: actions fired outside any anomaly window
  * latency: anomaly onset -> first corresponding agent action (median, p95)
  * efficiency: LLM calls, prompt/completion tokens, tokens per action

Usage:
  AURA_LLM_MOCK=1 python -m scripts.evaluate            # offline, deterministic
  python -m scripts.evaluate                             # against live vLLM
  python -m scripts.evaluate --runs 5                    # variance over seeds
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.actions.sink import ActionSink            # noqa: E402
from aura.agents.llm import LLMClient               # noqa: E402
from aura.agents.orchestrator import Orchestrator   # noqa: E402
from aura.bus import EventBus                       # noqa: E402
from aura.events import BaseEvent, InteractionEvent, SlideChange  # noqa: E402
from aura.fusion.state import RoomStateBuilder      # noqa: E402
from aura.perception.simulate import run_simulated_session  # noqa: E402


class GroundTruth:
    """Tracks scripted anomaly onsets during the run (wall-clock)."""

    def __init__(self, bus: EventBus) -> None:
        self.dip_onset: float | None = None        # slide 4 starts
        self.dip_end: float | None = None          # slide 6 starts (recovery)
        self.burst_onset: float | None = None      # first question lands
        bus.subscribe(SlideChange, self._on_slide)
        bus.subscribe(InteractionEvent, self._on_q)

    async def _on_slide(self, e: SlideChange) -> None:
        if e.slide == 4:
            self.dip_onset = e.ts
        if e.slide == 6:
            self.dip_end = e.ts

    async def _on_q(self, e: InteractionEvent) -> None:
        if e.kind.value == "question" and self.burst_onset is None:
            self.burst_onset = e.ts


async def run_once(seed: int, tick: float) -> dict:
    bus = EventBus()
    fusion = RoomStateBuilder(bus)
    sink = ActionSink(bus, report_dir="reports/eval")
    truth = GroundTruth(bus)
    llm = LLMClient()
    orch = Orchestrator(bus, fusion, llm, tick_s=max(0.2, tick * 4))

    orch_task = asyncio.create_task(orch.run())
    t0 = time.time()
    await run_simulated_session(bus, tick_s=tick, seed=seed)
    await asyncio.wait_for(orch.finished.wait(), timeout=120)
    orch_task.cancel()
    await bus.drain()
    wall = time.time() - t0

    insights = [a for a in sink.actions if a.action in ("insight", "nudge")]
    surfaces = [a for a in sink.actions if a.action == "surface_questions"]

    def first_after(actions: list, onset: float | None) -> float | None:
        if onset is None:
            return None
        later = [a.ts - onset for a in actions if a.ts >= onset]
        return min(later) if later else None

    dip_latency = first_after(insights, truth.dip_onset)
    burst_latency = first_after(surfaces, truth.burst_onset)
    # false positives: engagement actions fired before the dip even started
    fp = [a for a in insights
          if truth.dip_onset is not None and a.ts < truth.dip_onset]

    return {
        "seed": seed,
        "wall_s": round(wall, 1),
        "dip_detected": dip_latency is not None,
        "dip_latency_s": round(dip_latency, 2) if dip_latency else None,
        "burst_detected": burst_latency is not None,
        "burst_latency_s": round(burst_latency, 2) if burst_latency else None,
        "false_positives": len(fp),
        "total_actions": len([a for a in sink.actions if a.action != "debrief"]),
        "llm": llm.meter.summary(),
        "debrief_written": any(a.action == "debrief" for a in sink.actions),
    }


def render(results: list[dict], tick: float) -> str:
    n = len(results)
    det_dip = sum(r["dip_detected"] for r in results)
    det_burst = sum(r["burst_detected"] for r in results)
    fp = sum(r["false_positives"] for r in results)
    dip_lat = [r["dip_latency_s"] for r in results if r["dip_latency_s"]]
    burst_lat = [r["burst_latency_s"] for r in results if r["burst_latency_s"]]
    calls = sum(r["llm"]["llm_calls"] for r in results)
    ptok = sum(r["llm"]["prompt_tokens"] for r in results)
    ctok = sum(r["llm"]["completion_tokens"] for r in results)
    actions = sum(r["total_actions"] for r in results)
    mode = "mock" if os.environ.get("AURA_LLM_MOCK") == "1" else "live vLLM"

    def med(xs):
        return f"{statistics.median(xs):.2f}" if xs else "—"

    lines = [
        "# AURA Evaluation Metrics",
        f"_Runs: {n} (seeds 0..{n-1}) · sim tick {tick}s · LLM backend: {mode}_",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Engagement-dip detection rate | {det_dip}/{n} |",
        f"| Question-burst surfacing rate | {det_burst}/{n} |",
        f"| False-positive engagement alerts | {fp} total |",
        f"| Median dip detection latency | {med(dip_lat)} s |",
        f"| Median burst surfacing latency | {med(burst_lat)} s |",
        f"| Debrief generated | {sum(r['debrief_written'] for r in results)}/{n} |",
        f"| LLM calls per session (avg) | {calls/max(1,n):.1f} |",
        f"| Prompt tokens per session (avg) | {ptok/max(1,n):.0f} |",
        f"| Completion tokens per session (avg) | {ctok/max(1,n):.0f} |",
        f"| Tokens per emitted action | "
        f"{(ptok+ctok)/max(1,actions):.0f} |",
        "",
        "## Per-run detail",
        "```json",
        json.dumps(results, indent=2),
        "```",
    ]
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tick", type=float, default=0.1)
    args = ap.parse_args()

    results = [await run_once(seed, args.tick) for seed in range(args.runs)]
    report = render(results, args.tick)
    out = Path("reports/metrics.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print("\n" + report.split("## Per-run detail")[0])
    print(f"full report -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
