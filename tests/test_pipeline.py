"""End-to-end pipeline test: simulated session through fusion + agents (mock LLM).

Asserts the two scripted anomalies are caught:
  1. engagement collapse on slide 4 -> Analyst insight + Coach nudge
  2. question burst on slide 6 + pause -> Moderator surfaces questions
"""
import asyncio
import os

import pytest

os.environ["AURA_LLM_MOCK"] = "1"

from aura.actions.sink import ActionSink            # noqa: E402
from aura.agents.llm import LLMClient               # noqa: E402
from aura.agents.orchestrator import Orchestrator   # noqa: E402
from aura.bus import EventBus                       # noqa: E402
from aura.fusion.state import RoomStateBuilder      # noqa: E402
from aura.perception.simulate import run_simulated_session  # noqa: E402


@pytest.mark.asyncio
async def test_full_pipeline_catches_scripted_anomalies(tmp_path):
    bus = EventBus()
    fusion = RoomStateBuilder(bus)
    sink = ActionSink(bus, report_dir=tmp_path)
    orch = Orchestrator(bus, fusion, LLMClient(), tick_s=0.2)

    orch_task = asyncio.create_task(orch.run())
    await run_simulated_session(bus, tick_s=0.05)   # fast replay
    await asyncio.wait_for(orch.finished.wait(), timeout=30)
    orch_task.cancel()
    await bus.drain()

    agents_fired = {a.agent for a in sink.actions}
    kinds = {a.action for a in sink.actions}

    assert "Moderator" in agents_fired, "question burst was not surfaced"
    assert "surface_questions" in kinds
    assert "Summarizer" in agents_fired and "debrief" in kinds
    # engagement collapse should produce at least one insight or nudge
    assert kinds & {"insight", "nudge"}, "slide-4 engagement dip went unnoticed"
    # debrief file persisted
    assert list(tmp_path.glob("debrief_*.md")), "debrief report not written"


@pytest.mark.asyncio
async def test_room_state_token_budget():
    """The LLM-facing state must stay compact (token-efficiency criterion)."""
    import json
    bus = EventBus()
    fusion = RoomStateBuilder(bus)
    await run_simulated_session(bus, tick_s=0.01)
    await bus.drain()
    state = fusion.snapshot()
    payload = json.dumps(state.to_llm_json())
    assert len(payload) < 2800, f"room state too large: {len(payload)} chars"
