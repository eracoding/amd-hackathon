"""Action layer: where agent decisions become visible.

* Console HUD for the presenter (color-coded by priority)
* Markdown debrief report persisted at session end
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..bus import EventBus
from ..events import AgentAction

log = logging.getLogger("aura.actions")

_COLORS = {"high": "\033[91m", "medium": "\033[93m", "low": "\033[90m"}
_RESET = "\033[0m"


class ActionSink:
    def __init__(self, bus: EventBus, report_dir: str | Path = "reports") -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.actions: list[AgentAction] = []
        bus.subscribe(AgentAction, self._on_action)

    async def _on_action(self, action: AgentAction) -> None:
        self.actions.append(action)
        if action.action == "debrief":
            path = self._write_debrief(action)
            print(f"\n📄 Debrief written to {path}\n")
            return
        color = _COLORS.get(action.priority, "")
        stamp = time.strftime("%H:%M:%S", time.localtime(action.ts))
        print(f"{color}[{stamp}] ({action.agent}/{action.priority}) "
              f"{action.message}{_RESET}")

    def _write_debrief(self, action: AgentAction) -> Path:
        p = action.payload
        lines = [
            "# AURA Session Debrief",
            f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
            "## Summary",
            p.get("summary", action.message) or "(no summary)",
            "",
            "## Per-slide engagement",
        ]
        per_slide = p.get("per_slide") or []
        if per_slide:
            lines += ["| Slide | Engagement | Note |", "|---|---|---|"]
            lines += [f"| {s.get('slide','?')} | {s.get('engagement','?')} "
                      f"| {s.get('note','')} |" for s in per_slide]
        else:
            lines.append("_no per-slide data_")
        lines += ["", "## Unresolved questions"]
        lines += [f"- {q}" for q in p.get("unresolved_questions", [])] or ["_none_"]
        lines += ["", "## Action items"]
        lines += [f"- {a}" for a in p.get("action_items", [])] or ["_none_"]
        lines += ["", "## Agent activity log"]
        lines += [f"- `{a.agent}` → {a.action}: {a.message}"
                  for a in self.actions if a.action != "debrief"]

        path = self.report_dir / f"debrief_{int(time.time())}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        (self.report_dir / "last_debrief.json").write_text(
            json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
