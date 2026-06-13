"""
Sink consumes AttentionState updates and does logging or sending to an external system.
"""

import time

from ..core.observation import AttentionState

class ConsoleSink:
    def handle(self, state: AttentionState) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(state.t_wall))
        if not state.subjects:
            print(f"[{ts}] No subjects detected")
            return
        parts = []
        for sid, s in sorted(state.subjects.items()):
            parts.append(f"{sid} -> {s.target} ({s.confidence:.2f})")
        line = f"[{ts}] " + " ".join(parts)
        if state.joint_attention:
            shared = ", ".join(f"{t}: {len(m)}" for t, m in state.joint_attention.items())
            line += f" | Joint attention: {shared}"
        print(line)
