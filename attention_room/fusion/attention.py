"""
Fusion Engine: Combines data from multiple modalities to create a unified representation of attention.
"""

import logging
import queue
import threading
import time
from typing import Callable

from ..config import GazeConfig
from ..core.bus import Bus
from ..core.observation import AttentionState, SubjectAttention, Observation

log = logging.getLogger(__name__)

Handler = Callable[["FusionEngine", Observation], None]
Listener = Callable[[AttentionState], None]

def classify_gaze_target(yaw: float, pitch: float, blink: float, cfg: GazeConfig) -> tuple[str, float]:
    if blink >= cfg.blink_thresh:
        return "eyes_closed", min(1.0, blink)
    if abs(yaw) <= cfg.screen_yaw_deg and cfg.screen_pitch_lo <= pitch <= cfg.screen_pitch_hi:
        conf = 1.0 - min(1.0, abs(yaw) / max(cfg.screen_yaw_deg, 1e-6))
        return "own_screen", round(0.5 + 0.5 * conf, 2)
    if yaw > cfg.screen_yaw_deg:
        return "right", round(min(1.0, yaw / (2 * cfg.away_yaw_deg)), 2)
    if yaw < -cfg.screen_yaw_deg:
        return "left", round(min(1.0, -yaw / (2 * cfg.away_yaw_deg)), 2)
    if pitch > cfg.up_pitch_deg:
        return "up_away", round(min(1.0, pitch / (2 * cfg.up_pitch_deg)), 2)
    return "elsewhere", 0.5

def gaze_handler(engine: "FusionEngine", obs: Observation) -> None:
    p = obs.payload
    target, conf = classify_gaze_target(
        p.get("yaw", 0.0),
        p.get("pitch", 0.0),
        p.get("blink", 0.0),
        engine.cfg
    )
    sid = obs.subject_id or obs.source_id
    with engine._lock:
        engine._latest[sid] = SubjectAttention(
            subject_id=sid,
            target=target,
            confidence=conf,
            source_id=obs.source_id,
            t_wall=obs.t_wall,
            gaze={"yaw": p.get("yaw", 0.0), "pitch": p.get("pitch", 0.0)},
        )

class FusionEngine:
    def __init__(self, bus: Bus, cfg: GazeConfig, republish: bool = True) -> None:
        self.bus = bus
        self.cfg = cfg
        self.republish = republish
        self.handlers: dict[str, Handler] = {
            "gaze": gaze_handler,
        }
        self._listeners: list[Listener] = []
        self._latest: dict[str, SubjectAttention] = {}
        self._lock = threading.Lock()
        self._inbox: queue.Queue[Observation] = queue.Queue()
        self._stop = threading.Event()
        self._workers: list[threading.Thread] = []

    def register(self, modality: str, handler: Handler) -> None:
        self.handlers[modality] = handler
    
    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)
    
    def start(self) -> None:
        self.bus.subscribe(self._inbox.put, modalities=list(self.handlers))
        self._workers = [
            threading.Thread(target=self._consume, name="fusion-consume", daemon=True),
            threading.Thread(target=self._emit_loop, name="fusion-emit", daemon=True),
        ]
        for w in self._workers:
            w.start()

    def stop(self) -> None:
        self._stop.set()

    def _consume(self) -> None:
        while not self._stop.is_set():
            try:
                obs = self._inbox.get(timeout=0.2)
            except queue.Empty:
                continue
            handler = self.handlers.get(obs.modality)
            if handler is not None:
                try:
                    handler(self, obs)
                except Exception as e:
                    log.exception(f"Error handling observation {obs}: {e}")

    def _emit_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.cfg.emit_interval_s)
            self._emit_once()

    def _emit_once(self) -> None:
        now = time.time()
        with self._lock:
            fresh = {sid: s for sid, s in self._latest.items()
                     if now - s.t_wall <= self.cfg.stale_timeout_s}
            self._latest = fresh
            subjects = dict(fresh)
        
        joint: dict[str, list[str]] = {}
        groups: dict[str, list[str]] = {}
        for sid, s in subjects.items():
            groups.setdefault(s.target, []).append(sid)
        for target, members in groups.items():
            if len(members) >= 2 and target in self.cfg.shared_targets:
                joint[target] = sorted(members)
        
        state = AttentionState(t_wall=now, subjects=subjects, joint_attention=joint)
        for listener in self._listeners:
            try:
                listener(state)
            except Exception as e:
                log.exception(f"Error in listener {listener}: {e}")
        if self.republish:
            self.bus.publish(Observation(
                modality="attention_state",
                source_id="fusion",
                payload={
                    "subjects": {k: v.__dict__ for k, v in state.subjects.items()},
                    "joint_attention": state.joint_attention,
                }
            ))
