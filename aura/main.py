"""AURA entrypoint.

  python -m aura.main --sim                  # full pipeline on synthetic session
  python -m aura.main --live                 # webcam + mic + device gateway
  python -m aura.main --replay sessions/session.jsonl   # re-run a recording
  AURA_LLM_MOCK=1 python -m aura.main --sim  # no GPU needed (mock agents)

All modes serve the presenter HUD at http://localhost:8766 (disable: --no-hud).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .actions.sink import ActionSink
from .agents.llm import LLMClient
from .agents.orchestrator import Orchestrator
from .bus import EventBus, SessionRecorder, replay
from .fusion.state import RoomStateBuilder


async def run(args: argparse.Namespace) -> None:
    bus = EventBus()
    recorder = None
    if not args.replay:  # don't re-record a replay over itself
        recorder = SessionRecorder(bus, "sessions/session.jsonl")
    fusion = RoomStateBuilder(bus)
    sink = ActionSink(bus)  # noqa: F841 — registers itself on the bus
    tick = max(0.2, args.tick * 4) if (args.sim or args.replay) else 5.0
    orch = Orchestrator(bus, fusion, LLMClient(), tick_s=tick)

    tasks = [asyncio.create_task(orch.run())]

    if not args.no_monitor:
        from .actions.monitor import RoomMonitor
        monitor = RoomMonitor(bus, fusion, port=args.monitor_port,
                              push_period_s=min(1.0, tick))
        tasks.append(asyncio.create_task(monitor.run()))
        print(f"Room monitor (view-only): http://localhost:{args.monitor_port}")

    if args.theater:
        from .actions.theater import TheaterMonitor
        rec_dir = (Path(args.replay).parent if args.replay else None)
        theater = TheaterMonitor(bus, fusion, recording_dir=rec_dir,
                                 port=args.theater_port,
                                 push_period_s=min(0.5, tick), speed=args.speed,
                                 room_video=args.room_video,
                                 screen_video=args.screen_video)
        tasks.append(asyncio.create_task(theater.run()))
        print(f"Theater (live demo view): http://localhost:{args.theater_port}")

    if args.sim:
        from .perception.simulate import run_simulated_session
        tasks.append(asyncio.create_task(
            run_simulated_session(bus, tick_s=args.tick)))
    elif args.replay:
        tasks.append(asyncio.create_task(
            replay(bus, args.replay, speed=args.speed)))
    else:
        from .perception.attention import AttentionTracker
        from .perception.speech import SpeechPipeline
        tracker = AttentionTracker(bus, camera=args.camera)
        speech = SpeechPipeline(bus, model_size=args.whisper)
        tasks += [asyncio.create_task(tracker.run()),
                  asyncio.create_task(speech.run())]
        if args.deck:
            from .perception.slides import DeckIndex, ScreenObserver
            observer = ScreenObserver(bus, DeckIndex(args.deck))
            tasks.append(asyncio.create_task(observer.run()))
            print("Screen observer: watching the shared presentation "
                  "(slides + annotations)")
        if args.gateway:  # legacy/companion input path, off by default
            from .perception.interaction import InteractionGateway
            gateway = InteractionGateway(bus, port=args.port)
            tasks.append(asyncio.create_task(gateway.run()))
            print(f"(optional) participant gateway: http://<host>:{args.port}")
        print("Live mode: participants interact through the presentation "
              "itself (annotations, chat, voice). Ctrl-C to end session.")

    try:
        await orch.finished.wait()
        await asyncio.sleep(min(2.0, tick))  # let the HUD show the debrief
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await bus.drain()
        if recorder:
            recorder.close()
        print("\nToken/latency metrics:", orch.llm.meter.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA — agentic room intelligence")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sim", action="store_true", help="synthetic session")
    mode.add_argument("--live", action="store_true", help="webcam + mic + gateway")
    mode.add_argument("--replay", metavar="JSONL", help="replay a recorded session")
    parser.add_argument("--camera", default=0,
                        help="cv2 camera index, video file, or RTSP url")
    parser.add_argument("--whisper", default="small")
    parser.add_argument("--deck", help="presentation PDF for the screen "
                        "observer (slides + annotation detection)")
    parser.add_argument("--gateway", action="store_true",
                        help="also serve the legacy participant web client")
    parser.add_argument("--port", type=int, default=8765,
                        help="participant gateway port (with --gateway)")
    parser.add_argument("--monitor-port", type=int, default=8766)
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--theater", action="store_true",
                        help="live demo view: raw room+screen video + "
                             "transcript streaming in sync with agent reasoning")
    parser.add_argument("--theater-port", type=int, default=8767)
    parser.add_argument("--room-video",
                        help="explicit path to the room camera video for "
                             "theater (overrides auto-detection)")
    parser.add_argument("--screen-video",
                        help="explicit path to the screen capture video for "
                             "theater")
    parser.add_argument("--tick", type=float, default=0.5,
                        help="sim seconds per tick (lower = faster replay)")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="replay speed multiplier (0 = max speed)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nsession aborted")


if __name__ == "__main__":
    main()
