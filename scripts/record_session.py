"""Record a full AURA session: room camera + mic + presenter screen + live
participant interactions — one command, one manifest, synchronized clocks.

    python -m scripts.record_session --out recordings/dryrun1 \
        [--cam 0 | --cam rtsp://user:pass@192.168.1.50/stream1] \
        [--mic default] [--no-screen] [--port 8765]

While recording, participants open http://<your-laptop-ip>:8765 on their
phones to ask questions / react — those are captured digitally with exact
timestamps (no post-processing needed).

Stop with Ctrl-C. Output directory contains:
    room.mp4            audience camera
    audio.wav           presenter microphone (16 kHz mono — whisper-ready)
    screen.mp4          presenter display (drives slide tracking)
    interactions.jsonl  participant events (already in AURA event format)
    manifest.json       per-stream start timestamps (the sync source of truth)

Sync model: all recorders share this machine's clock; the manifest stores
each stream's spawn time. AURA's fusion window is 10 s, so the ±0.3 s ffmpeg
startup jitter is immaterial. For an audit marker, clap once while flashing
slide 1 right after recording starts.

Requires ffmpeg on PATH. Linux (x11grab/pulse/v4l2) and macOS (avfoundation)
supported; on macOS run `ffmpeg -f avfoundation -list_devices true -i ""`
to find device indices and pass --cam/--mic/--screen-dev accordingly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("record")


def ffmpeg_cmds(args, out: Path) -> dict[str, list[str]]:
    plat = sys.platform
    cmds: dict[str, list[str]] = {}

    # --- room camera ---
    cam = str(args.cam)
    if cam.startswith(("rtsp://", "http://", "https://")):
        cmds["room"] = ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", cam,
                        "-c", "copy", str(out / "room.mp4")]
    elif plat == "darwin":
        cmds["room"] = ["ffmpeg", "-y", "-f", "avfoundation", "-framerate", "15",
                        "-i", f"{cam}:none", "-pix_fmt", "yuv420p",
                        str(out / "room.mp4")]
    else:
        dev = cam if cam.startswith("/dev/") else f"/dev/video{cam}"
        cmds["room"] = ["ffmpeg", "-y", "-f", "v4l2", "-framerate", "15",
                        "-i", dev, "-pix_fmt", "yuv420p", str(out / "room.mp4")]

    # --- microphone (16 kHz mono wav: exactly what faster-whisper wants) ---
    if plat == "darwin":
        cmds["audio"] = ["ffmpeg", "-y", "-f", "avfoundation",
                         "-i", f"none:{args.mic}",
                         "-ac", "1", "-ar", "16000", str(out / "audio.wav")]
    else:
        cmds["audio"] = ["ffmpeg", "-y", "-f", "pulse", "-i", str(args.mic),
                         "-ac", "1", "-ar", "16000", str(out / "audio.wav")]

    # --- presenter screen ---
    if not args.no_screen:
        if plat == "darwin":
            cmds["screen"] = ["ffmpeg", "-y", "-f", "avfoundation",
                              "-framerate", "5", "-i", f"{args.screen_dev}:none",
                              "-pix_fmt", "yuv420p", str(out / "screen.mp4")]
        else:
            cmds["screen"] = ["ffmpeg", "-y", "-f", "x11grab", "-framerate", "5",
                              "-i", args.display, "-pix_fmt", "yuv420p",
                              str(out / "screen.mp4")]
    return cmds


async def run_gateway(out: Path, port: int) -> None:
    """Capture participant interactions live (CPU-only, no GPU needed)."""
    from aura.bus import EventBus, SessionRecorder
    from aura.perception.interaction import InteractionGateway
    bus = EventBus()
    SessionRecorder(bus, out / "interactions.jsonl")
    gw = InteractionGateway(bus, port=port)
    await gw.run()
    while True:  # serve until cancelled
        await asyncio.sleep(3600)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="0",
                    help="camera index, /dev/videoN, or rtsp:// URL")
    ap.add_argument("--mic", default="default",
                    help="pulse source (linux) / avfoundation index (mac)")
    ap.add_argument("--display", default=":0.0", help="x11 display (linux)")
    ap.add_argument("--screen-dev", default="1",
                    help="avfoundation screen index (mac)")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--no-cam", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--port", type=int, default=8765,
                    help="participant gateway port")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH — install it first "
                 "(apt install ffmpeg / brew install ffmpeg)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmds = ffmpeg_cmds(args, out)
    if args.no_cam:
        cmds.pop("room", None)
    if args.no_audio:
        cmds.pop("audio", None)

    manifest = {"created": time.time(), "streams": {}}
    procs: dict[str, subprocess.Popen] = {}
    for name, cmd in cmds.items():
        t0 = time.time()
        procs[name] = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=open(out / f"{name}.log", "w"))
        manifest["streams"][name] = {
            "path": cmd[-1].split("/")[-1], "t0": t0}
        log.info("recording %-6s -> %s", name, cmd[-1])

    manifest["streams"]["interactions"] = {
        "path": "interactions.jsonl", "t0": time.time()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    gw_task = asyncio.create_task(run_gateway(out, args.port))
    log.info("participant gateway: http://<this-machine-ip>:%d", args.port)
    log.info("RECORDING — clap once + show slide 1 now (sync marker). "
             "Ctrl-C to stop.")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    log.info("stopping recorders (finalizing files)...")
    gw_task.cancel()
    for name, p in procs.items():
        try:  # 'q' lets ffmpeg finalize the container cleanly
            p.stdin.write(b"q")
            p.stdin.flush()
        except (BrokenPipeError, OSError):
            p.send_signal(signal.SIGINT)
    for name, p in procs.items():
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        log.info("%-6s finalized (exit %s)", name, p.returncode)
    log.info("done -> %s  (next: scripts/ingest_recording.py)", out)


if __name__ == "__main__":
    asyncio.run(main())
