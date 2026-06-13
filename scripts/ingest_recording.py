"""Ingest a recorded session (from scripts/record_session.py) into one
time-ordered AURA event stream — then replay it through the live agents.

Runs entirely on the cloud GPU box (this is where heavy perception belongs):

    # in the AMD notebook, after uploading the recordings/dryrun1 folder
    python -m scripts.ingest_recording recordings/dryrun1 \
        --deck slides.pdf [--whisper medium] [--attn-fps 5]

    # then the demo:
    python -m aura.main --replay recordings/dryrun1/events.jsonl --speed 2

Stages (each optional — missing inputs are skipped with a warning):
  room.mp4    -> AttentionTracker over sampled frames -> AttentionEvents
  audio.wav   -> faster-whisper (GPU or CPU)          -> TranscriptSegments
                 + PauseDetected from inter-segment gaps >= 3 s
  screen.mp4  -> DeckIndex pHash matching             -> SlideChanges (+content)
  interactions.jsonl -> passed through unchanged (already true timestamps)

All events get absolute timestamps t0_stream + media_time, so cross-modal
ordering is exact regardless of how fast ingestion itself runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.bus import EventBus                       # noqa: E402
from aura.events import (                           # noqa: E402
    InteractionEvent, InteractionKind, PauseDetected, SessionEnd,
    TranscriptSegment,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ingest")

from aura.perception.speech import is_meta_question  # noqa: E402

PAUSE_GAP_S = 3.0


async def ingest_video(path: Path, t0: float, fps: float) -> list[dict]:
    import cv2
    from aura.perception.attention import AttentionTracker
    bus = EventBus()  # throwaway — we collect returned events directly
    tracker = AttentionTracker(bus)
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    events, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            media_t = idx / src_fps
            for e in await tracker.process_frame(frame):
                e.ts = t0 + media_t
                events.append({"topic": e.topic, **e.model_dump()})
        idx += 1
    cap.release()
    log.info("room video: %d frames sampled -> %d attention events",
             idx // step, len(events))
    return events


def ingest_audio(path: Path, t0: float, model_size: str,
                 device: str, language: str) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("faster-whisper not installed — skipping audio "
                    "(pip install faster-whisper)")
        return []
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    segments = list(model.transcribe(str(path), language=language,
                                     vad_filter=True)[0])
    if not segments:
        # VAD swallowing everything = quiet recording (common with phone
        # mics at distance). Retry without VAD before giving up.
        log.warning("VAD removed all audio — likely a quiet recording. "
                    "Retrying without VAD. (Consider normalizing: "
                    "ffmpeg -i in.m4a -af loudnorm -ac 1 -ar 16000 out.wav)")
        segments = list(model.transcribe(str(path), language=language,
                                         vad_filter=False)[0])
    events, prev_end = [], None
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if prev_end is not None and seg.start - prev_end >= PAUSE_GAP_S:
            p = PauseDetected(silence_s=seg.start - prev_end, source="ingest")
            p.ts = t0 + prev_end + (seg.start - prev_end)
            events.append({"topic": p.topic, **p.model_dump()})
        dur = max(0.5, seg.end - seg.start)
        e = TranscriptSegment(
            text=text, t_start=t0 + seg.start, t_end=t0 + seg.end,
            words_per_min=len(text.split()) / dur * 60.0, source="ingest")
        e.ts = t0 + seg.end
        events.append({"topic": e.topic, **e.model_dump()})
        if text.rstrip().endswith("?") and not is_meta_question(text):
            q = InteractionEvent(person_id="voice",
                                 kind=InteractionKind.question,
                                 text=text, source="voice")
            q.ts = t0 + seg.end
            events.append({"topic": q.topic, **q.model_dump()})
        prev_end = seg.end
    log.info("audio: %d transcript segments", len(events))
    return events


async def ingest_screen(path: Path, t0: float, deck_path: Path,
                        sample_s: float = 1.0) -> list[dict]:
    import cv2
    from PIL import Image
    from aura.perception.slides import DeckIndex, ScreenObserver
    deck = DeckIndex(deck_path)
    bus = EventBus()
    tracker = ScreenObserver(bus, deck, capture_fn=lambda: None,
                             patch_dir=path.parent / "annotations")
    collected: list[dict] = []
    async def _collect_ann(e):
        collected.append({"topic": e.topic, **e.model_dump()})
    bus.subscribe("ScreenAnnotationEvent", _collect_ann)
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps * sample_s)))
    events, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            e = await tracker.observe(img, ts=t0 + idx / src_fps)
            if e:
                events.append({"topic": e.topic, **e.model_dump()})
        idx += 1
    cap.release()
    await bus.drain()
    events += collected
    log.info("screen video: %d slide changes + %d annotations detected",
             len(events) - len(collected), len(collected))
    return events


def ingest_interactions(path: Path) -> list[dict]:
    events = [json.loads(ln) for ln in
              path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    log.info("interactions: %d events passed through", len(events))
    return events


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording_dir")
    ap.add_argument("--deck", help="presentation PDF (enables slide tracking)")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--whisper-device", default="auto",
                    help="auto|cuda|cpu (ROCm presents as cuda)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--attn-fps", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rec = Path(args.recording_dir)
    manifest = json.loads((rec / "manifest.json").read_text())
    streams = manifest["streams"]
    events: list[dict] = []

    if "room" in streams and (rec / streams["room"]["path"]).exists():
        events += await ingest_video(rec / streams["room"]["path"],
                                     streams["room"]["t0"], args.attn_fps)
    if "audio" in streams and (rec / streams["audio"]["path"]).exists():
        device = args.whisper_device
        if device == "auto":
            # faster-whisper runs on CTranslate2, which has NO ROCm backend.
            # On AMD GPUs torch reports cuda=True (ROCm masquerades), but
            # passing "cuda" to CTranslate2 would crash -> force CPU there.
            try:
                import torch
                is_rocm = getattr(torch.version, "hip", None) is not None
                device = "cuda" if (torch.cuda.is_available()
                                    and not is_rocm) else "cpu"
            except ImportError:
                device = "cpu"
        events += ingest_audio(rec / streams["audio"]["path"],
                               streams["audio"]["t0"], args.whisper,
                               device, args.language)
    if args.deck and "screen" in streams \
            and (rec / streams["screen"]["path"]).exists():
        events += await ingest_screen(rec / streams["screen"]["path"],
                                      streams["screen"]["t0"], Path(args.deck))
    ipath = rec / streams.get("interactions", {}).get("path",
                                                      "interactions.jsonl")
    if ipath.exists():
        events += ingest_interactions(ipath)

    if not events:
        sys.exit("nothing ingested — check the manifest and input files")

    events.sort(key=lambda e: e["ts"])
    end = SessionEnd(source="ingest")
    end.ts = events[-1]["ts"] + 1.0
    events.append({"topic": end.topic, **end.model_dump()})

    out = Path(args.out or rec / "events.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    dur = events[-1]["ts"] - events[0]["ts"]
    log.info("wrote %d events spanning %.1f s -> %s", len(events), dur, out)
    log.info("demo: python -m aura.main --replay %s --speed 2", out)


if __name__ == "__main__":
    asyncio.run(main())
