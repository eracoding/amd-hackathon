"""Ingest RAW recordings — no manifest, no session.jsonl, just the files you
have: a phone audio recording, an IP-camera video, and a screen capture of
the Teams window (slide + chat pane).

    python -m scripts.ingest_raw \
        --room cam.mp4 --audio mic.m4a --screen screen.mp4 \
        --deck slides.pdf \
        --slide-region 0.0,0.05,0.78,0.95 \
        --chat-region  0.78,0.10,1.0,0.95 \
        --sync "audio=12.5,room=8.2,screen=3.0" \
        --out recordings/session1

What it does:
  audio  -> faster-whisper transcript + pace + pauses; segments ending in
            '?' become voice questions
  room   -> per-person gaze attention (MediaPipe)
  screen -> SLIDE REGION: deck matching + drawn-annotation detection
            CHAT REGION: OCR (tesseract) of the Teams chat pane; new lines
            become typed questions/comments
  output -> {out}/events.jsonl — replay it through the agents:
            python -m aura.main --replay {out}/events.jsonl --speed 2
            then: python -m scripts.generate_pptb {out}/events.jsonl --pptx deck.pptx

SYNC: recordings started at different moments. Pick ONE moment present in
all of them (a clap, the first slide appearing, you saying "let's begin"),
note the time INSIDE each file where it happens (mm:ss or seconds), and pass
--sync "audio=<t>,room=<t>,screen=<t>". Don't over-engineer it: the fusion
window is 10 s, so ±1 s of eyeballing is fine. Omit --sync to assume all
files started simultaneously.

REGIONS are fractions of the frame: "x0,y0,x1,y1". With --vlm you can OMIT
them entirely: one VLM call locates the slide area and chat panel
automatically (explicit arguments always override). Without --vlm, omit
--slide-region to use the whole frame, and find regions by opening one
screenshot and estimating the boundaries.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.bus import EventBus                       # noqa: E402
from aura.events import (                           # noqa: E402
    InteractionEvent, InteractionKind, SessionEnd,
)
from scripts.ingest_recording import (              # noqa: E402
    ingest_audio, ingest_video,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ingest_raw")

BASE_T = 1000.0          # arbitrary epoch for relative timestamps
CHAT_SAMPLE_S = 2.0      # OCR the chat pane every N seconds
UI_NOISE = re.compile(
    r"^(type a (new )?message|reply|meeting chat|everyone|chat|raise|react"
    r"|view|leave|share|mute|camera|people|more|\d{1,2}:\d{2}( [AP]M)?)$",
    re.IGNORECASE)


def parse_region(spec: str | None) -> tuple[float, float, float, float] | None:
    if not spec:
        return None
    x0, y0, x1, y1 = (float(v) for v in spec.split(","))
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "region must be 0-1 fractions"
    return (x0, y0, x1, y1)


def parse_sync(spec: str | None) -> dict[str, float]:
    """'audio=12.5,room=1:23,screen=3' -> per-stream start offsets so the
    sync moment lands at the same global time in every stream."""
    if not spec:
        return {}
    marks: dict[str, float] = {}
    for part in spec.split(","):
        k, v = part.split("=")
        if ":" in v:
            mm, ss = v.split(":")
            marks[k.strip()] = int(mm) * 60 + float(ss)
        else:
            marks[k.strip()] = float(v)
    latest = max(marks.values())
    return {k: latest - m for k, m in marks.items()}   # start offsets >= 0


def crop_frac(img, region):
    if region is None:
        return img
    W, H = img.size
    return img.crop((int(region[0] * W), int(region[1] * H),
                     int(region[2] * W), int(region[3] * H)))


# ---------------------------------------------------------------- screen
async def ingest_screen_regions(path: Path, t0: float, deck_path: Path,
                                slide_region, chat_region,
                                out_dir: Path, sample_s: float = 1.0,
                                use_vlm: bool = False) -> list[dict]:
    import cv2
    from PIL import Image
    from aura.perception.slides import DeckIndex, ScreenObserver

    deck = DeckIndex(deck_path)
    bus = EventBus()
    vlm = None
    if use_vlm:
        from aura.perception.screen_vlm import ScreenVLM
        vlm = ScreenVLM()
    obs = ScreenObserver(bus, deck, capture_fn=lambda: None,
                         patch_dir=out_dir / "annotations", vlm=vlm)
    if vlm is not None and slide_region is None:
        # one VLM call replaces manual region measurement
        slide_region, chat_region = await _auto_regions(
            path, vlm, chat_region)
    collected: list[dict] = []

    async def collect(e):
        collected.append({"topic": e.topic, **e.model_dump()})
    bus.subscribe("ScreenAnnotationEvent", collect)
    bus.subscribe("SlideChange", collect)
    bus.subscribe("ScreenStateEvent", collect)

    chat = ChatPaneReader(vlm=vlm) if chat_region else None
    sims: list[float] = []
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps * sample_s)))
    chat_step = max(1, int(round(src_fps * CHAT_SAMPLE_S)))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = t0 + idx / src_fps
        if idx % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            await obs.observe(crop_frac(img, slide_region), ts=ts)
            sims.append(getattr(obs, "last_similarity", 0.0))
            obs_slide = obs.current or 0
            if chat and idx % chat_step == 0:
                pane = crop_frac(img, chat_region)
                msgs = (await chat.new_messages_vlm(pane)
                        if chat.vlm is not None
                        else chat.new_messages(pane))
                for person, line in msgs:
                    kind = (InteractionKind.question if "?" in line
                            else InteractionKind.comment)
                    e = InteractionEvent(person_id=person, kind=kind,
                                         text=line, slide=obs_slide,
                                         source="chat")
                    e.ts = ts
                    collected.append({"topic": e.topic, **e.model_dump()})
                    log.info("chat %s: %s", kind.value, line[:70])
        idx += 1
    cap.release()
    await bus.drain()
    if sims and not any(e["topic"] == "SlideChange" for e in collected):
        import statistics as _st
        log.warning(
            "no slide matched. Best-page similarity over %d samples: "
            "median %.2f, max %.2f (accept floor %.2f). %s",
            len(sims), _st.median(sims), max(sims), 0.75,
            "Similarities are LOW → the matcher is seeing the wrong crop: "
            "pass --slide-region (or --vlm for auto-detection), and run "
            "scripts/diagnose.py to measure it."
            if max(sims) < 0.6 else
            "Similarities are CLOSE → region is roughly right but distorted; "
            "tighten --slide-region to exclude UI chrome.")
    n_slides = sum(1 for e in collected if e["topic"] == "SlideChange")
    n_ann = sum(1 for e in collected if e["topic"] == "ScreenAnnotationEvent")
    n_chat = sum(1 for e in collected if e["topic"] == "InteractionEvent")
    log.info("screen: %d slide changes, %d annotations, %d chat messages",
             n_slides, n_ann, n_chat)
    return collected


async def _auto_regions(video_path: Path, vlm, chat_region):
    """Grab one representative frame, ask the VLM where the slide and chat
    panes are. Coarse is fine — the per-frame content-crop absorbs slop."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 4)     # 25% in: deck is up
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, chat_region
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    regions = await vlm.detect_regions(img)
    slide = regions.get("slide_region")
    chat = chat_region if chat_region else (
        tuple(regions["chat_region"]) if regions.get("chat_region") else None)
    return (tuple(slide) if slide else None), chat


class ChatPaneReader:
    """Reads the Teams chat pane: VLM-backed when available (robust to real
    fonts/avatars), tesseract OCR otherwise. Yields unseen messages only."""

    def __init__(self, vlm=None) -> None:
        self.vlm = vlm
        self._ok = vlm is not None
        if not self._ok:
            try:
                import pytesseract  # noqa: F401
                self._ok = True
            except ImportError:
                log.warning("no VLM and no tesseract — chat pane ignored "
                            "(pip install pytesseract / use --vlm)")
        self._seen: set[str] = set()
        self._prev_hash: int | None = None

    def new_messages(self, img) -> list[tuple[str, str]]:
        if not self._ok:
            return []
        if self.vlm is not None:
            return []        # async path handles VLM (new_messages_vlm)
        import pytesseract
        from aura.perception.slides import dhash
        h = dhash(img)
        if self._prev_hash is not None and h == self._prev_hash:
            return []                          # pane unchanged, skip OCR
        self._prev_hash = h
        # upscale for better OCR on small UI text
        big = img.convert("L").resize((img.width * 2, img.height * 2))
        text = pytesseract.image_to_string(big)
        out = []
        for raw in text.splitlines():
            line = " ".join(raw.split()).strip()
            if len(line) < 4 or UI_NOISE.match(line):
                continue
            key = re.sub(r"[^a-z0-9?]", "", line.lower())
            if len(key) < 4 or key in self._seen:
                continue
            self._seen.add(key)
            out.append(line)
        return self._group(out)

    async def new_messages_vlm(self, img) -> list[tuple[str, str]]:
        from aura.perception.slides import dhash
        h = dhash(img)
        if self._prev_hash is not None and h == self._prev_hash:
            return []
        self._prev_hash = h
        out = []
        for person, text in await self.vlm.read_chat(img):
            key = re.sub(r"[^a-z0-9?]", "", text.lower())
            if len(key) < 4 or key in self._seen:
                continue
            self._seen.add(key)
            out.append((person, text))
        return out

    @staticmethod
    def _looks_like_name(line: str) -> bool:
        words = line.split()
        return (1 <= len(words) <= 3 and len(line) <= 24
                and "?" not in line and not line[-1] in ".!,"
                and line[0].isupper())

    def _group(self, lines: list[str]) -> list[tuple[str, str]]:
        """Lines appearing in one OCR sample usually form one message,
        optionally preceded by the sender's name. Wrapped lines rejoin."""
        msgs: list[tuple[str, str]] = []
        person, buf = "chat", []
        for ln in lines:
            ln = re.sub(r"^[|!1]\s", "I ", ln)     # common OCR misreads of 'I'
            if self._looks_like_name(ln) and not buf:
                person = ln
                continue
            buf.append(ln)
        if buf:
            msgs.append((person, " ".join(buf)))
        return msgs


# ------------------------------------------------------------------ main
async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", help="audience camera video (IP cam recording)")
    ap.add_argument("--audio", help="microphone recording (wav/m4a/mp3/...)")
    ap.add_argument("--screen", help="screen capture of the Teams window")
    ap.add_argument("--deck", help="presentation PDF (needed with --screen)")
    ap.add_argument("--slide-region", help="x0,y0,x1,y1 fractions of the "
                    "frame containing the slide")
    ap.add_argument("--chat-region", help="x0,y0,x1,y1 fractions containing "
                    "the Teams chat pane")
    ap.add_argument("--sync", help='in-file times of one common moment, '
                    'e.g. "audio=12.5,room=1:23,screen=3"')
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--whisper-device", default="auto")
    ap.add_argument("--language", default="en")
    ap.add_argument("--vlm", action="store_true",
                    help="explain off-deck screens & read annotations with "
                         "Qwen2.5-VL (AURA_VLM_URL, default :8001)")
    ap.add_argument("--attn-fps", type=float, default=5.0)
    ap.add_argument("--out", default="recordings/raw_session")
    args = ap.parse_args()

    if not any([args.room, args.audio, args.screen]):
        sys.exit("provide at least one of --room / --audio / --screen")
    if args.screen and not args.deck:
        sys.exit("--screen needs --deck (the presentation PDF)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    offsets = parse_sync(args.sync)
    events: list[dict] = []
    manifest = {"streams": {}, "note": "generated by ingest_raw"}

    if args.room:
        t0 = BASE_T + offsets.get("room", 0.0)
        manifest["streams"]["room"] = {"path": args.room, "t0": t0}
        events += await ingest_video(Path(args.room), t0, args.attn_fps)
    if args.audio:
        t0 = BASE_T + offsets.get("audio", 0.0)
        manifest["streams"]["audio"] = {"path": args.audio, "t0": t0}
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
        events += ingest_audio(Path(args.audio), t0, args.whisper,
                               device, args.language)
    if args.screen:
        t0 = BASE_T + offsets.get("screen", 0.0)
        manifest["streams"]["screen"] = {"path": args.screen, "t0": t0}
        events += await ingest_screen_regions(
            Path(args.screen), t0, Path(args.deck),
            parse_region(args.slide_region), parse_region(args.chat_region),
            out, use_vlm=args.vlm)

    if not events:
        sys.exit("nothing ingested — check the input files")
    events.sort(key=lambda e: e["ts"])
    end = SessionEnd(source="ingest")
    end.ts = events[-1]["ts"] + 1.0
    events.append({"topic": end.topic, **end.model_dump()})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ev_path = out / "events.jsonl"
    with ev_path.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    dur = events[-1]["ts"] - events[0]["ts"]
    log.info("wrote %d events spanning %.1f s -> %s", len(events), dur, ev_path)
    log.info("next:  python -m aura.main --replay %s --speed 2", ev_path)
    log.info("then:  python -m scripts.generate_pptb %s --pptx <deck.pptx>",
             ev_path)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
