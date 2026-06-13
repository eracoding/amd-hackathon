"""Diagnose raw recordings BEFORE full ingestion — answers, in one command,
the questions a failed run raises:

    python -m scripts.diagnose --audio data/mic.m4a --screen data/screen.mp4 \
        --deck data/slides.pdf [--slide-region x0,y0,x1,y1] [--vlm]

AUDIO  — decodes a slice, reports loudness (RMS/peak dBFS), and verdicts:
         "too quiet for VAD" vs OK, with the exact normalize command.
SCREEN — extracts frames at 10/30/50/70% of the video into debug_frames/
         (open them in Jupyter to measure regions), and for each frame
         reports the best deck-page match similarity with: the full frame,
         your --slide-region (if given), and the VLM auto-region (if --vlm).
         The verdict tells you which region to use.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, BAD = "✅", "⚠️ ", "❌"


def diagnose_audio(path: Path, seconds: float = 90.0) -> None:
    print(f"\n=== AUDIO: {path} ===")
    try:
        import av
        import numpy as np
    except ImportError:
        print(f"{WARN} install faster-whisper (brings PyAV) to diagnose audio")
        return
    container = av.open(str(path))
    stream = container.streams.audio[0]
    print(f"codec={stream.codec_context.name} rate={stream.rate} "
          f"channels={stream.channels} "
          f"duration={float(stream.duration * stream.time_base) if stream.duration else '?'}s")
    chunks, total = [], 0.0
    for frame in container.decode(stream):
        a = frame.to_ndarray()
        chunks.append(a.astype(np.float32).ravel())
        total += frame.samples / frame.sample_rate
        if total >= seconds:
            break
    container.close()
    x = np.concatenate(chunks)
    if x.dtype.kind == "i" or np.abs(x).max() > 1.5:   # integer PCM scale
        x = x / 32768.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.abs(x).max())
    rms_db = 20 * math.log10(max(rms, 1e-9))
    peak_db = 20 * math.log10(max(peak, 1e-9))
    print(f"first {total:.0f}s: RMS {rms_db:.1f} dBFS, peak {peak_db:.1f} dBFS")
    if rms_db < -45:
        print(f"{BAD} very quiet — Silero VAD will discard it. Normalize:")
        print(f"   ffmpeg -i {path} -af loudnorm=I=-16 -ac 1 -ar 16000 "
              f"{path.stem}_norm.wav")
        print("   then ingest the _norm.wav (the pipeline also auto-retries "
              "without VAD now, but normalized audio transcribes better).")
    elif rms_db < -35:
        print(f"{WARN} quiet-ish — usable, but normalizing (command above) "
              "will improve transcription quality.")
    else:
        print(f"{OK} loudness fine for VAD + whisper")


async def diagnose_screen(path: Path, deck_path: Path, slide_region,
                          use_vlm: bool) -> None:
    import cv2
    from PIL import Image
    from aura.perception.slides import DeckIndex
    from scripts.ingest_raw import crop_frac, parse_region

    print(f"\n=== SCREEN: {path} ===")
    deck = DeckIndex(deck_path)
    print(f"deck: {len(deck)} pages")
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"video: {total} frames @ {fps:.0f} fps "
          f"({total / max(fps, 1):.0f}s), "
          f"{int(cap.get(3))}x{int(cap.get(4))}")
    out = Path("debug_frames")
    out.mkdir(exist_ok=True)
    frames = []
    for frac in (0.10, 0.30, 0.50, 0.70):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, f = cap.read()
        if ok:
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            fp = out / f"frame_{int(frac * 100)}pct.png"
            img.save(fp)
            frames.append((frac, img))
    cap.release()
    print(f"saved {len(frames)} frames -> {out}/ (open them to measure "
          "regions as fractions of width/height)")

    regions = {"full frame": None}
    if slide_region:
        regions["--slide-region"] = parse_region(slide_region)
    if use_vlm and frames:
        from aura.perception.screen_vlm import ScreenVLM
        det = await ScreenVLM().detect_regions(frames[len(frames) // 2][1])
        if det.get("slide_region"):
            regions["VLM auto"] = tuple(det["slide_region"])
            print(f"VLM auto regions: {det}")

    best_overall = ("", 0.0)
    for name, reg in regions.items():
        sims = []
        for frac, img in frames:
            page, sim = deck.match(crop_frac(img, reg))
            sims.append((frac, page, sim))
        med = sorted(s for _, _, s in sims)[len(sims) // 2]
        detail = " ".join(f"[{int(f*100)}%→p{p or '-'} {s:.2f}]"
                          for f, p, s in sims)
        mark = OK if med >= 0.75 else (WARN if med >= 0.55 else BAD)
        print(f"{mark} {name:<16} median sim {med:.2f}  {detail}")
        if med > best_overall[1]:
            best_overall = (name, med)

    name, med = best_overall
    if med >= 0.75:
        print(f"\nVERDICT: use {name!r} — matching will work "
              f"(accept floor 0.75).")
    elif med >= 0.55:
        print(f"\nVERDICT: {name!r} is close ({med:.2f}); the slide crop "
              "still includes chrome/borders. Tighten the region using the "
              "saved frames, or check the deck PDF matches what was "
              "presented (same slides, same aspect).")
    else:
        print("\nVERDICT: nothing is close. Either the region is wrong "
              "(measure it on the saved frames), or the PDF is not the deck "
              "that was on screen (re-export it), or the slide pane is tiny "
              "in the capture (record the presentation window, not the "
              "whole desktop).")


async def diagnose_room(path: Path) -> None:
    import cv2
    from PIL import Image
    from aura.bus import EventBus
    from aura.perception.attention import AttentionTracker
    print(f"\n=== ROOM CAMERA: {path} ===")
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    H = int(cap.get(4))
    print(f"video: {total} frames, {int(cap.get(3))}x{H}")
    tracker = AttentionTracker(EventBus())
    out = Path("debug_frames"); out.mkdir(exist_ok=True)
    heights, face_counts = [], []
    for frac in (0.15, 0.40, 0.65, 0.90):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, frame = cap.read()
        if not ok:
            continue
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
            out / f"room_{int(frac*100)}pct.png")
        tracker._tracklets.clear()
        for _ in range(2):                       # let detection settle
            await tracker.process_frame(frame)
        boxes = [trk.box for trk in tracker._tracklets]
        face_counts.append(len(boxes))
        hs = [int((b[3] - b[1]) * H) for b in boxes]
        heights += hs
        print(f"  {int(frac*100)}%: {len(boxes)} face(s), "
              f"heights px: {hs or '—'}")
    if not heights:
        print(f"{BAD} MediaPipe FaceMesh found NO faces. Wide IP-cam shots "
              "defeat it (needs faces ≳80 px tall). Use your detector-based "
              "stack instead:")
        print("   python -m scripts.ingest_room_ar --room", path,
              "--ar-path <attention_room> --estimator onnx --model <model> "
              "--merge-into <events.jsonl> --t0 <room t0>")
    elif sorted(heights)[len(heights) // 2] < 80:
        print(f"{WARN} faces found but small (median "
              f"{sorted(heights)[len(heights)//2]} px) — tracking will be "
              "flaky. Prefer ingest_room_ar (ONNX detector), or crop/zoom "
              "the camera region.")
    else:
        print(f"{OK} faces large enough — AURA's built-in tracker is fine.")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--room")
    ap.add_argument("--screen")
    ap.add_argument("--deck")
    ap.add_argument("--slide-region")
    ap.add_argument("--vlm", action="store_true")
    args = ap.parse_args()
    if args.audio:
        diagnose_audio(Path(args.audio))
    if args.screen:
        if not args.deck:
            sys.exit("--screen needs --deck")
        await diagnose_screen(Path(args.screen), Path(args.deck),
                              args.slide_region, args.vlm)
    if args.room:
        await diagnose_room(Path(args.room))
    if not (args.audio or args.screen or args.room):
        sys.exit("provide --audio, --screen and/or --room")


if __name__ == "__main__":
    asyncio.run(main())
