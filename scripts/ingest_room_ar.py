"""Ingest the room video through YOUR `attention_room` gaze stack instead of
AURA's built-in MediaPipe tracker — built for exactly your failure mode:
wide IP-camera shots where FaceMesh finds nothing but your SCRFD/ONNX
detectors + gaze estimators do.

    python -m scripts.ingest_room_ar \
        --room data/cam.mp4 \
        --ar-path ~/attention_room \
        --estimator onnx --model ~/attention_room/models/gaze.onnx \
        --merge-into recordings/session1/events.jsonl \
        --t0 1000.0            # the room stream's t0 from manifest.json

What you provide: your package path, which estimator (mediapipe|onnx) and
its model file — the same arguments your `run_gaze_node.py` uses. The
adapter maps each GazeResult through your own `classify_gaze_target`
(target == "screen" -> confidence becomes the attention score) and emits
AURA AttentionEvents, then merges them (time-sorted) into an existing
events.jsonl so you can re-run the replay + PPT-B without re-ingesting
audio/screen.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.events import AttentionEvent, SessionEnd   # noqa: E402

AWAY_SCORE = 0.15        # attention when gaze target is not the screen


def load_attention_room(ar_path: Path):
    """Import the user's package + its gaze classifier and config."""
    sys.path.insert(0, str(ar_path))
    fusion = importlib.import_module("attention_room.fusion.attention")
    classify = fusion.classify_gaze_target
    # GazeConfig may live in fusion or config — be tolerant
    cfg_cls = getattr(fusion, "GazeConfig", None)
    if cfg_cls is None:
        cfg_mod = importlib.import_module("attention_room.config")
        cfg_cls = getattr(cfg_mod, "GazeConfig")
    return classify, cfg_cls()


def build_estimator(ar_path: Path, kind: str, model: str,
                    cfg, max_faces: int):
    est = importlib.import_module(
        "attention_room.modalities.gaze.estimators")
    if kind == "mediapipe":
        return est.MediaPipeGazeEstimator(model, cfg, max_faces=max_faces)
    onnx = importlib.import_module(
        "attention_room.modalities.gaze.onnx_estimator")
    return onnx.OnnxGazeEstimator(model, cfg)


def result_to_event(r, classify, cfg, ts: float) -> AttentionEvent:
    """GazeResult -> AURA AttentionEvent via the user's own target logic."""
    target, conf = classify(getattr(r, "yaw", 0.0), getattr(r, "pitch", 0.0),
                            getattr(r, "blink", 0.0), cfg)
    attention = float(conf) if target == "screen" else AWAY_SCORE
    pid = (f"person_{r.track_id}" if getattr(r, "track_id", None) is not None
           else f"person_{getattr(r, 'face_index', 0)}")
    e = AttentionEvent(person_id=pid, attention=round(attention, 3),
                       yaw=round(getattr(r, "yaw", 0.0), 1),
                       pitch=round(getattr(r, "pitch", 0.0), 1),
                       source="attention_room")
    e.ts = ts
    return e


def ingest_room_with_estimator(video: Path, estimator, classify, cfg,
                               t0: float, sample_fps: float = 5.0
                               ) -> list[dict]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / sample_fps)))
    events, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            media_t = idx / src_fps
            for r in estimator.estimate(rgb, int(media_t * 1000)):
                e = result_to_event(r, classify, cfg, t0 + media_t)
                events.append({"topic": e.topic, **e.model_dump()})
        idx += 1
    cap.release()
    people = {e["person_id"] for e in events}
    print(f"attention_room adapter: {len(events)} attention events, "
          f"{len(people)} tracklet(s): {sorted(people)[:8]}")
    return events


def merge_events(new: list[dict], target: Path) -> None:
    existing = [json.loads(ln) for ln in
                target.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # drop the old SessionEnd and any prior camera events (re-ingestion)
    existing = [e for e in existing
                if e["topic"] != "SessionEnd"
                and not (e["topic"] == "AttentionEvent"
                         and e.get("source") in ("camera", "attention_room"))]
    merged = sorted(existing + new, key=lambda e: e["ts"])
    end = SessionEnd(source="ingest")
    end.ts = merged[-1]["ts"] + 1.0
    merged.append({"topic": end.topic, **end.model_dump()})
    with target.open("w", encoding="utf-8") as fh:
        for e in merged:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"merged -> {target} ({len(merged)} events)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--ar-path", required=True,
                    help="folder containing the attention_room package")
    ap.add_argument("--estimator", choices=["mediapipe", "onnx"],
                    default="onnx")
    ap.add_argument("--model", required=True,
                    help="estimator model file (same as run_gaze_node.py)")
    ap.add_argument("--max-faces", type=int, default=6)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--t0", type=float, default=1000.0,
                    help="room stream t0 (see manifest.json)")
    ap.add_argument("--merge-into",
                    help="existing events.jsonl to merge into")
    ap.add_argument("--out", default="room_events.jsonl")
    args = ap.parse_args()

    classify, cfg = load_attention_room(Path(args.ar_path))
    estimator = build_estimator(Path(args.ar_path), args.estimator,
                                args.model, cfg, args.max_faces)
    events = ingest_room_with_estimator(Path(args.room), estimator,
                                        classify, cfg, args.t0, args.fps)
    if not events:
        sys.exit("no faces found — check the model path and run "
                 "scripts/diagnose.py --room to see face sizes")
    if args.merge_into:
        merge_events(events, Path(args.merge_into))
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
