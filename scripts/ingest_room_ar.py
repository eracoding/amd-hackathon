"""Ingest the room video through YOUR `attention_room` gaze stack instead of
AURA's built-in MediaPipe tracker — built for exactly your failure mode:
wide IP-camera shots where FaceMesh finds nothing but your SCRFD/ONNX
detectors + gaze estimators do.

    python -m scripts.ingest_room_ar \
        --room data/cam.mp4 \
        --ar-path ~/attention_room \
        --estimator onnx --model ~/attention_room/models/gaze.onnx \
        --detector retinaface --det-width 1280 --min-conf 0.3 \
        --merge-into recordings/session1/events.jsonl \
        --t0 1000.0            # the room stream's t0 from manifest.json

FAR-DISTANCE FACES (the whole reason this exists): --detector retinaface
(default) uses uniface's RetinaFace, which finds small/distant faces a
webcam-oriented landmarker misses. If people still aren't detected, raise
--det-width (1280 -> 1600) and lower --min-conf (0.3 -> 0.2). Requires
`pip install uniface` and your gaze ONNX model.

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

SCREEN_TARGETS = {"screen", "own_screen", "presenter", "front",
                  "shared_screen", "board"}
AWAY_SCORE = 0.15        # attention when gaze target is not the screen


def _is_attending(target: str) -> bool:
    return str(target).lower() in SCREEN_TARGETS


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
                    cfg, max_faces: int, detector: str = "retinaface",
                    det_width: int = 960, min_conf: float = 0.5,
                    detector_model: str | None = None):
    est = importlib.import_module(
        "attention_room.modalities.gaze.estimators")
    if kind == "mediapipe":
        return est.MediaPipeGazeEstimator(model, cfg, max_faces=max_faces)

    onnx = importlib.import_module(
        "attention_room.modalities.gaze.onnx_estimator")

    # FACE DETECTION choice — critical for far-distance room shots.
    # RetinaFace (uniface) detects small/distant faces that MediaPipe's
    # landmarker (built for near webcam faces) misses entirely.
    face_box_provider = None
    if detector == "retinaface":
        detectors = importlib.import_module(
            "attention_room.modalities.gaze.detectors")
        face_box_provider = detectors.UnifaceFaceBoxProvider(
            max_faces=max_faces, det_width=det_width,
            min_confidence=min_conf)
        log.info(f"face detector: RetinaFace (uniface), "
                 f"det_width={det_width}, min_conf={min_conf}")
    else:
        log.info("face detector: MediaPipe landmarker (near-face only)")

    return onnx.OnnxGazeEstimator(
        model, cfg, max_faces=max_faces,
        accelerator=_cpu_accelerator(ar_path),
        face_box_provider=face_box_provider,
        detector_model=(detector_model if face_box_provider is None
                        else None))


def _cpu_accelerator(ar_path: Path):
    """Force the CPU ONNX provider. onnxruntime-gpu on a ROCm box can't load
    CUDA libs (libcublasLt.so.12 etc.), so letting it try just spams errors
    and slows startup. CPU is correct for offline ingestion anyway."""
    try:
        dev = importlib.import_module("attention_room.core.device")
        for kw in ("preference", "device", "pref"):
            try:
                return dev.Accelerator(**{kw: "cpu"})
            except TypeError:
                continue
        acc = dev.Accelerator()
        for attr in ("preference", "pref", "device"):
            if hasattr(acc, attr):
                setattr(acc, attr, "cpu")
        return acc
    except Exception:
        return None


def result_to_event(r, classify, cfg, ts: float) -> AttentionEvent:
    """GazeResult -> AURA AttentionEvent via the user's own target logic."""
    target, conf = classify(getattr(r, "yaw", 0.0), getattr(r, "pitch", 0.0),
                            getattr(r, "blink", 0.0), cfg)
    attention = float(conf) if _is_attending(target) else AWAY_SCORE
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
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_sampled = (total // step) if total else 0
    print(f"processing ~{n_sampled} sampled frames "
          f"(every {step}th of {total}) on CPU — this can take a few "
          f"minutes; progress every 25 frames...", flush=True)
    import time as _t
    events, idx, done, t_start = [], 0, 0, _t.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            media_t = idx / src_fps
            faces_here = 0
            for r in estimator.estimate(rgb, int(media_t * 1000)):
                e = result_to_event(r, classify, cfg, t0 + media_t)
                events.append({"topic": e.topic, **e.model_dump()})
                faces_here += 1
            done += 1
            if done % 25 == 0:
                rate = done / max(1e-6, _t.time() - t_start)
                eta = (n_sampled - done) / max(rate, 1e-6)
                print(f"  {done}/{n_sampled} frames "
                      f"({rate:.1f} fps, ETA {eta:.0f}s) — "
                      f"{len({e['person_id'] for e in events})} tracklet(s) "
                      f"so far", flush=True)
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
                    help="gaze ONNX model file (same as run_gaze_node.py)")
    ap.add_argument("--detector", choices=["retinaface", "mediapipe"],
                    default="retinaface",
                    help="face detector. retinaface (uniface) for "
                         "far/distant faces; mediapipe for near webcam faces")
    ap.add_argument("--detector-model",
                    help="face_landmarker.task — only needed for "
                         "--detector mediapipe")
    ap.add_argument("--det-width", type=int, default=960,
                    help="RetinaFace detection width; raise to 1280/1600 for "
                         "smaller/more distant faces (slower)")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="face detection confidence floor; lower (0.3) to "
                         "catch faint distant faces")
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
                                args.model, cfg, args.max_faces,
                                detector=args.detector,
                                det_width=args.det_width,
                                min_conf=args.min_conf,
                                detector_model=args.detector_model)
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
