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
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.events import AttentionEvent, SessionEnd   # noqa: E402

log = logging.getLogger("aura.ingest_room_ar")

SCREEN_TARGETS = {"screen", "own_screen", "presenter", "front",
                  "shared_screen", "board"}
AWAY_SCORE = 0.15        # legacy flat away-score (strict mode only)


@dataclass
class GazeCal:
    """Calibration for turning head pose into a *graded* attention score.

    The room camera usually is NOT co-located with the screen, so faces that
    are genuinely looking at the screen still read off-axis. The classifier's
    narrow `own_screen` cone then misses them and everyone collapses to the
    flat AWAY_SCORE (the "stuck at 15%" symptom). Grading attention smoothly
    from how far the gaze is off a configurable centre fixes that: engagement
    now varies as people turn their heads, and `--yaw-center`/`--pitch-center`
    let you compensate for the camera angle.
    """
    yaw_center: float = 0.0      # head yaw (deg) when looking AT the screen
    pitch_center: float = 0.0    # head pitch (deg) when looking AT the screen
    yaw_cone: float = 35.0       # deg off-centre at which attention hits ~0
    pitch_cone: float = 28.0
    away_floor: float = 0.05     # floor for fully off-axis faces
    strict: bool = False         # True -> original binary screen/AWAY_SCORE


def _graded_attention(yaw: float, pitch: float, cal: GazeCal) -> float:
    """1.0 looking straight at the (calibrated) screen centre, decaying to 0
    at the cone edges — same falloff shape AURA's native tracker uses."""
    y = max(0.0, 1.0 - abs(yaw - cal.yaw_center) / max(cal.yaw_cone, 1e-6))
    p = max(0.0, 1.0 - abs(pitch - cal.pitch_center) / max(cal.pitch_cone, 1e-6))
    return y * p


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


def result_to_event(r, classify, cfg, ts: float, cal: GazeCal | None = None
                    ) -> AttentionEvent:
    """GazeResult -> AURA AttentionEvent.

    cal is None  -> original strict behaviour (own_screen -> conf, else 0.15).
                    Kept as the default so existing unit tests are unchanged.
    cal provided -> graded: screen-lookers keep the classifier confidence, but
                    everyone else decays smoothly with gaze angle instead of
                    snapping to a flat 0.15. This is what the ingest CLI uses,
                    so the room engagement actually moves.
    """
    yaw = getattr(r, "yaw", 0.0)
    pitch = getattr(r, "pitch", 0.0)
    target, conf = classify(yaw, pitch, getattr(r, "blink", 0.0), cfg)
    if cal is None or cal.strict:
        attention = float(conf) if _is_attending(target) else AWAY_SCORE
    else:
        graded = _graded_attention(yaw, pitch, cal)
        attention = (max(float(conf), graded) if _is_attending(target)
                     else max(graded, cal.away_floor))
    pid = (f"person_{r.track_id}" if getattr(r, "track_id", None) is not None
           else f"person_{getattr(r, 'face_index', 0)}")
    e = AttentionEvent(person_id=pid, attention=round(attention, 3),
                       yaw=round(yaw, 1),
                       pitch=round(pitch, 1),
                       source="attention_room")
    e.ts = ts
    return e


def ingest_room_with_estimator(video: Path, estimator, classify, cfg,
                               t0: float, sample_fps: float = 5.0,
                               cal: GazeCal | None = None
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
    target_hist: dict[str, int] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            media_t = idx / src_fps
            faces_here = 0
            for r in estimator.estimate(rgb, int(media_t * 1000)):
                e = result_to_event(r, classify, cfg, t0 + media_t, cal)
                events.append({"topic": e.topic, **e.model_dump()})
                # tally raw gaze target (cheap, pure-arithmetic) for diagnostics
                tgt, _ = classify(getattr(r, "yaw", 0.0),
                                  getattr(r, "pitch", 0.0),
                                  getattr(r, "blink", 0.0), cfg)
                target_hist[tgt] = target_hist.get(tgt, 0) + 1
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
    if events:
        scores = [e["attention"] for e in events]
        mean = sum(scores) / len(scores)
        attending = sum(1 for s in scores if s >= 0.4) / len(scores)
        hist = ", ".join(f"{k}={v}" for k, v in
                         sorted(target_hist.items(), key=lambda kv: -kv[1]))
        print(f"  gaze targets: {hist}")
        print(f"  attention: mean={mean:.2f}, "
              f"{attending*100:.0f}% of frames >=0.40")
        if mean <= AWAY_SCORE + 0.02:
            print("  ⚠ attention is flat/low — the camera angle likely shifts "
                  "head pose off the screen cone. Re-run with --yaw-center / "
                  "--pitch-center set to the pose people show when looking at "
                  "the screen (read the dominant 'left'/'right'/'up_away' bias "
                  "above), and/or widen --yaw-cone / --pitch-cone.")
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
    # --- gaze calibration (fixes the "engagement stuck at 15%" symptom) ------
    ap.add_argument("--yaw-center", type=float, default=0.0,
                    help="head yaw (deg) people show when looking AT the "
                         "screen; set to the dominant left/right bias if the "
                         "camera is off to one side")
    ap.add_argument("--pitch-center", type=float, default=0.0,
                    help="head pitch (deg) when looking at the screen (e.g. a "
                         "high camera makes attentive faces read pitched-down)")
    ap.add_argument("--yaw-cone", type=float, default=35.0,
                    help="deg off-centre at which attention decays to 0 "
                         "(widen for a wide-angle room shot)")
    ap.add_argument("--pitch-cone", type=float, default=28.0)
    ap.add_argument("--away-floor", type=float, default=0.05,
                    help="attention floor for fully off-axis faces")
    ap.add_argument("--strict-screen", action="store_true",
                    help="disable graded scoring; restore the original binary "
                         "own_screen->conf / else->0.15 behaviour")
    ap.add_argument("--merge-into",
                    help="existing events.jsonl to merge into")
    ap.add_argument("--out", default="room_events.jsonl")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cal = GazeCal(yaw_center=args.yaw_center, pitch_center=args.pitch_center,
                  yaw_cone=args.yaw_cone, pitch_cone=args.pitch_cone,
                  away_floor=args.away_floor, strict=args.strict_screen)

    classify, cfg = load_attention_room(Path(args.ar_path))
    estimator = build_estimator(Path(args.ar_path), args.estimator,
                                args.model, cfg, args.max_faces,
                                detector=args.detector,
                                det_width=args.det_width,
                                min_conf=args.min_conf,
                                detector_model=args.detector_model)
    events = ingest_room_with_estimator(Path(args.room), estimator,
                                        classify, cfg, args.t0, args.fps,
                                        cal=cal)
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
