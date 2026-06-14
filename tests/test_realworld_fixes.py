"""Regressions for the real-footage failure fixes: live-reference diffing,
VLM title fallback, meta-question filter, attention_room adapter mapping,
proxy-safe WS URLs."""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pypdfium2")
pytest.importorskip("fpdf")
from PIL import Image, ImageDraw                          # noqa: E402

from aura.bus import EventBus                             # noqa: E402
from aura.perception.slides import DeckIndex, ScreenObserver  # noqa: E402
from aura.perception.speech import is_meta_question       # noqa: E402


@pytest.fixture(scope="module")
def deck_pdf(tmp_path_factory):
    from fpdf import FPDF
    p = tmp_path_factory.mktemp("deck") / "deck.pdf"
    pdf = FPDF(orientation="L", unit="mm", format=(180, 320))
    for i, t in enumerate(["Alpha intro", "Beta numbers", "Gamma design"]):
        pdf.add_page()
        pdf.set_font("helvetica", "B", 28)
        pdf.set_y(30)
        pdf.cell(0, 16, f"{i+1}. {t}", align="C")
    pdf.output(str(p))
    return p


def _rerendered(img, rng):
    """Simulate PowerPoint Live: same content, DIFFERENT renderer — shifted,
    rescaled, brightness-shifted. PDF-reference diffing breaks here."""
    f = np.array(img.convert("RGB").resize((1100, 600)))
    canvas = np.full((720, 1280, 3), 18, np.uint8)
    canvas[60:660, 90:1190] = np.clip(
        f.astype(int) - 25 + rng.integers(-5, 5, f.shape), 0, 255
    ).astype(np.uint8)
    return Image.fromarray(canvas)


@pytest.mark.asyncio
async def test_live_reference_survives_rerendering(deck_pdf, tmp_path):
    """With a re-rendered deck (renderer mismatch), live-ref mode must stay
    silent on clean frames and still catch real ink."""
    deck = DeckIndex(deck_pdf)
    bus = EventBus()
    got = []
    async def collect(e): got.append(e)
    bus.subscribe("ScreenAnnotationEvent", collect)
    obs = ScreenObserver(bus, deck, capture_fn=lambda: None,
                         patch_dir=tmp_path, ref_mode="live")
    rng = np.random.default_rng(7)
    page = deck.render_page(2)
    for _ in range(4):                       # clean re-rendered frames
        await obs.observe(_rerendered(page, rng))
    await bus.drain()
    assert got == [], "renderer mismatch must not produce phantom ink"
    ann = page.convert("RGB").copy()
    ImageDraw.Draw(ann).line([(320, 380), (520, 360), (600, 430)],
                             fill=(220, 40, 40), width=7)
    for _ in range(3):
        await obs.observe(_rerendered(ann, rng))
    await bus.drain()
    assert len(got) == 1 and got[0].slide == 2


@pytest.mark.asyncio
async def test_vlm_title_fallback(deck_pdf, tmp_path, monkeypatch):
    """NCC fails entirely (unknown rendering) -> VLM reads the title ->
    fuzzy match identifies the slide."""
    monkeypatch.setenv("AURA_LLM_MOCK", "1")
    from aura.perception.screen_vlm import ScreenVLM
    deck = DeckIndex(deck_pdf)
    bus = EventBus()
    slides = []
    async def collect(e): slides.append(e)
    bus.subscribe("SlideChange", collect)
    vlm = ScreenVLM()
    async def fake_title(_img):
        return {"title": "2. Beta numbers"}
    monkeypatch.setattr(vlm, "read_slide_title", fake_title)
    obs = ScreenObserver(bus, deck, capture_fn=lambda: None,
                         patch_dir=tmp_path, vlm=vlm)
    noise = Image.fromarray(
        np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8))
    for _ in range(4):                        # NCC similarity ~0 -> fallback
        await obs.observe(noise)
    await bus.drain()
    assert [e.slide for e in slides] == [2]
    assert slides[0].source == "screen-vlm"
    assert "Beta numbers" in slides[0].title


def test_meta_question_filter():
    for t in ("Do we have a question on the charts?",
              "Any questions so far?", "What's your question?",
              "You have a question?", "Avez-vous des questions?"):
        assert is_meta_question(t), t
    for t in ("Where is the data stored?",
              "Can this run fully on a Jetson?",
              "Why does the slope use minutes?"):
        assert not is_meta_question(t), t


def test_attention_room_adapter_mapping():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.ingest_room_ar import result_to_event

    @dataclass
    class R:
        yaw: float; pitch: float; blink: float
        track_id: int = None; face_index: int = 0

    def classify(yaw, pitch, blink, _cfg):
        return ("screen", 0.9) if abs(yaw) < 20 else ("right", 0.8)

    e = result_to_event(R(5.0, 2.0, 0.1, track_id=3), classify, None, 1234.5)
    assert e.person_id == "person_3" and e.attention == 0.9
    assert e.ts == 1234.5 and e.source == "attention_room"
    e2 = result_to_event(R(45.0, 0.0, 0.1), classify, None, 1.0)
    assert e2.attention == 0.15 and e2.person_id == "person_0"


def test_ws_url_preserves_proxy_path():
    from aura.actions.monitor import MONITOR_HTML
    from aura.perception.interaction import CLIENT_HTML
    for html in (MONITOR_HTML, CLIENT_HTML):
        assert "location.pathname" in html, \
            "WS URL must include the jupyter-proxy path prefix"
        assert "+'://'+location.host+'/ws'" not in html.replace(" ", "")


def test_retinaface_provider_injection(monkeypatch):
    """build_estimator with detector='retinaface' must construct the uniface
    provider and inject it, so MediaPipe is never used on far-distance shots."""
    import sys
    import types
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent))

    ar = Path("/tmp/ar/attention_room")
    if not ar.exists():
        pytest.skip("attention_room package not present")
    sys.path.insert(0, str(ar))

    # stub uniface.detection.RetinaFace
    uni = types.ModuleType("uniface")
    det = types.ModuleType("uniface.detection")
    class _Det:
        def __init__(self, *a, **k): pass
        def detect(self, bgr):
            h, w = bgr.shape[:2]
            class _F:
                bbox = [w * 0.46, h * 0.40, w * 0.54, h * 0.52]
                confidence = 0.9
                landmarks = [[w * 0.48, h * 0.45], [w * 0.52, h * 0.45]]
            return [_F()]
    det.RetinaFace = _Det
    uni.detection = det
    monkeypatch.setitem(sys.modules, "uniface", uni)
    monkeypatch.setitem(sys.modules, "uniface.detection", det)

    import importlib
    detectors = importlib.import_module(
        "attention_room.modalities.gaze.detectors")
    prov = detectors.UnifaceFaceBoxProvider(max_faces=6, det_width=1280,
                                            min_confidence=0.3)
    boxes = prov(np.zeros((1080, 1920, 3), np.uint8), 0)
    assert len(boxes) == 1
    assert boxes[0].track_id is not None, "tracker must assign a stable id"
    assert (boxes[0].x2 - boxes[0].x1) > 0
