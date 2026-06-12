"""Slide-awareness tests: deck indexing, distorted-capture matching, the
debounced tracker, and manual control. Skipped if the PDF stack is absent."""
import asyncio

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pypdfium2")
pytest.importorskip("fpdf")
from PIL import Image  # noqa: E402

from aura.bus import EventBus                            # noqa: E402
from aura.perception.slides import (                     # noqa: E402
    DeckIndex, ManualSlideControl, ScreenSlideTracker,
)

TITLES = ["Alpha intro", "Beta numbers", "Gamma architecture",
          "Delta privacy", "Epsilon roadmap"]


@pytest.fixture(scope="module")
def deck_pdf(tmp_path_factory):
    from fpdf import FPDF
    p = tmp_path_factory.mktemp("deck") / "deck.pdf"
    pdf = FPDF(orientation="L", unit="mm", format=(180, 320))
    for i, t in enumerate(TITLES):
        pdf.add_page()
        pdf.set_font("helvetica", "B", 28)
        pdf.set_y(30)
        pdf.cell(0, 16, f"{i+1}. {t}", align="C")
        pdf.set_font("helvetica", "", 14)
        pdf.set_y(70)
        pdf.multi_cell(0, 8, f"Body for slide {i+1}. " * 6)
    pdf.output(str(p))
    return p


def _distort(img: Image.Image, rng) -> Image.Image:
    """Simulate a screen capture: rescale, letterbox, dim, add noise."""
    f = np.array(img.convert("RGB").resize((1180, 660)))
    canvas = np.full((720, 1280, 3), 18, np.uint8)
    canvas[30:690, 50:1230] = np.clip(
        f.astype(int) - 12 + rng.integers(-6, 6, f.shape), 0, 255
    ).astype(np.uint8)
    return Image.fromarray(canvas)


def _renders(deck_pdf):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(deck_pdf))
    return [p.render(scale=1.0).to_pil().convert("RGB") for p in doc]


def test_deck_index_text_and_titles(deck_pdf):
    deck = DeckIndex(deck_pdf)
    assert len(deck) == 5
    assert "Gamma architecture" in deck.page(3)["title"]
    assert "Body for slide 4" in deck.page(4)["text"]


def test_matching_under_capture_distortion(deck_pdf):
    deck = DeckIndex(deck_pdf)
    rng = np.random.default_rng(1)
    for i, img in enumerate(_renders(deck_pdf), start=1):
        page, sim = deck.match(_distort(img, rng))
        assert page == i, f"page {i} mismatched to {page} (sim {sim:.3f})"
        assert sim > 0.9


def test_noise_and_demo_frames_rejected(deck_pdf):
    deck = DeckIndex(deck_pdf)
    noise = Image.fromarray(
        np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8))
    page, _ = deck.match(noise)
    assert page is None, "a non-slide frame must not match any page"


@pytest.mark.asyncio
async def test_tracker_debounce_and_event_content(deck_pdf):
    deck = DeckIndex(deck_pdf)
    bus = EventBus()
    got = []
    async def collect(e): got.append(e)
    bus.subscribe("SlideChange", collect)
    tracker = ScreenSlideTracker(bus, deck, capture_fn=lambda: None)
    rng = np.random.default_rng(2)
    frames = []
    for img in _renders(deck_pdf):
        frames += [_distort(img, rng)] * 3      # 3 captures per slide
    for fr in frames:
        await tracker.observe(fr)
    await bus.drain()
    assert [e.slide for e in got] == [1, 2, 3, 4, 5]
    assert "Delta privacy" in got[3].title
    assert "Body for slide 4" in got[3].content


@pytest.mark.asyncio
async def test_manual_control_bounds(deck_pdf):
    deck = DeckIndex(deck_pdf)
    bus = EventBus()
    got = []
    async def collect(e): got.append(e)
    bus.subscribe("SlideChange", collect)
    ctl = ManualSlideControl(bus, deck)
    await ctl.next(); await ctl.next(); await ctl.prev()
    for _ in range(9):
        await ctl.next()
    await bus.drain()
    assert got[0].slide == 1 and got[-1].slide == 5  # clamped to deck length
    assert got[0].content.startswith("1. Alpha intro")


@pytest.mark.asyncio
async def test_screen_observer_annotation_detection(deck_pdf, tmp_path):
    """Ink drawn on the shared slide -> ScreenAnnotationEvent with merged
    bbox; clean frames -> nothing; repeats -> no re-emission."""
    from PIL import ImageDraw
    from aura.perception.slides import ScreenObserver
    deck = DeckIndex(deck_pdf)
    bus = EventBus()
    got = []
    async def collect(e): got.append(e)
    bus.subscribe("ScreenAnnotationEvent", collect)
    obs = ScreenObserver(bus, deck, capture_fn=lambda: None,
                         patch_dir=tmp_path)
    rng = np.random.default_rng(3)
    page = deck.render_page(3)

    for _ in range(3):                       # clean phase
        await obs.observe(_distort(page, rng))
    await bus.drain()
    assert got == [], "clean slide must produce no annotations"

    ann = page.convert("RGB").copy()
    d = ImageDraw.Draw(ann)
    d.ellipse([300, 180, 420, 280], outline=(220, 40, 40), width=6)
    for _ in range(3):                       # annotated phase
        await obs.observe(_distort(ann, rng))
    await bus.drain()
    assert len(got) == 1, f"expected 1 annotation, got {len(got)}"
    e = got[0]
    assert e.slide == 3 and e.kind == "drawn"
    assert 0.2 < e.bbox[0] < 0.45 and 0.2 < e.bbox[1] < 0.55
    import pathlib
    assert pathlib.Path(e.patch_path).exists()

    for _ in range(3):                       # repeat phase
        await obs.observe(_distort(ann, rng))
    await bus.drain()
    assert len(got) == 1, "annotation must not re-emit"
