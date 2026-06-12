"""Tier-2 screen understanding tests: gating, budgets, the multimodal HTTP
request shape against a stub VL server, and mock-mode pipeline enrichment."""
import asyncio
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pypdfium2")
from PIL import Image, ImageDraw                     # noqa: E402
from aiohttp import web                              # noqa: E402

from aura.bus import EventBus                        # noqa: E402
from aura.perception.screen_vlm import (             # noqa: E402
    MATCH_CONFIDENCE_FLOOR, UNMATCHED_STREAK_TRIGGER, ScreenVLM,
)


def test_confidence_gate():
    v = ScreenVLM()
    # well-matched samples never trigger
    for _ in range(20):
        assert not v.note_match(0.95)
    # low confidence must persist before discovery fires
    fires = [v.note_match(0.2) for _ in range(UNMATCHED_STREAK_TRIGGER + 2)]
    assert fires.index(True) == UNMATCHED_STREAK_TRIGGER - 1
    # one good match resets the streak
    assert not v.note_match(MATCH_CONFIDENCE_FLOOR + 0.01)
    assert not v.note_match(0.2)


@pytest.mark.asyncio
async def test_multimodal_request_shape_and_dedupe(monkeypatch):
    monkeypatch.delenv("AURA_LLM_MOCK", raising=False)
    calls = []

    async def chat(request: web.Request) -> web.Response:
        body = await request.json()
        calls.append(body)
        return web.json_response({
            "choices": [{"message": {"content":
                json.dumps({"kind": "demo", "summary": "terminal output"})}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 25}})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    try:
        v = ScreenVLM(base_url=f"http://127.0.0.1:{port}/v1", model="stub-vl")
        v.llm.mock = False
        img = Image.new("RGB", (1280, 720), (30, 30, 30))
        out = await v.classify_screen(img)
        assert out == {"kind": "demo", "summary": "terminal output"}
        # request shape: system + user with [image_url, text] content array
        body = calls[0]
        assert body["model"] == "stub-vl"
        user = body["messages"][1]
        kinds = [c["type"] for c in user["content"]]
        assert kinds == ["image_url", "text"]
        assert user["content"][0]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,")
        # consecutive identical kinds dedupe to None (one call per content)
        assert await v.classify_screen(img) is None
        assert v.summary()["classify_calls"] == 2
        assert v.summary()["prompt_tokens"] == 1800
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_observer_enriches_annotations_in_mock_mode(tmp_path,
                                                          monkeypatch):
    """Full pipeline with mock VLM: detected ink gets text + intent."""
    monkeypatch.setenv("AURA_LLM_MOCK", "1")
    from fpdf import FPDF
    from aura.perception.slides import DeckIndex, ScreenObserver
    pdf_path = tmp_path / "d.pdf"
    pdf = FPDF(orientation="L", unit="mm", format=(180, 320))
    for i in range(3):
        pdf.add_page()
        pdf.set_font("helvetica", "B", 28)
        pdf.set_y(30)
        pdf.cell(0, 16, f"{i+1}. Topic {i+1}", align="C")
    pdf.output(str(pdf_path))

    deck = DeckIndex(pdf_path)
    bus = EventBus()
    got = []
    async def collect(e): got.append(e)
    bus.subscribe("ScreenAnnotationEvent", collect)
    obs = ScreenObserver(bus, deck, capture_fn=lambda: None,
                         patch_dir=tmp_path, vlm=ScreenVLM())
    rng = np.random.default_rng(5)

    def distort(img):
        f = np.array(img.convert("RGB").resize((1180, 660)))
        c = np.full((720, 1280, 3), 18, np.uint8)
        c[30:690, 50:1230] = np.clip(
            f.astype(int) - 12 + rng.integers(-6, 6, f.shape), 0, 255
        ).astype(np.uint8)
        return Image.fromarray(c)

    page = deck.render_page(2)
    for _ in range(3):
        await obs.observe(distort(page))
    ann = page.convert("RGB").copy()
    ImageDraw.Draw(ann).ellipse([300, 180, 430, 290],
                                outline=(220, 40, 40), width=7)
    for _ in range(3):
        await obs.observe(distort(ann))
    await bus.drain()
    assert len(got) == 1
    assert got[0].text == "I do not understand this"
    assert got[0].kind == "question"


@pytest.mark.asyncio
async def test_region_detection_clamping(monkeypatch):
    """VLM region boxes are clamped, validated, and overlap-checked."""
    monkeypatch.setenv("AURA_LLM_MOCK", "1")
    v = ScreenVLM()
    img = Image.new("RGB", (1280, 720))
    out = await v.detect_regions(img)
    assert out["slide_region"] == [0.0, 0.0, 0.78, 1.0]
    assert out["chat_region"] == [0.78, 0.0, 1.0, 1.0]

    # malformed / implausible boxes are rejected, not propagated
    async def bad(_s, _i, _p, max_tokens=120):
        return {"slide_region": [0.4, 0.4, 0.45, 0.45],   # too small
                "chat_region": "garbage"}
    monkeypatch.setattr(v, "_chat_vision", bad)
    out = await v.detect_regions(img)
    assert out == {"slide_region": None, "chat_region": None}


@pytest.mark.asyncio
async def test_vlm_chat_reader_dedupe(monkeypatch):
    """VLM chat path dedupes across samples and respects pane-hash gating."""
    monkeypatch.setenv("AURA_LLM_MOCK", "1")
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent.parent))
    from scripts.ingest_raw import ChatPaneReader
    r = ChatPaneReader(vlm=ScreenVLM())
    pane = Image.new("RGB", (300, 700), (248, 248, 248))
    first = await r.new_messages_vlm(pane)
    assert first == [("Anna", "Where is the data stored?")]
    # identical pane -> hash-gated, no second VLM call result
    assert await r.new_messages_vlm(pane) == []
    # changed pane but same message text -> deduped
    pane2 = Image.new("RGB", (300, 700), (240, 240, 240))
    assert await r.new_messages_vlm(pane2) == []
