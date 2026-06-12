"""Slide awareness: which slide is on screen, and what does it say.

Three sources, layered:
  * DeckIndex        — parse the presentation (PDF) once: per-page text +
                       perceptual hash of the rendered page. Export your .pptx
                       to PDF (File > Export) — universal, no Office APIs.
  * ScreenSlideTracker — capture the presenter screen (live via mss, or frames
                       from a screen *recording* for offline ingestion), hash
                       each frame, match against the deck. Emits SlideChange
                       with the page's text content so agents reason about
                       what is actually on screen. No VLM tokens needed.
  * ManualSlideControl — zero-dependency fallback: next()/prev()/goto().

A VLM (Qwen2.5-VL on the same vLLM server) is the optional escalation path
for content hashing can't see (live demos, videos, whiteboards) — interface
stubbed in `describe_frame_with_vlm`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ..bus import EventBus
from ..events import ScreenAnnotationEvent, ScreenStateEvent, SlideChange

log = logging.getLogger("aura.slides")

try:
    import numpy as np
    import pypdfium2 as pdfium
    from PIL import Image
    _PDF_OK = True
except ImportError:  # pragma: no cover
    _PDF_OK = False

try:  # live screen capture is optional (offline ingestion doesn't need it)
    import mss  # noqa: F401
    _MSS_OK = True
except ImportError:  # pragma: no cover
    _MSS_OK = False

HASH_SIZE = 8                # dHash -> 64-bit (kept for cheap change gating)
SIG_SIZE = 48                # NCC signature resolution
MIN_SIM = 0.75               # absolute floor for accepting a page match
MIN_MARGIN = 0.02            # best must beat second-best by this much
SURE_SIM = 0.97              # above this, accept even without margin
                             # (near-duplicate slides: best guess is correct)
DEBOUNCE_FRAMES = 2          # consecutive agreeing frames before emitting


def dhash(img: "Image.Image", size: int = HASH_SIZE) -> int:
    """Difference hash: robust to scaling, compression, mild brightness."""
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0]) if bits.size == 64 else \
        int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _content_crop(img: "Image.Image") -> "Image.Image":
    """Crop away letterbox/pillarbox borders: estimate the border color from
    the image corners, keep the bounding box of everything that differs."""
    g = np.asarray(img.convert("L"), dtype=np.int16)
    c = 8
    corners = np.concatenate([g[:c, :c].ravel(), g[:c, -c:].ravel(),
                              g[-c:, :c].ravel(), g[-c:, -c:].ravel()])
    border = int(np.median(corners))
    mask = np.abs(g - border) > 25
    if mask.mean() < 0.02:          # nearly uniform frame — keep as is
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img.crop((int(cols[0]), int(rows[0]),
                     int(cols[-1]) + 1, int(rows[-1]) + 1))


def signature(img: "Image.Image", size: int = SIG_SIZE) -> "np.ndarray":
    """Zero-mean, unit-norm grayscale thumbnail of the content area.
    Cosine similarity between signatures discriminates text slides that share
    a template — which a 64-bit perceptual hash cannot."""
    g = np.asarray(_content_crop(img).convert("L").resize(
        (size, size), Image.LANCZOS), dtype=np.float32)
    g -= g.mean()
    n = float(np.linalg.norm(g))
    return (g / n).flatten() if n > 1e-6 else g.flatten()


class DeckIndex:
    """Per-page text + perceptual hash for a PDF deck."""

    def __init__(self, pdf_path: str | Path, render_scale: float = 1.0) -> None:
        if not _PDF_OK:
            raise RuntimeError("pip install pypdfium2 pillow numpy")
        self.path = Path(pdf_path)
        doc = pdfium.PdfDocument(str(self.path))
        self.pages: list[dict] = []
        for i, page in enumerate(doc):
            text = page.get_textpage().get_text_range() or ""
            bmp = page.render(scale=render_scale)
            img = bmp.to_pil()
            title = next((ln.strip() for ln in text.splitlines() if ln.strip()),
                         f"Slide {i + 1}")
            self.pages.append({"n": i + 1, "title": title[:120],
                               "text": " ".join(text.split())[:800],
                               "hash": dhash(img), "sig": signature(img)})
        doc.close()
        log.info("deck indexed: %s (%d pages)", self.path.name, len(self.pages))

    def __len__(self) -> int:
        return len(self.pages)

    def match(self, img: "Image.Image") -> tuple[int | None, float]:
        """Return (page_number, similarity) for the best-matching page, or
        (None, sim) when no page is confidently identifiable — e.g. a live
        demo, video, or transition is on screen."""
        s = signature(img)
        sims = sorted(((float(s @ p["sig"]), p["n"]) for p in self.pages),
                      reverse=True)
        (best_sim, best_n), second = sims[0], (sims[1][0] if len(sims) > 1 else -1.0)
        if best_sim >= SURE_SIM or (
                best_sim >= MIN_SIM and best_sim - second >= MIN_MARGIN):
            return best_n, best_sim
        return None, best_sim

    def page(self, n: int) -> dict:
        return self.pages[n - 1]

    def render_page(self, n: int, scale: float = 1.0) -> "Image.Image":
        """Lazy render cache — used by ScreenObserver (clean reference) and
        the PPT-B generator (slide backgrounds)."""
        key = (n, scale)
        if not hasattr(self, "_render_cache"):
            self._render_cache = {}
        if key not in self._render_cache:
            doc = pdfium.PdfDocument(str(self.path))
            self._render_cache[key] = doc[n - 1].render(scale=scale).to_pil()
            doc.close()
        return self._render_cache[key]


class ScreenSlideTracker:
    """Matches captured frames against the deck; emits SlideChange on change.

    `capture_fn` returns a PIL Image (or None). Default: live primary-screen
    grab via mss. For offline ingestion, inject frames from a recording.
    """

    def __init__(self, bus: EventBus, deck: DeckIndex,
                 capture_fn=None, period_s: float = 0.5) -> None:
        self.bus = bus
        self.deck = deck
        self.period_s = period_s
        self.current: int | None = None
        self._pending: tuple[int | None, int] = (None, 0)  # (page, streak)
        if capture_fn is None:
            if not _MSS_OK:
                raise RuntimeError("pip install mss for live screen capture, "
                                   "or inject capture_fn (offline ingestion).")
            self._sct = mss.mss()
            capture_fn = self._grab_primary
        self.capture_fn = capture_fn
        self._stop = asyncio.Event()

    def _grab_primary(self):
        mon = self._sct.monitors[1]
        raw = self._sct.grab(mon)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    async def observe(self, img, ts: float | None = None) -> SlideChange | None:
        """Feed one frame; returns the SlideChange if one was emitted."""
        page, sim = self.deck.match(img)
        self.last_similarity = sim
        if page is None or page == self.current:
            self._pending = (None, 0)
            return None
        last, streak = self._pending
        streak = streak + 1 if page == last else 1
        self._pending = (page, streak)
        if streak < DEBOUNCE_FRAMES:
            return None
        self.current = page
        self._pending = (None, 0)
        p = self.deck.page(page)
        event = SlideChange(slide=page, title=p["title"], content=p["text"],
                            source="screen")
        if ts is not None:
            event.ts = ts
        await self.bus.publish(event)
        log.info("slide -> %d (%s)", page, p["title"])
        return event

    async def run(self) -> None:
        log.info("screen slide tracker started (period=%.1fs)", self.period_s)
        while not self._stop.is_set():
            img = await asyncio.to_thread(self.capture_fn)
            if img is not None:
                await self.observe(img)
            await asyncio.sleep(self.period_s)

    def stop(self) -> None:
        self._stop.set()


class ManualSlideControl:
    """Fallback slide source: presenter advances slides explicitly."""

    def __init__(self, bus: EventBus, deck: DeckIndex | None = None) -> None:
        self.bus = bus
        self.deck = deck
        self.current = 0

    async def goto(self, n: int) -> None:
        self.current = max(1, n if self.deck is None else min(n, len(self.deck)))
        if self.deck:
            p = self.deck.page(self.current)
            title, content = p["title"], p["text"]
        else:
            title, content = f"Slide {self.current}", ""
        await self.bus.publish(SlideChange(slide=self.current, title=title,
                                           content=content, source="manual"))

    async def next(self) -> None:
        await self.goto(self.current + 1)

    async def prev(self) -> None:
        await self.goto(self.current - 1)


async def describe_frame_with_vlm(llm_client, img: "Image.Image") -> dict:
    """Optional VLM escalation (Qwen2.5-VL on the same vLLM server) for
    on-screen content the deck can't explain: live demos, videos, whiteboard.
    Costs real tokens — call only when DeckIndex.match returns None for a
    sustained period."""
    import base64
    import io
    buf = io.BytesIO()
    img.convert("RGB").resize((960, int(960 * img.height / img.width))).save(
        buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    # OpenAI-compatible multimodal message; requires a VL model being served.
    return await llm_client.chat_json(
        system=("You describe what is currently on a presenter's screen. "
                'Respond ONLY with JSON: {"kind": "demo|video|code|whiteboard'
                '|document|other", "summary": str}'),
        user=f'<image data follows> data:image/jpeg;base64,{b64[:200000]}',
    )


# ===========================================================================
# ScreenObserver: slide tracking + on-slide annotation detection
# ===========================================================================
DIFF_SIZE = (640, 360)        # diff resolution (w, h)
DIFF_THRESH = 55              # |captured - reference| per-pixel threshold
MIN_REGION_PX = 60            # ignore smaller specks at DIFF_SIZE
MERGE_PAD = 14                # merge components closer than this (px @ DIFF_SIZE)
STABLE_FRAMES = 2             # region must persist N samples before emitting


class ScreenObserver(ScreenSlideTracker):
    """Extends slide tracking with annotation detection.

    For the workflow where participants draw/type directly on the shared
    slide (Teams / PowerPoint Live annotation mode): once a frame is matched
    to a deck page, the content area is diffed against the clean page
    render. New stable regions = annotations -> ScreenAnnotationEvent with a
    normalized bbox and a cropped PNG patch (used later by the PPT-B
    generator to re-draw the team's ink onto the deck).
    """

    def __init__(self, bus: EventBus, deck: DeckIndex,
                 capture_fn=None, period_s: float = 0.5,
                 patch_dir: str | Path = "sessions/annotations",
                 vlm=None) -> None:
        super().__init__(bus, deck, capture_fn=capture_fn, period_s=period_s)
        self.vlm = vlm                 # optional ScreenVLM (Tier-2)
        self.patch_dir = Path(patch_dir)
        self.patch_dir.mkdir(parents=True, exist_ok=True)
        self._refs: dict[int, "np.ndarray"] = {}        # page -> gray ref
        self._acc: dict[int, "np.ndarray"] = {}         # page -> emitted mask
        self._pending_regions: list[dict] = []           # {page, box, streak}
        self._patch_n = 0

    # ---------------------------------------------------------------- refs
    def _ref_gray(self, page: int) -> "np.ndarray":
        import cv2
        if page not in self._refs:
            img = self.deck.render_page(page)
            g = cv2.cvtColor(
                cv2.resize(np.asarray(img.convert("RGB")), DIFF_SIZE),
                cv2.COLOR_RGB2GRAY)
            g = cv2.GaussianBlur(g, (3, 3), 0)
            # text/graphic edges shimmer under video compression; a dilated
            # edge mask excludes them from the diff so they can't fire as
            # phantom annotations. Real ink survives: it lands in whitespace
            # or is far larger than the 2-px edge band.
            edges = cv2.dilate(cv2.Canny(g, 60, 120),
                               np.ones((5, 5), np.uint8))
            self._refs[page] = (g, edges)
        return self._refs[page]

    # ------------------------------------------------------------- observe
    async def observe(self, img, ts: float | None = None):
        """Slide tracking (parent) + annotation diffing on the current page
        + VLM escalation when the screen shows something off-deck."""
        event = await super().observe(img, ts=ts)
        if self.vlm is not None and self.vlm.note_match(self.last_similarity):
            out = await self.vlm.classify_screen(img)
            if out:
                se = ScreenStateEvent(kind=out.get("kind", "other"),
                                      summary=out.get("summary", ""),
                                      source="screen")
                if ts is not None:
                    se.ts = ts
                await self.bus.publish(se)
        if self.current is not None:
            await self._detect_annotations(img, ts=ts)
        return event

    async def _detect_annotations(self, img, ts: float | None = None) -> None:
        import cv2
        page = self.current
        content = _content_crop(img)
        cur = cv2.cvtColor(
            cv2.resize(np.asarray(content.convert("RGB")), DIFF_SIZE),
            cv2.COLOR_RGB2GRAY)
        cur = cv2.GaussianBlur(cur, (3, 3), 0)
        ref, edge_mask = self._ref_gray(page)
        diff = (cv2.absdiff(cur, ref) > DIFF_THRESH).astype(np.uint8)
        diff[edge_mask > 0] = 0
        acc = self._acc.setdefault(
            page, np.zeros(DIFF_SIZE[::-1], dtype=np.uint8))
        new = cv2.morphologyEx(diff & ~acc, cv2.MORPH_OPEN,
                               np.ones((2, 2), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(new)
        w, h = DIFF_SIZE
        # components -> candidate boxes; merge boxes closer than MERGE_PAD px
        # (one stroke often splits where it crosses masked text edges)
        cands = [tuple(stats[i][:4]) + (int(stats[i][4]), i)
                 for i in range(1, n) if stats[i][4] >= MIN_REGION_PX // 3]
        merged: list[list] = []          # [x0, y0, x1, y1, area, {labels}]
        for x, y, bw, bh, area, li in cands:
            box = [x, y, x + bw, y + bh, area, {li}]
            while True:
                hit = next((m for m in merged if
                            box[0] - MERGE_PAD < m[2] and
                            m[0] - MERGE_PAD < box[2] and
                            box[1] - MERGE_PAD < m[3] and
                            m[1] - MERGE_PAD < box[3]), None)
                if hit is None:
                    break
                merged.remove(hit)
                box = [min(box[0], hit[0]), min(box[1], hit[1]),
                       max(box[2], hit[2]), max(box[3], hit[3]),
                       box[4] + hit[4], box[5] | hit[5]]
            merged.append(box)
        emitted_boxes = []
        next_pending: list[dict] = []
        for x0, y0, x1, y1, area, lset in merged:
            if area < MIN_REGION_PX:
                continue
            bw, bh = x1 - x0, y1 - y0
            # match against pending regions by overlap (noise jitters exact
            # coordinates frame to frame; overlap matching is stable)
            match = next((p for p in self._pending_regions
                          if p["page"] == page and
                          x0 - MERGE_PAD < p["box"][2] and
                          p["box"][0] - MERGE_PAD < x1 and
                          y0 - MERGE_PAD < p["box"][3] and
                          p["box"][1] - MERGE_PAD < y1), None)
            streak = (match["streak"] if match else 0) + 1
            if match:
                self._pending_regions.remove(match)
            if streak < STABLE_FRAMES:
                next_pending.append({"page": page, "streak": streak,
                                     "box": [x0, y0, x1, y1]})
                continue
            for li in lset:
                acc[labels == li] = 1
            emitted_boxes.append((x0, y0, x1, y1))
            bbox = [x0 / w, y0 / h, x1 / w, y1 / h]
            patch = self._save_patch(content, bbox)
            text, intent = "", ""
            if self.vlm is not None:
                read = await self.vlm.read_annotation(Image.open(patch))
                text, intent = read.get("text", ""), read.get("intent", "")
            e = ScreenAnnotationEvent(
                slide=page, bbox=[round(v, 4) for v in bbox],
                area_frac=round(area / (w * h), 5),
                kind=intent or ("typed" if bw > 3.5 * bh and bh < h * 0.06
                                else "drawn"),
                text=text, patch_path=patch, source="screen")
            if ts is not None:
                e.ts = ts
            await self.bus.publish(e)
            log.info("annotation on slide %d bbox=%s (%s)",
                     page, e.bbox, e.kind)
        # pending regions that vanished this frame are dropped implicitly
        self._pending_regions = [p for p in self._pending_regions
                                 if p["page"] != page] + next_pending

    def _save_patch(self, content_img, bbox) -> str:
        W, H = content_img.size
        pad = 6
        x0 = max(0, int(bbox[0] * W) - pad)
        y0 = max(0, int(bbox[1] * H) - pad)
        x1 = min(W, int(bbox[2] * W) + pad)
        y1 = min(H, int(bbox[3] * H) + pad)
        self._patch_n += 1
        p = self.patch_dir / f"slide{self.current}_a{self._patch_n}.png"
        crop = content_img.crop((x0, y0, x1, y1)).convert("RGB")
        # make the slide background transparent so the pasted patch overlays
        # only the ink, not a rectangle of captured pixels
        a = np.asarray(crop, dtype=np.int16)
        edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        bg = np.median(edge, axis=0)
        alpha = (np.abs(a - bg).sum(axis=2) > 70).astype(np.uint8) * 255
        rgba = np.dstack([a.astype(np.uint8), alpha])
        Image.fromarray(rgba, "RGBA").save(p)
        return str(p)
