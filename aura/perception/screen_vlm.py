"""Tier-2 screen understanding: a VLM explains what the deterministic layer
cannot — but only what it cannot.

Design lessons adopted from vision-based desktop-agent practice:
  * SPECIALIZED ROLES beat one general model: a screen CLASSIFIER (what kind
    of content is showing) and an annotation READER (what does this ink say)
    are separate, narrow calls with strict JSON contracts.
  * NEVER ask a VLM for precise localization — VLMs misplace boxes on dense
    content. The deterministic diff already localized the annotation; the
    VLM only receives the pre-cropped patch and explains its meaning.
  * CONFIDENCE-GATED FALLBACK: the cheap matcher runs every sample; the VLM
    fires only when the matcher's confidence stays low for several samples
    (a live demo, a video, a whiteboard) — "trust the gate, fall back to
    discovery". Per-session call budgets keep token cost bounded.

Backend: any OpenAI-compatible multimodal endpoint. On the AMD box, run a
second vLLM instance serving Qwen2.5-VL-7B-Instruct and point AURA_VLM_URL
at it (the text model keeps :8000; MI300X memory fits both comfortably).
Set AURA_LLM_MOCK=1 for a deterministic offline stand-in.
"""
from __future__ import annotations

import base64
import io
import logging
import os

from ..agents.llm import LLMClient

log = logging.getLogger("aura.screen_vlm")

UNMATCHED_STREAK_TRIGGER = 4      # samples below confidence before classify
CLASSIFY_BUDGET = 12              # max classifier calls per session
READ_BUDGET = 30                  # max annotation reads per session
MATCH_CONFIDENCE_FLOOR = 0.65     # matcher similarity considered "explained"

CLASSIFIER_SYSTEM = (
    "You see one screenshot region from a meeting's shared screen. Classify "
    "what is currently showing. Respond ONLY with JSON: "
    '{"kind": "slide|demo|video|code|whiteboard|document|other", '
    '"summary": "<one sentence, max 20 words>"}'
)

READER_SYSTEM = (
    "You see a small cropped image of a handwritten or typed annotation that "
    "a participant made on a presentation slide. Respond ONLY with JSON: "
    '{"text": "<transcription, empty if pure drawing>", '
    '"intent": "question|emphasis|correction|approval|unclear"}'
)


def _img_b64(img, max_w: int = 960, quality: int = 80) -> str:
    img = img.convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(max_w * img.height / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class ScreenVLM:
    """Two specialized multimodal calls with budgets and a confidence gate."""

    def __init__(self, base_url: str | None = None,
                 model: str | None = None) -> None:
        self.llm = LLMClient(
            base_url=base_url or os.environ.get(
                "AURA_VLM_URL", "http://localhost:8001/v1"),
            model=model or os.environ.get(
                "AURA_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
        self._unmatched_streak = 0
        self._classify_calls = 0
        self._read_calls = 0
        self._last_kind: str | None = None

    # ----------------------------------------------------------- gating
    def note_match(self, similarity: float) -> bool:
        """Feed the matcher's confidence each sample. Returns True when the
        screen has been unexplained long enough to warrant a classify call."""
        if similarity >= MATCH_CONFIDENCE_FLOOR:
            self._unmatched_streak = 0
            self._last_kind = None          # back on the deck
            return False
        self._unmatched_streak += 1
        return (self._unmatched_streak >= UNMATCHED_STREAK_TRIGGER
                and self._classify_calls < CLASSIFY_BUDGET)

    # ----------------------------------------------------------- calls
    async def _chat_vision(self, system: str, img, prompt: str,
                           max_tokens: int = 150) -> dict:
        if self.llm.mock:
            return self._mock(system)
        import aiohttp
        import time as _time
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{_img_b64(img)}"}},
                {"type": "text", "text": prompt},
            ]},
        ]
        t0 = _time.time()
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{self.llm.base_url}/chat/completions",
                json={"model": self.llm.model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": 0.1},
                timeout=aiohttp.ClientTimeout(total=self.llm.timeout_s),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        self.llm.meter.record(data.get("usage", {}),
                              (_time.time() - t0) * 1000)
        from ..agents.llm import _extract_json
        try:
            return _extract_json(data["choices"][0]["message"]["content"])
        except (ValueError, KeyError):
            return {}

    def _mock(self, system: str) -> dict:
        if "Classify" in system:
            return {"kind": "demo", "summary": "A terminal window with "
                                               "code output is visible."}
        return {"text": "I do not understand this", "intent": "question"}

    async def classify_screen(self, img) -> dict | None:
        """What is on screen when it isn't a known slide? Deduplicates
        consecutive identical kinds so a 5-minute demo costs one call."""
        self._classify_calls += 1
        self._unmatched_streak = 0
        out = await self._chat_vision(
            CLASSIFIER_SYSTEM, img, "Classify this screen region.")
        kind = out.get("kind")
        if not kind or kind == self._last_kind:
            return None
        self._last_kind = kind
        log.info("screen content: %s — %s", kind, out.get("summary", ""))
        return out

    async def read_annotation(self, patch_img) -> dict:
        """Explain a pre-localized annotation patch. The crop comes from the
        deterministic diff — the VLM never localizes, only interprets."""
        if self._read_calls >= READ_BUDGET:
            return {}
        self._read_calls += 1
        return await self._chat_vision(
            READER_SYSTEM, patch_img,
            "Transcribe and classify this annotation.")

    def summary(self) -> dict:
        return {"classify_calls": self._classify_calls,
                "read_calls": self._read_calls,
                **self.llm.meter.summary()}
