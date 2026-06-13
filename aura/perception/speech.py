"""Presenter speech pipeline: mic -> faster-whisper -> TranscriptSegment events.

faster-whisper (CTranslate2) runs int8 on CPU comfortably for `small`, keeping
the GPU free for the LLM. Audio is captured in fixed chunks with simple
energy-based VAD; sustained silence emits PauseDetected for the Moderator agent.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..bus import EventBus
from ..events import (InteractionEvent, InteractionKind, PauseDetected,
                      TranscriptSegment)

log = logging.getLogger("aura.speech")

try:
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
    _AUDIO_OK = True
except (ImportError, OSError):  # pragma: no cover
    _AUDIO_OK = False  # OSError: sounddevice without PortAudio installed

META_QUESTION = __import__("re").compile(
    r"(any (more |other )?questions|do (we|you) have (a |any )?question"
    r"|what('s| is) your question|you have a question|questions so far"
    r"|des questions|avez-vous des questions)", __import__("re").IGNORECASE)


def is_meta_question(text: str) -> bool:
    """Presenter prompts ABOUT questions are not audience questions."""
    return bool(META_QUESTION.search(text))

SAMPLE_RATE = 16_000
CHUNK_S = 5.0
SILENCE_RMS = 0.01
PAUSE_THRESHOLD_S = 3.0


class SpeechPipeline:
    def __init__(self, bus: EventBus, model_size: str = "small",
                 device: str = "cpu", compute_type: str = "int8") -> None:
        if not _AUDIO_OK:
            raise RuntimeError("Install faster-whisper + sounddevice for live "
                               "audio, or run with --sim.")
        self.bus = bus
        log.info("loading faster-whisper %s (%s/%s)", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._stop = asyncio.Event()
        self._silence_since: float | None = None

    async def run(self) -> None:
        log.info("speech pipeline started (chunk=%.1fs)", CHUNK_S)
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            t0 = time.time()
            audio = await loop.run_in_executor(
                None,
                lambda: sd.rec(int(CHUNK_S * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                               channels=1, dtype="float32", blocking=True).flatten(),
            )
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < SILENCE_RMS:
                if self._silence_since is None:
                    self._silence_since = t0
                elif time.time() - self._silence_since >= PAUSE_THRESHOLD_S:
                    await self.bus.publish(PauseDetected(
                        silence_s=time.time() - self._silence_since, source="mic"))
                    self._silence_since = None
                continue
            self._silence_since = None

            segments, _info = await loop.run_in_executor(
                None, lambda: self.model.transcribe(audio, language="en",
                                                    vad_filter=True))
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                dur = max(0.5, seg.end - seg.start)
                await self.bus.publish(TranscriptSegment(
                    text=text, t_start=t0 + seg.start, t_end=t0 + seg.end,
                    words_per_min=len(text.split()) / dur * 60.0, source="mic",
                ))
                if text.rstrip().endswith("?") and not is_meta_question(text):
                    # voice question heuristic (diarization = future work)
                    await self.bus.publish(InteractionEvent(
                        person_id="voice", kind=InteractionKind.question,
                        text=text, source="voice"))

    def stop(self) -> None:
        self._stop.set()
