"""LLM backend client.

Speaks the OpenAI-compatible chat API exposed by vLLM (ROCm build) serving
Qwen2.5-7B-Instruct locally. Tracks token usage per call because token
efficiency is an explicit evaluation criterion.

Set AURA_LLM_MOCK=1 to develop/test the full pipeline with a deterministic
rule-based stand-in (no GPU required) — useful before Phase 0 completes.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

log = logging.getLogger("aura.llm")

try:
    import aiohttp
    _HTTP_OK = True
except ImportError:  # pragma: no cover
    _HTTP_OK = False


@dataclass
class UsageMeter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, usage: dict, latency_ms: float) -> None:
        self.calls += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict:
        lat = sorted(self.latencies_ms)
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "median_latency_ms": lat[len(lat) // 2] if lat else 0,
        }


class LLMClient:
    def __init__(self,
                 base_url: str | None = None,
                 model: str | None = None,
                 timeout_s: float = 60.0) -> None:
        self.base_url = (base_url or os.environ.get(
            "AURA_LLM_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.environ.get(
            "AURA_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.timeout_s = timeout_s
        self.mock = os.environ.get("AURA_LLM_MOCK", "0") == "1"
        self.meter = UsageMeter()

    async def chat_json(self, system: str, user: str,
                        max_tokens: int = 400, temperature: float = 0.2) -> dict:
        """Chat completion that must return JSON; retries once on parse failure."""
        if self.mock:
            return self._mock_response(system, user)
        if not _HTTP_OK:
            raise RuntimeError("Install aiohttp, or set AURA_LLM_MOCK=1.")

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        for attempt in range(2):
            t0 = time.time()
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": self.model, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature},
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            self.meter.record(data.get("usage", {}), (time.time() - t0) * 1000)
            text = data["choices"][0]["message"]["content"]
            try:
                return _extract_json(text)
            except ValueError:
                log.warning("non-JSON LLM output (attempt %d): %.120s", attempt, text)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user",
                                 "content": "Respond again with ONLY valid JSON."})
        return {"action": "noop", "reason": "json_parse_failed"}

    # ------------------------------------------------------------- mock
    @staticmethod
    def _mock_response(system: str, user: str) -> dict:
        """Deterministic heuristic stand-in keyed on the agent role in `system`."""
        state = {}
        try:
            state = json.loads(user[user.index("{"):])
        except (ValueError, json.JSONDecodeError):
            pass
        eng = state.get("engagement", {})
        if "EngagementAnalyst" in system:
            slope = eng.get("slope_per_min", 0)
            wpm = state.get("speech", {}).get("wpm", 0)
            low = slope < -0.1 or eng.get("now", 1) < 0.45
            slide = state.get("slide", {})
            return {"finding": "engagement_drop" if low else "stable",
                    "confidence": 0.82 if low else 0.6,
                    "probable_cause": (
                        f"pace rose to ~{wpm} wpm on dense material "
                        f"(slide {slide.get('n','?')}: {slide.get('title','')})"
                        if low else "attention steady, pace comfortable"),
                    "evidence": (f"engagement {eng.get('now')}, slope "
                                 f"{slope}/min" if low else ""),
                    "severity": "high" if low else "low"}
        if "Moderator" in system:
            qs = state.get("pending_questions", [])
            return {"action": "surface_questions" if qs else "noop",
                    "question_ids": [q["id"] for q in qs[:3]],
                    "summary": "; ".join(q["text"] for q in qs[:3])}
        if "PresenterCoach" in system:
            an = state.get("notes", {}).get("analyst", {})
            wpm = state.get("speech", {}).get("wpm", 0)
            return {"action": "nudge",
                    "what_went_wrong": an.get("probable_cause",
                                              "pace outran comprehension"),
                    "message": (f"You hit ~{wpm} wpm on dense content and the "
                                "room drifted — pause, restate the core idea "
                                "in one plain sentence, then check it landed "
                                "before moving on."),
                    "priority": "high"}
        if "Summarizer" in system:
            per_slide, unresolved = [], []
            if "SLIDE_STATS:" in user:
                try:
                    raw = user.split("SLIDE_STATS:", 1)[1].split("\n", 1)[0]
                    stats = json.loads(raw)
                    per_slide = [{"slide": s["slide"],
                                  "engagement": s["engagement"],
                                  "note": s.get("title", "")} for s in stats]
                    dip = min((s for s in stats if s["engagement"] is not None),
                              key=lambda s: s["engagement"], default=None)
                    summary = ("Session completed. Lowest engagement on slide "
                               f"{dip['slide']} ({dip['title']}, "
                               f"{dip['engagement']:.2f})." if dip
                               else "Session completed.")
                except (ValueError, KeyError):
                    summary = "Session completed."
            else:
                summary = "Session completed."
            unresolved = [q["text"] for q in state.get("pending_questions", [])]
            return {"summary": summary, "per_slide": per_slide,
                    "unresolved_questions": unresolved,
                    "action_items": ["Follow up on participant questions.",
                                     "Simplify the densest slide before reuse."]}
        return {"action": "noop"}


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction: handles markdown fences and prose wrappers."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])
