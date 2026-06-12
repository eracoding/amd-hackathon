"""Validates the real (non-mock) LLM path against a stub vLLM server.

Covers: OpenAI-compatible request shape, clean-JSON parsing, markdown-fenced
JSON tolerance, the retry-on-garbage loop, and token usage metering. The only
thing not exercised is the model weights — which is exactly the part that
swaps in unchanged when vLLM serves Qwen2.5 on the AMD box.
"""
import asyncio
import json

import pytest
from aiohttp import web

from aura.agents.llm import LLMClient

RESPONSES = {
    "clean": '{"finding": "stable", "confidence": 0.7}',
    "fenced": '```json\n{"finding": "engagement_drop", "severity": "high"}\n```',
    "prose_wrapped": 'Sure! Here is the analysis: {"action": "nudge", '
                     '"message": "slow down"} hope that helps.',
    "garbage_then_clean": None,  # handled statefully below
}


def _make_app(script: list[str]):
    calls = {"n": 0, "payloads": []}

    async def chat(request: web.Request) -> web.Response:
        body = await request.json()
        calls["payloads"].append(body)
        key = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        if key == "garbage":
            content = "I refuse to emit JSON today."
        else:
            content = RESPONSES[key]
        return web.json_response({
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        })

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    return app, calls


async def _serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    return runner, port


@pytest.mark.asyncio
@pytest.mark.parametrize("key,expect", [
    ("clean", {"finding": "stable", "confidence": 0.7}),
    ("fenced", {"finding": "engagement_drop", "severity": "high"}),
    ("prose_wrapped", {"action": "nudge", "message": "slow down"}),
])
async def test_json_extraction_variants(key, expect, monkeypatch):
    monkeypatch.delenv("AURA_LLM_MOCK", raising=False)
    app, calls = _make_app([key])
    runner, port = await _serve(app)
    try:
        client = LLMClient(base_url=f"http://127.0.0.1:{port}/v1", model="stub")
        client.mock = False
        out = await client.chat_json("system", "user")
        assert out == expect
        # request shape matches what vLLM expects
        body = calls["payloads"][0]
        assert body["model"] == "stub"
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_retry_then_success_and_metering(monkeypatch):
    monkeypatch.delenv("AURA_LLM_MOCK", raising=False)
    app, calls = _make_app(["garbage", "clean"])
    runner, port = await _serve(app)
    try:
        client = LLMClient(base_url=f"http://127.0.0.1:{port}/v1", model="stub")
        client.mock = False
        out = await client.chat_json("system", "user")
        assert out == {"finding": "stable", "confidence": 0.7}
        assert calls["n"] == 2, "should have retried exactly once"
        m = client.meter.summary()
        assert m["llm_calls"] == 2
        assert m["prompt_tokens"] == 240 and m["completion_tokens"] == 60
        assert m["median_latency_ms"] > 0
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gives_noop_after_exhausted_retries(monkeypatch):
    monkeypatch.delenv("AURA_LLM_MOCK", raising=False)
    app, _ = _make_app(["garbage", "garbage"])
    runner, port = await _serve(app)
    try:
        client = LLMClient(base_url=f"http://127.0.0.1:{port}/v1", model="stub")
        client.mock = False
        out = await client.chat_json("system", "user")
        assert out == {"action": "noop", "reason": "json_parse_failed"}
    finally:
        await runner.cleanup()
