"""Participant device gateway: tablets/laptops post questions, comments and
reactions over WebSocket; events land on the bus anchored to the current slide.

Includes a minimal embedded HTML client so demo audiences just open a URL on
their phones — no app install.
"""
from __future__ import annotations

import json
import logging

from ..bus import EventBus
from ..events import InteractionEvent, InteractionKind, SlideChange

log = logging.getLogger("aura.interaction")

try:
    from aiohttp import WSMsgType, web
    _WEB_OK = True
except ImportError:  # pragma: no cover
    _WEB_OK = False

CLIENT_HTML = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>AURA participant</title>
<body style="font-family:system-ui;max-width:28rem;margin:2rem auto;padding:0 1rem">
<h3>AURA &mdash; ask / comment / react</h3>
<p>Current slide: <b id=s>?</b></p>
<textarea id=t rows=3 style="width:100%" placeholder="Your question or comment"></textarea>
<div style="display:flex;gap:.5rem;margin-top:.5rem">
<button onclick="send('question')">Ask</button>
<button onclick="send('comment')">Comment</button>
<button onclick="react()">&#128077;</button>
</div>
<p id=ok style="color:green"></p>
<script>
const pid = 'web_' + Math.random().toString(36).slice(2,7);
const ws = new WebSocket((location.protocol=='https:'?'wss':'ws')+'://'+location.host+'/ws');
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.slide) s.textContent = m.slide; };
function send(kind){ if(!t.value.trim()) return;
  ws.send(JSON.stringify({person_id:pid, kind, text:t.value}));
  t.value=''; ok.textContent='sent \u2713'; setTimeout(()=>ok.textContent='',1200); }
function react(){ ws.send(JSON.stringify({person_id:pid, kind:'reaction', text:'\\ud83d\\udc4d'})); }
</script>"""


class InteractionGateway:
    def __init__(self, bus: EventBus, host: str = "0.0.0.0", port: int = 8765) -> None:
        if not _WEB_OK:
            raise RuntimeError("Install aiohttp for the interaction gateway, "
                               "or run with --sim.")
        self.bus = bus
        self.host, self.port = host, port
        self.current_slide = 0
        self._clients: set[web.WebSocketResponse] = set()
        bus.subscribe(SlideChange, self._on_slide)

    async def _on_slide(self, event: SlideChange) -> None:
        self.current_slide = event.slide
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_json({"slide": event.slide, "title": event.title})
            except ConnectionError:
                dead.add(ws)
        self._clients -= dead

    async def _index(self, _req: web.Request) -> web.Response:
        return web.Response(text=CLIENT_HTML, content_type="text/html")

    async def _ws(self, req: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        self._clients.add(ws)
        await ws.send_json({"slide": self.current_slide})
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
                await self.bus.publish(InteractionEvent(
                    person_id=str(data.get("person_id", "web_anon")),
                    kind=InteractionKind(data.get("kind", "comment")),
                    text=str(data.get("text", ""))[:500],
                    slide=self.current_slide,
                    source="device",
                ))
            except (ValueError, KeyError):
                log.warning("malformed interaction payload: %s", msg.data[:200])
        self._clients.discard(ws)
        return ws

    async def run(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        log.info("interaction gateway at http://%s:%d", self.host, self.port)
