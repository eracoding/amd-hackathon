"""AURA room monitor — ONE view-only window, split in two.

LEFT  — proposals & interaction log: the *inference* of the modalities,
        humanized ("person_2 drew on slide 4", "typed question detected",
        "person_1 appears disengaged (gaze)") plus agent proposals.
RIGHT — per-device capture: what each input device is receiving, live —
        camera (per-person attention), microphone (transcript, pace),
        screen (current slide, detected annotations), chat/devices.

Participants never connect here — they interact through the presentation
itself (Teams / PowerPoint Live annotations, chat, voice). This window is
for the room display or the operator. Served on :8766.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..bus import EventBus
from ..events import (
    AgentAction, InteractionEvent, PauseDetected,
    ScreenAnnotationEvent, SessionEnd, TranscriptSegment,
)
from ..fusion.state import RoomStateBuilder

log = logging.getLogger("aura.monitor")

try:
    from aiohttp import WSMsgType, web
    _WEB_OK = True
except ImportError:  # pragma: no cover
    _WEB_OK = False

MONITOR_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AURA — room monitor</title>
<style>
:root{--bg:#10151D;--panel:#171E29;--line:#232C3A;--ink:#E8EDF4;--dim:#8A96A8;
 --ok:#37B2A0;--warn:#E0A458;--alert:#E2574C;--blue:#5B8DD9;
 --mono:ui-monospace,'SF Mono',Cascadia Mono,Consolas,monospace}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;
 height:100vh;display:flex;flex-direction:column;padding:14px;gap:10px}
header{display:flex;justify-content:space-between;align-items:baseline}
header h1{font:700 14px/1 var(--mono);letter-spacing:.06em}
header h1 b{color:var(--ok)}
#status{font:12px var(--mono);color:var(--dim)}
main{flex:1;display:grid;grid-template-columns:1.1fr 1fr;gap:10px;min-height:0}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:10px;
 padding:12px 14px;display:flex;flex-direction:column;min-height:0}
.pane>h2{font:600 11px/1 var(--mono);letter-spacing:.14em;color:var(--dim);
 text-transform:uppercase;margin-bottom:8px}
/* LEFT: log */
#log{flex:1;overflow-y:auto;display:flex;flex-direction:column-reverse}
#log li{list-style:none;padding:7px 0;border-bottom:1px solid var(--line);
 font-size:13.5px;display:flex;gap:8px;align-items:baseline}
.badge{font:700 9.5px var(--mono);letter-spacing:.08em;padding:2px 7px;
 border-radius:99px;border:1px solid;white-space:nowrap}
.b-proposal{color:var(--alert);border-color:var(--alert)}
.b-question{color:var(--warn);border-color:var(--warn)}
.b-annotation{color:var(--blue);border-color:var(--blue)}
.b-gaze{color:var(--dim);border-color:var(--line)}
.b-voice{color:var(--ok);border-color:var(--ok)}
.b-system{color:var(--dim);border-color:var(--line)}
#log time{margin-left:auto;color:var(--dim);font:10.5px var(--mono);white-space:nowrap}
/* RIGHT: capture grid */
#capture{flex:1;display:grid;grid-template-rows:auto auto 1fr auto;gap:10px;min-height:0}
.dev{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.dev h3{font:600 10px/1 var(--mono);letter-spacing:.12em;color:var(--dim);
 text-transform:uppercase;margin-bottom:8px}
.dev h3 span{color:var(--ok)}
.bar{display:flex;align-items:center;gap:8px;margin:4px 0;font:12px var(--mono)}
.bar i{flex:1;height:8px;border-radius:4px;background:var(--line);position:relative;font-style:normal}
.bar i b{position:absolute;inset:0;border-radius:4px;background:var(--ok);transition:width .5s}
.bar.low i b{background:var(--alert)}
.bar em{min-width:42px;text-align:right;font-style:normal}
#mic-txt{color:var(--dim);font-size:12.5px;overflow:hidden;max-height:64px}
#screen-info{font:13px var(--mono)}
#screen-info b{color:var(--ok)}
#anns{color:var(--dim);font-size:12px;margin-top:4px}
#chat li{list-style:none;font-size:12.5px;padding:4px 0;border-bottom:1px solid var(--line)}
.empty{color:var(--dim);font-size:12px}
</style></head><body>
<header><h1>AURA <b>·</b> room monitor</h1><div id="status">waiting for session…</div></header>
<main>
 <div class="pane"><h2>Proposals &amp; interaction log</h2>
  <ul id="log"><li class="empty">observing the room…</li></ul></div>
 <div class="pane"><h2>Per-device capture</h2>
  <div id="capture">
   <div class="dev"><h3>📷 Camera <span>— gaze attention</span></h3>
    <div id="people"><div class="empty">no one detected yet</div></div></div>
   <div class="dev"><h3>🎙 Microphone <span id="wpm"></span></h3>
    <div id="mic-txt" class="empty">listening…</div></div>
   <div class="dev"><h3>🖥 Screen <span>— shared slide</span></h3>
    <div id="screen-info">between slides</div>
    <div id="anns">no annotations yet</div></div>
   <div class="dev"><h3>💬 Chat / devices</h3>
    <ul id="chat"><li class="empty">no typed input yet</li></ul></div>
  </div></div>
</main>
<script>
const $=id=>document.getElementById(id);
const ws=new WebSocket((location.protocol=='https:'?'wss':'ws')+'://'+location.host+'/ws');
let logEmpty=true, chatEmpty=true;
function addLog(badge,text,ts){
 if(logEmpty){$('log').innerHTML='';logEmpty=false}
 const li=document.createElement('li');
 li.innerHTML=`<span class="badge b-${badge}">${badge.toUpperCase()}</span>`+
  `<span>${text}</span><time>${new Date((ts||Date.now()/1000)*1000).toTimeString().slice(0,8)}</time>`;
 $('log').prepend(li);
 while($('log').children.length>40)$('log').lastChild.remove();
}
ws.onmessage=e=>{
 const m=JSON.parse(e.data);
 if(m.type==='state'){
  $('status').textContent=`engagement ${Math.round((m.engagement||0)*100)}% · `+
   `${(m.slope||0)<-0.05?'falling':( (m.slope||0)>0.05?'rising':'steady')}`;
  $('people').innerHTML=(m.people||[]).map(p=>
   `<div class="bar ${p.low?'low':''}"><span>${p.id}</span><i><b style="width:${Math.round(p.score*100)}%"></b></i><em>${Math.round(p.score*100)}%</em></div>`
  ).join('')||'<div class="empty">no one detected yet</div>';
  $('wpm').textContent=m.wpm?`— ${Math.round(m.wpm)} wpm`:'';
  if(m.transcript){$('mic-txt').classList.remove('empty');
   $('mic-txt').textContent='…'+m.transcript.slice(-260)}
  $('screen-info').innerHTML=m.slide?`slide <b>${m.slide}</b> — ${m.slide_title||''}`:'between slides';
  $('anns').textContent=m.annotations_total?`${m.annotations_total} annotation(s) detected on deck`+
   (m.annotations_slide?` · ${m.annotations_slide} on this slide`:''):'no annotations yet';
 } else if(m.type==='log'){ addLog(m.badge,m.text,m.ts);
 } else if(m.type==='chat'){
  if(chatEmpty){$('chat').innerHTML='';chatEmpty=false}
  const li=document.createElement('li');
  li.textContent=`${m.person}: “${m.text}”`;
  $('chat').prepend(li);
  while($('chat').children.length>5)$('chat').lastChild.remove();
 } else if(m.type==='end'){ $('status').textContent='session ended — generating PPT-B / debrief';
  addLog('system','session ended',m.ts); }
};
</script></body></html>"""


class RoomMonitor:
    """View-only split window: inference log (left) + per-device capture
    (right). No participant interaction — the room is observed, not polled."""

    def __init__(self, bus: EventBus, fusion: RoomStateBuilder,
                 host: str = "0.0.0.0", port: int = 8766,
                 push_period_s: float = 1.0) -> None:
        if not _WEB_OK:
            raise RuntimeError("Install aiohttp for the monitor.")
        self.bus = bus
        self.fusion = fusion
        self.host, self.port = host, port
        self.push_period_s = push_period_s
        self._clients: set[web.WebSocketResponse] = set()
        self._stop = asyncio.Event()
        self._gaze_low: dict[str, bool] = {}
        self._ann_total = 0
        self._ann_by_slide: dict[int, int] = {}
        bus.subscribe(AgentAction, self._on_action)
        bus.subscribe(InteractionEvent, self._on_interaction)
        bus.subscribe(ScreenAnnotationEvent, self._on_annotation)
        bus.subscribe(TranscriptSegment, self._on_speech)
        bus.subscribe(PauseDetected, self._on_pause)
        bus.subscribe(SessionEnd, self._on_end)

    # ------------------------------------------------------------ broadcast
    async def _send_all(self, msg: dict) -> None:
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_json(msg)
            except ConnectionError:
                dead.add(ws)
        self._clients -= dead

    async def _log(self, badge: str, text: str, ts: float | None = None) -> None:
        await self._send_all({"type": "log", "badge": badge, "text": text,
                              "ts": ts or time.time()})

    # ------------------------------------------------------------ handlers
    async def _on_action(self, e: AgentAction) -> None:
        if e.action == "noop":
            return
        label = {"nudge": "proposal", "surface_questions": "proposal",
                 "insight": "proposal", "debrief": "system"}.get(e.action,
                                                                 "system")
        await self._log(label, f"{e.agent}: {e.message}", e.ts)

    async def _on_interaction(self, e: InteractionEvent) -> None:
        kind = e.kind.value
        src = "voice" if e.source == "voice" else "typed"
        if kind == "question":
            badge = "voice" if src == "voice" else "question"
            await self._log(badge,
                            f"{e.person_id} asks ({src}): “{e.text}”", e.ts)
        elif kind == "comment":
            await self._log("question",
                            f"{e.person_id} comments: “{e.text}”", e.ts)
        if src == "typed" and e.text:
            await self._send_all({"type": "chat", "person": e.person_id,
                                  "text": e.text[:120]})

    async def _on_annotation(self, e: ScreenAnnotationEvent) -> None:
        self._ann_total += 1
        self._ann_by_slide[e.slide] = self._ann_by_slide.get(e.slide, 0) + 1
        what = "writes" if e.kind == "typed" else "draws"
        await self._log("annotation",
                        f"someone {what} on slide {e.slide}"
                        + (f": “{e.text}”" if e.text else ""), e.ts)

    async def _on_speech(self, _e: TranscriptSegment) -> None:
        pass  # transcript reaches the right pane via state pushes

    async def _on_pause(self, e: PauseDetected) -> None:
        await self._log("system", f"presenter paused ({e.silence_s:.0f}s)",
                        e.ts)

    async def _on_end(self, e: SessionEnd) -> None:
        await self._send_all({"type": "end", "ts": e.ts})
        self._stop.set()

    # ------------------------------------------------------------ state
    def _state_msg(self) -> dict:
        s = self.fusion.snapshot()
        people: dict[str, list[float]] = {}
        now = time.time()
        for ev in self.fusion._attn:  # noqa: SLF001
            if now - ev.ts <= 10.0:
                people.setdefault(ev.person_id, []).append(ev.attention)
        plist = [{"id": p, "score": sum(v) / len(v),
                  "low": sum(v) / len(v) < 0.4}
                 for p, v in sorted(people.items())]
        return {"type": "state",
                "engagement": round(s.engagement, 3),
                "slope": round(s.engagement_slope, 3),
                "slide": s.slide, "slide_title": s.slide_title,
                "people": plist, "wpm": round(s.speech_wpm),
                "transcript": s.transcript_tail,
                "annotations_total": self._ann_total,
                "annotations_slide": self._ann_by_slide.get(s.slide, 0)}

    async def _push_gaze_transitions(self, plist: list[dict]) -> None:
        for p in plist:
            was = self._gaze_low.get(p["id"], False)
            if p["low"] and not was:
                await self._log("gaze", f"{p['id']} appears disengaged "
                                        "(looking away)")
            elif was and not p["low"]:
                await self._log("gaze", f"{p['id']} re-engaged (gaze back "
                                        "on screen)")
            self._gaze_low[p["id"]] = p["low"]

    # ------------------------------------------------------------ server
    async def _index(self, _req: web.Request) -> web.Response:
        return web.Response(text=MONITOR_HTML, content_type="text/html")

    async def _ws(self, req: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        self._clients.add(ws)
        await ws.send_json(self._state_msg())
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
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
        log.info("room monitor at http://%s:%d (view-only)", self.host,
                 self.port)
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self.push_period_s)
                msg = self._state_msg()
                await self._push_gaze_transitions(msg["people"])
                if self._clients:
                    await self._send_all(msg)
        finally:
            await runner.cleanup()
