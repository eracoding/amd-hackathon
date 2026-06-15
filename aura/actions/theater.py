"""AURA Theater — the self-explanatory demo view.

Unlike the analytic RoomMonitor (inference log + capture stats), Theater makes
the session feel LIVE: it serves the actual room and screen recordings and
plays them in sync with the replay clock, streams the transcript word-window
as the presenter "speaks", and overlays the agent reasoning as it fires.

Layout (single window):
  ┌─────────────── TOP: three raw streams ───────────────┐
  │  ROOM CAM (video)  │  SCREEN (video)  │  MIC (live    │
  │  + gaze chips      │  + slide/annot   │  transcript)  │
  ├─────────────── BOTTOM: AURA reasoning ───────────────┤
  │  engagement meter  │  agent feed (Analyst → Coach →   │
  │                    │  Moderator → Summarizer, grouped)│
  └───────────────────────────────────────────────────────┘

Videos are streamed from the recording folder; the page seeks them to
(now - session_start) each tick so they track the replay even at --speed 2.

Served on :8767. View-only.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ..bus import EventBus
from ..events import (
    AgentAction, InteractionEvent, PauseDetected, ScreenAnnotationEvent,
    ScreenStateEvent, SessionEnd, SlideChange, TranscriptSegment,
)
from ..fusion.state import RoomStateBuilder

log = logging.getLogger("aura.theater")

try:
    from aiohttp import WSMsgType, web
    _WEB_OK = True
except ImportError:  # pragma: no cover
    _WEB_OK = False

THEATER_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AURA — live room</title>
<style>
:root{--bg:#0C1016;--panel:#141B25;--line:#222C3A;--ink:#E8EDF4;--dim:#8A96A8;
 --ok:#37B2A0;--warn:#E0A458;--alert:#E2574C;--blue:#5B8DD9;--violet:#9B8CFF;
 --mono:ui-monospace,'SF Mono',Cascadia Mono,Consolas,monospace}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;
 height:100vh;display:flex;flex-direction:column;padding:12px;gap:10px;overflow:hidden}
header{display:flex;justify-content:space-between;align-items:baseline}
header h1{font:700 14px/1 var(--mono);letter-spacing:.08em}
header h1 b{color:var(--ok)}
#clock{font:12px var(--mono);color:var(--dim)}
.streams{display:grid;grid-template-columns:1fr 1fr 0.9fr;gap:10px;height:42vh}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
 padding:10px;display:flex;flex-direction:column;min-height:0}
.card h2{font:600 10px/1 var(--mono);letter-spacing:.13em;color:var(--dim);
 text-transform:uppercase;margin-bottom:8px;display:flex;justify-content:space-between}
.card h2 .live{color:var(--alert)}
.vidwrap{position:relative;flex:1;min-height:0;border-radius:6px;overflow:hidden;
 background:#000;display:flex;align-items:center;justify-content:center}
video{width:100%;height:100%;object-fit:contain;background:#000}
.novid{color:var(--dim);font:12px var(--mono)}
#gazechips{position:absolute;top:6px;left:6px;right:6px;display:flex;flex-wrap:wrap;gap:4px}
.chip{font:600 10px var(--mono);padding:2px 7px;border-radius:99px;
 background:rgba(12,16,22,.82);border:1px solid var(--ok);color:var(--ok)}
.chip.low{border-color:var(--alert);color:var(--alert)}
#slidebadge{position:absolute;bottom:6px;left:6px;font:600 11px var(--mono);
 padding:3px 9px;border-radius:6px;background:rgba(12,16,22,.85);color:var(--ink)}
#annbadge{position:absolute;bottom:6px;right:6px;font:600 10px var(--mono);
 padding:3px 9px;border-radius:6px;background:rgba(226,87,76,.18);
 border:1px solid var(--alert);color:var(--alert);display:none}
#mic{flex:1;overflow-y:auto;display:flex;flex-direction:column-reverse;
 font-size:13.5px;line-height:1.5}
#mic .seg{padding:3px 0;color:var(--dim)}
#mic .seg.now{color:var(--ink)}
#mic .spk{color:var(--violet);font:600 11px var(--mono);margin-right:6px}
.bottom{flex:1;display:grid;grid-template-columns:0.7fr 1.3fr;gap:10px;min-height:0}
#meterbox{display:flex;flex-direction:column;justify-content:center;gap:14px}
#meter{height:30px;border-radius:6px;background:var(--line);position:relative;overflow:hidden}
#meterfill{position:absolute;inset:0;border-radius:6px;transition:width .6s,background .6s;width:0}
#engtxt{font:700 40px/1 var(--mono);text-align:center}
#engsub{font:12px var(--mono);color:var(--dim);text-align:center}
#feed{overflow-y:auto;display:flex;flex-direction:column-reverse;gap:8px}
.act{border-left:3px solid var(--line);padding:7px 11px;background:rgba(255,255,255,.02);
 border-radius:0 8px 8px 0}
.act .who{font:700 10px var(--mono);letter-spacing:.08em;display:flex;
 justify-content:space-between;margin-bottom:3px}
.act .msg{font-size:13.5px;line-height:1.5}
.act.analyst{border-left-color:var(--warn)} .act.analyst .who{color:var(--warn)}
.act.coach{border-left-color:var(--alert)} .act.coach .who{color:var(--alert)}
.act.moderator{border-left-color:var(--blue)} .act.moderator .who{color:var(--blue)}
.act.summarizer{border-left-color:var(--ok)} .act.summarizer .who{color:var(--ok)}
.act .when{color:var(--dim);font-weight:400}
.empty{color:var(--dim);font:12px var(--mono)}
</style></head><body>
<header><h1>AURA <b>·</b> live room</h1><div id="clock">t = 0s</div></header>

<div class="streams">
 <div class="card"><h2>📷 Room camera <span class="live">● LIVE</span></h2>
  <div class="vidwrap">
   <video id="roomvid" muted playsinline></video>
   <div id="gazechips"></div>
   <div class="novid" id="roomnovid" style="display:none">no room video</div>
  </div></div>
 <div class="card"><h2>🖥 Shared screen <span class="live">● LIVE</span></h2>
  <div class="vidwrap">
   <video id="screenvid" muted playsinline></video>
   <div id="slidebadge">—</div>
   <div id="annbadge">✎ annotation</div>
   <div class="novid" id="screennovid" style="display:none">no screen video</div>
  </div></div>
 <div class="card"><h2>🎙 Microphone <span class="live">● LIVE</span></h2>
  <div id="mic"><div class="empty">listening…</div></div></div>
</div>

<div class="bottom">
 <div class="card" id="meterbox">
  <div><div id="engtxt">—</div><div id="engsub">room engagement</div></div>
  <div id="meter"><div id="meterfill"></div></div>
 </div>
 <div class="card"><h2>AURA reasoning — 4 agents</h2>
  <div id="feed"><div class="empty">agents are watching…</div></div></div>
</div>

<script>
const $=id=>document.getElementById(id);
const META=__META__;          // injected: {hasRoom,hasScreen,t0,speed}
const T0=META.t0, SPEED=META.speed||1;
// Absolute base path so resources resolve under a Jupyter/notebook proxy that
// serves this page at /proxy/<port> WITHOUT a trailing slash. A *relative*
// "video/room" would otherwise resolve against /proxy/ and become
// /proxy/video/room (dropping the port segment) -> 404 -> "video failed to
// load". The WebSocket below already does this; the <video> tags must too.
const BASE=(location.pathname.endsWith('/')?location.pathname.slice(0,-1):location.pathname);
const roomV=$('roomvid'), screenV=$('screenvid');
function initVid(v, name, hasIt, novidId){
  if(!hasIt){ v.style.display='none'; $(novidId).style.display='block'; return; }
  v.src=BASE+'/video/'+name+'?t='+Date.now();   // absolute (proxy-safe) + cache-bust
  v.preload='auto'; v.muted=true; v.playsInline=true;
  v.addEventListener('error', ()=>{
    v.style.display='none'; $(novidId).textContent='video failed to load';
    $(novidId).style.display='block';
  });
  v.addEventListener('loadeddata', ()=>{ v.play().catch(()=>{}); });
  v.load();
}
initVid(roomV,'room',META.hasRoom,'roomnovid');
initVid(screenV,'screen',META.hasScreen,'screennovid');

// one user gesture unblocks autoplay+seeking on strict proxies/browsers
let kicked=false;
function kickstart(){ if(kicked) return; kicked=true;
  for(const v of [roomV,screenV]){ if(v.src){ v.play().catch(()=>{}); } }
}
document.addEventListener('click', kickstart, {once:true});
document.body.insertAdjacentHTML('afterbegin',
  '<div id="playhint" style="position:fixed;top:50%;left:50%;'+
  'transform:translate(-50%,-50%);z-index:99;background:rgba(20,27,37,.95);'+
  'border:1px solid #5B8DD9;color:#E8EDF4;padding:14px 22px;border-radius:10px;'+
  'font:600 14px system-ui;cursor:pointer">▶ Click anywhere to start the video streams</div>');
document.addEventListener('click', ()=>{const h=$('playhint'); if(h) h.remove();}, {once:true});

let sessionStart=null, feedEmpty=true, micEmpty=true;
const segs=[];

function syncVideos(elapsed){
  for(const v of [roomV, screenV]){
    if(!v.src||v.readyState<2) continue;
    const dur=v.duration||1e9, target=Math.min(elapsed, dur-0.1);
    // only hard-seek on large drift; small drift self-corrects via playback,
    // so a proxy that refuses range seeks still plays (just may lag slightly)
    if(Math.abs(v.currentTime-target)>1.5){ try{v.currentTime=target}catch(e){} }
    if(v.paused && kicked) v.play().catch(()=>{});
  }
}

const ws=new WebSocket((location.protocol=='https:'?'wss':'ws')+'://'+location.host+
  BASE+'/ws');

ws.onmessage=e=>{
 const m=JSON.parse(e.data);
 if(m.type==='state'){
   if(sessionStart===null && m.ts) sessionStart=m.ts;
   const elapsed=sessionStart!==null?(m.ts-sessionStart):0;
   $('clock').textContent='t = '+Math.round(elapsed)+'s';
   syncVideos(elapsed);
   const eng=m.engagement??0, pct=Math.round(eng*100);
   $('engtxt').textContent=pct+'%';
   const f=$('meterfill'); f.style.width=pct+'%';
   f.style.background=eng<0.4?'var(--alert)':(eng<0.65?'var(--warn)':'var(--ok)');
   $('engsub').textContent='room engagement · '+
     ((m.slope||0)<-0.05?'falling':((m.slope||0)>0.05?'rising':'steady'));
   $('gazechips').innerHTML=(m.people||[]).map(p=>
     `<span class="chip ${p.low?'low':''}">${p.id} ${Math.round(p.score*100)}%</span>`).join('');
   if(m.slide) $('slidebadge').textContent='slide '+m.slide+(m.slide_title?' — '+m.slide_title:'');
   else if(m.screen_kind) $('slidebadge').textContent=m.screen_kind+': '+(m.screen_summary||'');
 } else if(m.type==='transcript'){
   if(micEmpty){$('mic').innerHTML='';micEmpty=false}
   const d=document.createElement('div'); d.className='seg now';
   d.innerHTML=(m.speaker?`<span class="spk">${m.speaker}</span>`:'')+m.text;
   $('mic').prepend(d);
   [...$('mic').children].forEach((c,i)=>{ if(i>0)c.classList.remove('now'); });
   while($('mic').children.length>12)$('mic').lastChild.remove();
 } else if(m.type==='annotation'){
   $('annbadge').style.display='block';
   setTimeout(()=>{$('annbadge').style.display='none'},4000);
 } else if(m.type==='action'){
   if(feedEmpty){$('feed').innerHTML='';feedEmpty=false}
   const a=m.agent.toLowerCase();
   const cls=a.includes('analyst')?'analyst':a.includes('coach')?'coach':
     a.includes('moderator')?'moderator':'summarizer';
   const el=document.createElement('div'); el.className='act '+cls;
   el.innerHTML=`<div class="who"><span>${m.agent}</span>`+
     `<span class="when">t=${m.elapsed||0}s</span></div>`+
     `<div class="msg">${m.message}</div>`;
   $('feed').prepend(el);
   while($('feed').children.length>10)$('feed').lastChild.remove();
 } else if(m.type==='end'){ $('clock').textContent+=' · ended'; }
};
</script></body></html>"""


class TheaterMonitor:
    def __init__(self, bus: EventBus, fusion: RoomStateBuilder,
                 recording_dir: str | Path | None = None,
                 host: str = "0.0.0.0", port: int = 8767,
                 push_period_s: float = 0.5, speed: float = 1.0,
                 room_video: str | None = None,
                 screen_video: str | None = None) -> None:
        if not _WEB_OK:
            raise RuntimeError("Install aiohttp for the theater monitor.")
        self.bus = bus
        self.fusion = fusion
        self.host, self.port = host, port
        self.push_period_s = push_period_s
        self.speed = speed
        self.rec = Path(recording_dir) if recording_dir else None
        self._overrides = {"room": room_video, "screen": screen_video}
        self._clients: set[web.WebSocketResponse] = set()
        self._stop = asyncio.Event()
        self._session_start: float | None = None
        bus.subscribe(AgentAction, self._on_action)
        bus.subscribe(TranscriptSegment, self._on_transcript)
        bus.subscribe(ScreenAnnotationEvent, self._on_annotation)
        bus.subscribe(SessionEnd, self._on_end)

    def _video(self, name: str) -> Path | None:
        # explicit override wins (set via --room-video / --screen-video)
        override = self._overrides.get(name)
        if override and Path(override).exists():
            return Path(override)
        bases = []
        if self.rec:
            bases += [self.rec, self.rec.parent]
        bases.append(Path.cwd())
        # 1) sidecar file next to events.jsonl
        for base in bases:
            for ext in (".mp4", ".webm", ".mov", ".mkv"):
                p = base / f"{name}{ext}"
                if p.exists():
                    return p
        # 2) manifest path, resolved against several plausible roots
        import json
        for base in bases:
            mani = base / "manifest.json"
            if not mani.exists():
                continue
            try:
                streams = json.loads(mani.read_text()).get("streams", {})
            except (ValueError, OSError):
                continue
            if name not in streams:
                continue
            raw = streams[name]["path"]
            cands = [Path(raw), Path(raw).resolve(),
                     base / raw, base / Path(raw).name,
                     base.parent / raw, base.parent / Path(raw).name,
                     Path.cwd() / raw, Path.cwd() / Path(raw).name]
            for c in cands:
                if c.exists():
                    return c
        return None
        for ext in (".mp4", ".webm", ".mov", ".mkv"):
            p = self.rec / f"{name}{ext}"
            if p.exists():
                return p
        # manifest may name it differently
        import json
        mani = self.rec / "manifest.json"
        if mani.exists():
            streams = json.loads(mani.read_text()).get("streams", {})
            if name in streams:
                cand = self.rec / Path(streams[name]["path"]).name
                if cand.exists():
                    return cand
                # path may be relative to the recording's parent
                cand2 = (self.rec / streams[name]["path"])
                if cand2.exists():
                    return cand2
        return None

    async def _send_all(self, msg: dict) -> None:
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_json(msg)
            except ConnectionError:
                dead.add(ws)
        self._clients -= dead

    def _elapsed(self, ts: float) -> int:
        if self._session_start is None:
            self._session_start = ts
        return int(ts - self._session_start)

    async def _on_action(self, e: AgentAction) -> None:
        if e.action == "noop":
            return
        await self._send_all({"type": "action", "agent": e.agent,
                              "message": e.message,
                              "elapsed": self._elapsed(e.ts)})

    async def _on_transcript(self, e: TranscriptSegment) -> None:
        await self._send_all({"type": "transcript", "text": e.text,
                              "speaker": "presenter"})

    async def _on_annotation(self, _e: ScreenAnnotationEvent) -> None:
        await self._send_all({"type": "annotation"})

    async def _on_end(self, _e: SessionEnd) -> None:
        await self._send_all({"type": "end"})
        self._stop.set()

    def _state_msg(self) -> dict:
        s = self.fusion.snapshot()
        people: dict[str, list[float]] = {}
        now = time.time()
        for ev in self.fusion._attn:  # noqa: SLF001
            if now - ev.ts <= 10.0:
                people.setdefault(ev.person_id, []).append(ev.attention)
        return {"type": "state", "ts": s.ts,
                "engagement": round(s.engagement, 3),
                "slope": round(s.engagement_slope, 3),
                "slide": s.slide, "slide_title": s.slide_title,
                "screen_kind": (s.screen_state or {}).get("kind"),
                "screen_summary": (s.screen_state or {}).get("summary"),
                "people": [{"id": p, "score": sum(v) / len(v),
                            "low": sum(v) / len(v) < 0.4}
                           for p, v in sorted(people.items())]}

    # ------------------------------------------------------------ server
    async def _index(self, _req: web.Request) -> web.Response:
        import json as _json
        meta = {"hasRoom": self._video("room") is not None,
                "hasScreen": self._video("screen") is not None,
                "t0": 0, "speed": self.speed}
        html = THEATER_HTML.replace("__META__", _json.dumps(meta))
        return web.Response(text=html, content_type="text/html")

    async def _serve_video(self, req: web.Request) -> web.StreamResponse:
        name = req.match_info["name"]
        path = self._video(name)
        if path is None:
            return web.Response(status=404, text="no video")
        ctype = {".mp4": "video/mp4", ".webm": "video/webm",
                 ".mov": "video/quicktime", ".mkv": "video/x-matroska"
                 }.get(path.suffix.lower(), "video/mp4")
        # FileResponse honors Range requests (needed for seeking); set an
        # explicit content-type and accept-ranges so proxies/browsers comply.
        resp = web.FileResponse(path, headers={
            "Content-Type": ctype,
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        })
        return resp

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
        app.router.add_get("/video/{name}", self._serve_video)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        log.info("theater monitor at http://%s:%d", self.host, self.port)
        rv, sv = self._video("room"), self._video("screen")
        log.info("theater room video:   %s", rv or "NOT FOUND (no room video)")
        log.info("theater screen video: %s", sv or "NOT FOUND (no screen video)")
        if rv is None and sv is None:
            log.warning("no videos resolved — pass --room-video / --screen-video "
                        "with explicit paths, or place room.mp4 / screen.mp4 "
                        "next to events.jsonl")
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self.push_period_s)
                if self._clients:
                    await self._send_all(self._state_msg())
        finally:
            await runner.cleanup()
