# AURA — Agentic Understanding of Room Attention

Multimodal, agentic room intelligence for collaborative presentations.
Participants interact through the presentation itself — drawing/typing on the
shared slide (Teams / PowerPoint Live), chat, and voice. AURA observes
(camera gaze, mic, screen capture), fuses everything into a live RoomState,
reasons over it with a crew of LLM agents, shows its inference in one
view-only split monitor (interaction log + proposals | per-device capture),
and produces **PPT-B**: the original deck enriched with the team's detected
annotations, per-slide capture stats, and a final synthesis slide.

**Track 2 (Multimodal)** with a **Track 1 (Agents)** reasoning layer.
Designed for AMD GPUs (ROCm); local LLM via vLLM serving Qwen2.5-7B-Instruct.

See **ROADMAP.md** for the full architecture, phase plan, model/storage
budget, risk register, and evaluation methodology.

## Quickstart (no hardware required)

```bash
pip install pydantic aiohttp pytest pytest-asyncio
AURA_LLM_MOCK=1 python -m aura.main --sim          # full pipeline, mock agents
python -m pytest tests/ -q                          # end-to-end regression
```

You should see, on a compressed synthetic session:
1. **Room monitor** at http://localhost:8766 — view-only split window:
   inference log + agent proposals (left), per-device capture (right)
2. **Moderator** surfaces the slide-6 question burst at the presenter's pause
3. **EngagementAnalyst** catches the scripted slide-4 engagement collapse
4. **PresenterCoach** issues a pacing nudge
5. **Summarizer** writes `reports/debrief_*.md`; on-slide annotations are
   detected from the screen and logged ("someone draws on slide 4")
6. Every event is recorded to `sessions/session.jsonl`

Replay any recorded session (same agents, deterministic):

```bash
AURA_LLM_MOCK=1 python -m aura.main --replay sessions/session.jsonl --speed 2
```

Generate the evaluation metrics table (detection rate, latency, tokens):

```bash
AURA_LLM_MOCK=1 python -m scripts.evaluate --runs 3   # -> reports/metrics.md
```

## PPT-B — the deck, enriched by the room

```bash
python -m scripts.generate_pptb recordings/dryrun1/events.jsonl \
    --pptx deck.pptx          # or --deck slides.pdf if no .pptx
```

Detected ink is re-drawn in place (transparent patches), each slide gets a
discreet capture panel, and a final synthesis slide answers: *what matters
for you, as a team* — including unresolved questions.

## Real LLM on the AMD GPU

```bash
# one-time: vLLM ROCm build (see requirements.txt notes)
bash scripts/serve_llm.sh                # serves Qwen2.5-7B on :8000
python -m aura.main --sim                # agents now reason with the real model
```

Token usage and median latency print at session end (`LLMClient.meter`) —
these feed the evaluation metrics table.

## Live mode (webcam + mic + participant devices)

```bash
pip install -r requirements.txt
python -m aura.main --live --camera 0 --whisper small --deck slides.pdf
# AURA observes the shared presentation: slide changes + drawn annotations.
# Voice questions are detected from the transcript. (--gateway re-enables
# the legacy participant web client if you want a typed-input fallback.)
```

## Ingesting RAW recordings (no manifest, no recorder script)

Got loose files — phone audio, IP-cam video, a Teams screen capture with the
chat pane visible? `ingest_raw` builds everything from them:

```bash
python -m scripts.ingest_raw \
    --room cam.mp4 --audio mic.m4a --screen screen.mp4 --deck slides.pdf \
    --slide-region 0.0,0.05,0.78,0.95 --chat-region 0.78,0.10,1.0,0.95 \
    --sync "audio=12.5,room=1:23,screen=3.0" \
    --out recordings/session1
```

Slide region → deck matching + drawn-annotation detection; chat region →
OCR'd Teams messages (sender + text, grouped); audio → transcript, pace,
pauses, and voice questions ('?'). Add `--vlm` to escalate to Qwen2.5-VL:
off-deck screens (demos, videos, whiteboards) get classified, and detected
ink gets transcribed + intent-tagged ("question|emphasis|correction|...").
Serve the VL model with a second vLLM instance and point AURA_VLM_URL at it
(default :8001). `--sync` takes the in-file time of ONE
shared moment (a clap, slide 1 appearing) per stream; ±1 s precision is
plenty. Then replay + PPT-B as usual.

## Dry-run demo with a cloud GPU

Sensors local, GPU on notebook.amd.com? Record once, ingest in the cloud,
replay through the live agents — see **docs/DRY_RUN_GUIDE.md**:

```bash
python -m scripts.record_session --out recordings/dryrun1     # local, one cmd
python -m scripts.ingest_recording recordings/dryrun1 --deck slides.pdf  # cloud
python -m aura.main --replay recordings/dryrun1/events.jsonl --speed 2   # demo
```

## Layout

```
aura/events.py        typed event contracts (Pydantic)
aura/bus.py           async pub/sub + JSONL session recorder/replay
aura/perception/      attention (MediaPipe), speech (faster-whisper),
                      interaction gateway (WebSocket), simulator
aura/perception/slides.py  deck index (PDF) + screen→slide matching + manual ctl
aura/fusion/state.py  RoomStateBuilder — dual temporal windows, compact LLM view
aura/agents/          LLM client (vLLM/OpenAI-compatible), 4-agent crew,
                      orchestrator with trigger policy + cooldowns
aura/actions/sink.py  console feed + Markdown debrief (per-slide engagement)
aura/actions/monitor.py  view-only split monitor (log+proposals | capture)
scripts/generate_pptb.py PPT-B generator (deck + annotations + synthesis)
scripts/evaluate.py   metrics harness vs. scripted ground truth
scripts/record_session.py  one-command multimodal recorder (ffmpeg + gateway)
scripts/ingest_recording.py  recordings → unified events.jsonl (cloud-side)
```

## Privacy

No identities, no face recognition, no raw video beyond the perception layer.
People are anonymous tracklets; only scalar attention/pose signals propagate.
