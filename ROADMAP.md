# AURA — Agentic Understanding of Room Attention
## Implementation Roadmap & Technical Design Document

**Track:** 2 — Multimodal (with Track 1 agentic orchestration as the reasoning layer)
**Target hardware:** AMD Instinct-class GPU (ROCm), 192 GB VRAM, 25 GB persistent storage
**Author:** Ulugbek Shernazarov

---

## 1. Problem Definition & Relevance (rubric: 10%)

Enterprise presentations, trainings, and design reviews are one-directional by default.
Presenters fly blind: they cannot tell when the room disengages, which slide triggered
silent confusion, or which typed questions on participants' devices were never addressed.
Post-meeting, institutional knowledge evaporates — no structured record links *what was
said* to *how the room reacted* to *what participants asked*.

**AURA** is an intelligent multimodal room environment that perceives, fuses, and
*reasons about* collective intent in real time:

- **Camera** → per-person attention/gaze estimation (who is looking at the screen,
  who has disengaged, head-pose dynamics)
- **Microphone** → streaming ASR of the presenter, aligned to slide timeline
- **Tablets/laptops** → interaction events (comments, questions, reactions pinned
  to slides)

A multi-agent reasoning layer consumes the fused room state and **acts**: it flags
engagement collapse, surfaces unanswered questions at natural pauses, coaches the
presenter on pacing, and emits a structured, slide-aligned session debrief.

**Enterprise relevance:** corporate training ROI measurement, sales-pitch coaching,
all-hands meeting analytics, accessibility (live captions + question routing), and
hybrid-meeting equity (remote participants' typed questions get equal weight). The
same architecture generalizes to industrial control rooms and classrooms.

**Privacy-by-design:** no face recognition, no identity persistence. Participants are
anonymous tracklets (`person_0`, `person_1`); only derived scalar signals (attention
score, head pose) leave the perception layer. Raw video never reaches the LLM.

---

## 2. System Architecture

```
┌─────────────────────────  PERCEPTION LAYER  ─────────────────────────┐
│  Camera ──► AttentionTracker (MediaPipe FaceLandmarker, per-person   │
│             gaze proxy + head pose, 15–30 fps, CPU/GPU)              │
│  Mic ─────► SpeechPipeline (faster-whisper, streaming chunks,        │
│             slide-timestamped transcript)                            │
│  Devices ─► InteractionGateway (WebSocket/HTTP: comments, questions, │
│             reactions, slide anchor)                                 │
│  [SimulatedSession: deterministic synthetic generator for all three  │
│   streams — enables development & demo without hardware]            │
└───────────────┬──────────────────────────────────────────────────────┘
                │  typed events (Pydantic schemas)
                ▼
┌─────────────────────────  EVENT BUS  ────────────────────────────────┐
│  Async in-process pub/sub (asyncio), topic-based, with session       │
│  recorder (JSONL) for replay & evaluation                            │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌─────────────────────────  FUSION LAYER  ─────────────────────────────┐
│  RoomStateBuilder: sliding temporal windows (10 s / 60 s) →          │
│  RoomState snapshot: per-person attention trajectories, room-level   │
│  engagement index, transcript segment, pending interactions,         │
│  slide context, derived deltas (engagement slope, silence gaps)      │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼  compact JSON state (~1–2 KB) — token-efficient by design
┌─────────────────────────  AGENTIC LAYER  ────────────────────────────┐
│  Orchestrator (event-driven trigger policy, not naive polling):      │
│    • EngagementAnalyst  — interprets attention dynamics, names the   │
│      probable cause (pace, density, topic shift)                     │
│    • Moderator          — clusters/dedups participant questions,     │
│      decides WHEN to surface them (detects pauses/transitions)       │
│    • PresenterCoach     — actionable nudges (slow down, recap,       │
│      invite questions), rate-limited to avoid noise                  │
│    • Summarizer         — end-of-session slide-aligned debrief:      │
│      per-slide engagement, unresolved questions, action items        │
│  LLM backend: local vLLM (ROCm) serving Qwen2.5-7B-Instruct via      │
│  OpenAI-compatible API. Agents use structured-JSON tool contracts.   │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌─────────────────────────  ACTION LAYER  ─────────────────────────────┐
│  ActionSink: presenter HUD feed (console/web), participant           │
│  acknowledgements, persisted debrief report (Markdown)               │
└──────────────────────────────────────────────────────────────────────┘
```

**Why agentic (not a single prompt):** each agent has a distinct trigger condition,
context window, and action authority. The Moderator must reason about *timing*
(interrupt vs. wait); the Analyst reasons about *causality* in signal dynamics; the
Coach must be *rate-limited and prioritized*. Separating them keeps prompts small
(token efficiency — a Track 1 judging criterion), makes behavior auditable, and lets
agents communicate through the shared RoomState rather than long chat histories.

---

## 3. Technology Stack (AMD/ROCm-validated)

| Component | Choice | Rationale & ROCm notes |
|---|---|---|
| LLM serving | **vLLM (ROCm build) + Qwen2.5-7B-Instruct** | OpenAI-compatible server; bf16 weights ≈ 15 GB → fits 25 GB storage. 192 GB VRAM is ample; storage, not VRAM, is the binding constraint. Fallback: Qwen2.5-3B (~6 GB) if disk pressure. |
| ASR | **faster-whisper (small/medium)** | CTranslate2 runs CPU int8 well — keeps GPU free for the LLM; `small` ≈ 460 MB. Alternative: whisper via PyTorch-ROCm on GPU. |
| Vision | **MediaPipe Face Landmarker** | CPU-real-time, no CUDA dependency at all — sidesteps ROCm vision-op gaps entirely. Head pose via solvePnP (OpenCV) + iris landmarks as gaze proxy. |
| Agent framework | **Custom lightweight orchestrator (asyncio)** | Avoids heavyweight frameworks; full control of trigger policy and token budget; trivially swappable to LangGraph later. |
| Schemas/events | Pydantic v2 | Typed event contracts; JSONL session recording for replay/eval. |
| Deep learning rt | PyTorch ROCm wheels (`--index-url https://download.pytorch.org/whl/rocm6.x`) | Only needed if GPU ASR/vision is enabled. |

**Storage budget (25 GB):** Qwen2.5-7B bf16 ≈ 15.2 GB + whisper-small 0.5 GB +
MediaPipe task file 0.03 GB + env/wheels ≈ 4 GB → **~20 GB, within budget.**
If tight: GPTQ-Int4 Qwen2.5-14B (~10 GB) gives better quality/GB, but quantized-kernel
support on ROCm should be validated first (Phase 0 task).

---

## 4. Implementation Roadmap

### Phase 0 — Environment Validation (Day 1)
- [ ] Verify ROCm visibility: `rocm-smi`, `torch.cuda.is_available()` (ROCm masquerades as cuda device)
- [ ] Install vLLM ROCm build; smoke-test with Qwen2.5-7B-Instruct; measure tokens/s
- [ ] Validate faster-whisper CPU int8 latency on 10 s audio chunks (< 2 s target)
- [ ] Validate MediaPipe FaceLandmarker on sample frames
- **Exit criteria:** all three modalities produce output on target hardware

### Phase 1 — Event Backbone & Simulation (Days 2–3)
- [ ] Pydantic event schemas: `AttentionEvent`, `TranscriptSegment`, `InteractionEvent`, `SlideChange`
- [ ] Async pub/sub `EventBus` with topic subscription + JSONL `SessionRecorder`
- [ ] **`SimulatedSession`**: scripted 10-minute synthetic presentation (engagement dip
      at slide 4, question burst at slide 6) — this is the regression/demo backbone
- **Exit criteria:** simulated session replays deterministically end-to-end

### Phase 2 — Perception Modules (Days 4–7)
- [ ] `AttentionTracker`: multi-face landmark detection → head pose (yaw/pitch via PnP)
      → screen-attention score ∈ [0,1] per anonymous tracklet; EMA smoothing
- [ ] `SpeechPipeline`: chunked streaming ASR; VAD gating; emits timestamped segments;
      pause detection (≥ 3 s silence → `PauseDetected` event for the Moderator)
- [ ] `InteractionGateway`: WebSocket endpoint for tablet/laptop clients; minimal web
      client (HTML) for the demo audience to post questions/reactions tied to slide #
- **Exit criteria:** live webcam + mic produce real events on the bus

### Phase 3 — Fusion (Days 8–9)
- [ ] `RoomStateBuilder`: ring buffers per modality; 10 s tactical + 60 s strategic
      windows; engagement index = attendance-weighted mean attention; slope detection
- [ ] Compact JSON serialization of RoomState (budget: ≤ 700 tokens) — token efficiency
      as an explicit, measured design goal
- **Exit criteria:** RoomState snapshots are correct on the simulated session (unit tests)

### Phase 4 — Agentic Layer (Days 10–14)
- [ ] OpenAI-compatible LLM client (points at local vLLM; env-switchable)
- [ ] Trigger policy: Analyst on engagement slope < −0.15/min OR every 60 s;
      Moderator on `PauseDetected` ∧ pending questions; Coach on Analyst findings
      (cooldown 90 s); Summarizer on `SessionEnd`
- [ ] Structured outputs: every agent returns validated JSON (finding, confidence,
      recommended action) — parse-or-retry loop, max 2 retries
- [ ] Inter-agent communication via shared blackboard (RoomState annotations)
- **Exit criteria:** on the simulated session, agents fire at the scripted dip and
      question burst, with sensible, validated JSON actions

### Phase 5 — Actions, Demo & Evaluation (Days 15–18)
- [ ] Presenter HUD (rich console / simple web view) streaming agent nudges
- [ ] End-of-session Markdown debrief (per-slide engagement chart + unresolved Qs)
- [ ] **Quantitative evaluation** (feeds rubric criterion 2):
      - Attention tracker: accuracy vs. hand-labeled 5-min clip (target ≥ 85% on
        attending/not-attending binary)
      - ASR: WER on presenter audio sample
      - Agent layer: trigger precision/recall on scripted simulated anomalies;
        median end-to-end latency (event → action); tokens consumed per session
- [ ] Record demo: live webcam segment + simulated full-session replay
- **Exit criteria:** metrics table populated; 5-min demo script rehearsed

### Phase 6 — Stretch (if time allows)
- IP-camera ingestion (RTSP) for true room coverage; speaker diarization;
  vision-language model (Qwen2.5-VL-3B) for slide-content understanding so agents
  can reason about *what is on screen*, not just slide numbers; per-participant
  confusion classifier distilled from agent labels (mini fine-tuning story → touches
  Track 3 narrative in Future Work).

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| vLLM ROCm install friction | Medium | Phase 0 gate; fallback to llama.cpp (HIP build) or transformers + PyTorch-ROCm naive serving |
| 25 GB storage overflow | Medium | Model budget table above; HF cache symlinked & pruned; use 3B fallback |
| Multi-face tracking jitter | Medium | EMA smoothing + IoU-based tracklet association; demo with ≤ 4 people |
| Agent over-triggering (noisy HUD) | High | Cooldowns, confidence thresholds, max-actions-per-minute cap |
| Live demo hardware failure | Medium | Simulated session replay is a first-class demo path, not a hack |
| LLM JSON non-compliance | Medium | Pydantic validation + bounded retry; few-shot in system prompts |

---

## 6. Mapping to Evaluation Criteria

- **Technical implementation (40%)** — three real modalities, fused state, multi-agent
  reasoning with structured outputs, all on ROCm; measured accuracy/latency/token metrics.
- **Learnings & future work (20%)** — edge deployment path, privacy-preserving sensing,
  scaling from one room to a fleet, distillation/fine-tuning roadmap.
- **Innovation (15%)** — a custom use case nobody else submits; "the room as an agent."
- **Problem & relevance (10%)** — enterprise training/meetings, quantified pain.
- **Demo (15%)** — live perception + deterministic replay; narrative arc: *perceive →
  fuse → reason → act → debrief*.

---

## 7. Repository Layout

```
aura/
├── ROADMAP.md                  # this document
├── README.md                   # quickstart
├── requirements.txt            # ROCm-aware dependencies
├── configs/default.yaml        # all tunables
├── aura/
│   ├── events.py               # Pydantic event schemas
│   ├── bus.py                  # async pub/sub + session recorder
│   ├── perception/
│   │   ├── attention.py        # MediaPipe attention tracker
│   │   ├── speech.py           # faster-whisper streaming ASR
│   │   ├── interaction.py      # device interaction gateway
│   │   └── simulate.py         # synthetic session generator
│   ├── fusion/state.py         # RoomStateBuilder
│   ├── agents/
│   │   ├── llm.py              # OpenAI-compatible client (vLLM)
│   │   ├── base.py             # structured-output agent base class
│   │   ├── crew.py             # the four agents
│   │   └── orchestrator.py     # trigger policy & dispatch
│   ├── actions/sink.py         # HUD + debrief writer
│   └── main.py                 # entrypoints: --sim | --live
├── scripts/serve_llm.sh        # vLLM ROCm launch
└── tests/test_pipeline.py
```
