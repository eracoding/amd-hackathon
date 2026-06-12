# presentation.md — AURA hackathon submission deck (5 slides)

Follows the required template field-by-field. Each slide has **On slide**
(generate the deck from this) and **Speaker notes**. The dry-run choreography
remapped onto these 5 slides is at the bottom — this same deck is what you
present in the recorded demo.

Fill `{...}` placeholders before generating. Numbers marked ⚙ are measured
in this repo today (mock LLM backend where noted); re-run
`scripts/evaluate.py` against live vLLM on the MI300X and replace.

---

## Slide 1 — Basic Information

**On slide**

- **Team name:** {AURA / your team name}
- **Team members & roles:**
  - Ulugbek “Era” Shernazarov — research, system design & full implementation,
    demo engineering, presentation
  - {Teammate name} — scenario & interaction design, demo direction,
    business case
- **Use case:** *AURA — Agentic Understanding of Room Attention.*
  A multimodal meeting companion for co-located collaborative sessions.
- **Short description:** One presentation (PPT-A) is dispatched to every
  participant’s device. AURA fuses what the room *does* — gaze attention
  (camera), presenter speech (mic), typed/drawn inputs (laptops & tablet) —
  into a live room state. A crew of LLM agents reasons over it, coaches the
  presenter in real time (HMI-1: the capture view), and produces **PPT-B**:
  the original deck enriched with the team’s aggregated reactions —
  *“this is what matters for you, as a team.”*
- **Track:** Track 2 — Multimodal (with a Track-1-style multi-agent
  orchestration layer)

**Speaker notes**
“This room is watching itself right now — the demo you’ll see is AURA
monitoring this very presentation. One deck, four devices, four input
modalities, one companion.”

---

## Slide 2 — Problem & Context

**On slide**

- **Problem statement:** There is no tool today that lets ten people work on
  the same document, in the same room, with a companion that watches every
  input — attention, speech, typing, drawing — and tells the team:
  *this is what is important for you.* Presenters fly blind; typed questions
  die in side channels; group convergence requires everyone to talk, and
  most people can’t talk and listen at once.
- **Target users / stakeholders:** presenters & facilitators; enterprise
  L&D and corporate training; design-review and workshop teams; hybrid
  teams where quiet/remote participants lose their voice; (research
  stakeholders: TCS France × CEA-List, AI Kinesis).
- **Why it matters:**
  - Unproductive meetings cost an estimated tens of billions of $/year
  - **Feedback latency is the disease:** presenters learn they lost the
    room weeks later, via surveys — AURA closes the loop in seconds
  - Inputs are perishable: unaddressed questions and annotations vanish
    when the meeting ends — no institutional memory
- **Mapped hackathon challenge:** Track 2 — Multimodal (vision + speech +
  interaction pipelines for decision-making); incorporates Track 1 traits
  (multiple cooperating agents, token/latency efficiency as scored metrics).

**Speaker notes**
“The failure isn’t attention — that’s human. The failure is that nobody
closes the loop in real time, and nothing aggregates what the group is
silently telling you.”

---

## Slide 3 — Solution Overview

**On slide**

- **Workflow** (diagram; left → right):

```
PPT-A ──dispatch──► A1 A2 A3 A4 (3 laptops + tablet)
                     │   │   │   │
   camera ► gaze/attention (per person, anonymous)
   mic    ► streaming ASR + pace + pause detection      ─► EVENT BUS
   devices► typed comments/questions · reactions ·          │
            drawn annotations (slide-anchored)               ▼
                                              FUSION: RoomState
                                     (10 s tactical / 60 s strategic,
                                      ≤700 tokens by design)
                                                             │
                              ┌─ 4 AGENTS ──────────────────┴──┐
                              │ EngagementAnalyst · Moderator   │
                              │ PresenterCoach   · Summarizer   │
                              └───────┬──────────────────┬──────┘
                                HMI-1: live capture   PPT-B: deck + aggregated
                                view (presenter HUD)  team results & synthesis
```

- **AI approach:** multimodal perception (vision + speech + interaction)
  fused into a compact state; event-driven **multi-agent** reasoning with
  structured-JSON contracts; screen→slide grounding via perceptual matching
  (VLM escalation designed for non-slide content).
- **Key technologies:** AMD MI300X + ROCm · vLLM serving Qwen2.5-7B-Instruct
  · MediaPipe face/pose (CPU) · faster-whisper (CTranslate2 int8) · asyncio
  event bus + Pydantic event contracts · aiohttp (participant gateway + HUD)
  · pypdfium2 deck matching.
- **Built during the hackathon:** the full working pipeline — perception
  modules, fusion, 4-agent orchestrator with trigger policies & cooldowns,
  live presenter HUD, participant web client, one-command session recorder,
  cloud ingestion pipeline (recordings → unified event stream), replay
  engine, deterministic evaluation harness, 16-test suite.

**Speaker notes**
“Note what never crosses the network: pixels. Perception is local CPU; the
GPU sees only ~3 KB of fused state per agent call. That’s the privacy story
and the latency story at once.”

---

## Slide 4 — Model Insights

**On slide**

- **Models used:**
  - Qwen2.5-7B-Instruct, bf16, served by vLLM on ROCm (agent reasoning)
  - faster-whisper *small/medium*, int8 CPU (ASR — keeps GPU free)
  - MediaPipe FaceMesh + PnP head pose (attention; CPU, no CUDA deps)
- **Datasets / training:** no fine-tuning in scope — prompt-engineered
  agents with structured-output contracts (fine-tuned coach = future work).
- **Tokens (example scenarios):** room state ≤700 tokens by design ⚙
  - Scenario A — engagement dip (Analyst + Coach): 2 calls,
    ≈ {2.3K} prompt + {0.3K} completion tokens
  - Scenario B — full 9-min session (all agents + debrief):
    ≈ {8–12} calls, ≈ {10–15K} total tokens
- **End-to-end latency** (anomaly onset → presenter sees action):
  - dip detection: **2.6 s median** ⚙ · question surfacing: **0.4 s median** ⚙
    (pipeline, mock backend) + {0.5–1.5 s} LLM inference on MI300X
    → end-to-end target **< 4 s** {measured: …}
  - Detection quality: 3/3 dips, 3/3 question bursts, **0 false positives** ⚙;
    slide matching 5/5 @ 0.99 similarity under capture distortion ⚙
- **GPU usage / right-sizing:** MI300X (192 GB) available; actual footprint
  ≈ {16 GB} weights + {4–8 GB} KV cache → **the workload fits a 24 GB-class
  GPU**; perception deliberately on CPU. Headroom on MI300X funds the
  roadmap (Qwen2.5-VL screen understanding on the same server).

**Speaker notes**
“Right-sizing is a result, not an accident: token-budgeted fusion means one
mid-size LLM serves the whole room, and the 192 GB GPU is roadmap headroom,
not a requirement.”

---

## Slide 5 — Impact & Demo Summary

**On slide**

- **Expected impact / value:**
  - Feedback latency: weeks (surveys) → **seconds**
  - Zero lost inputs: every question is surfaced live or logged as
    *unresolved* in PPT-B — institutional memory by default
  - Per-slide engagement analytics → content improves between sessions
  - Privacy-preserving by architecture: anonymous tracklets, no pixels to
    the LLM — deployable where cameras are sensitive
- **Key differentiators / innovation:**
  - A **companion, not a dashboard**: agents act (nudge, surface, propose) —
    they don’t just chart
  - Calm-by-design agentic UX: per-agent trigger policies, cooldowns,
    action budget — *zero false positives is the harder half of detection*
  - PPT-B: the meeting’s output is an enriched version of its own input
    document — group convergence without forcing everyone to talk
  - One generalizable “unified attention” layer: same architecture serves a
    warehouse operator flooded by subsystem messages (research agenda)
- **Demo flow — what to notice:**
  1. Staged engagement collapse → Analyst diagnoses *cause* (pace + density)
     → Coach nudges → presenter recovers **on camera**
  2. Questions typed mid-talk sit pending → surfaced exactly at the
     presenter’s pause → answered from the HUD
  3. A calm slide where **nothing fires** — correct silence
  4. Debrief/PPT-B: weakest slide named, one question flagged *unresolved*
- **Future extensions:** drawn-ink aggregation into PPT-B (PowerPoint
  co-editing path) · ScreenAgent-style VLM screen perception for live
  demos/whiteboards · edge deployment (Jetson-class) · PEFT-tuned
  organization-specific coach.

**Speaker notes**
“Everything you saw ran on one AMD GPU against this very deck. The demo is
the product.”

---

---

# Appendix A — Dry-run choreography remapped to THIS 5-slide deck

Runtime ≈ 7:00. Same cast as `DEMO_DECK_AND_CHOREOGRAPHY.md`
(P1 “engaged analyst”, P2 “distracted skeptic”), same trigger math.
Pre-flight checklist and recording commands unchanged (`docs/DRY_RUN_GUIDE.md`).

| Time | Slide | Presenter | Room | AURA beat |
|---|---|---|---|---|
| 0:00 | 1 — Basic info | Clap + “AURA dry run”, calm intro | Both attentive; **0:30 P1 👍** | Chips appear, meter green |
| 0:45 | 2 — Problem | Normal pace, the “no tool in the world” line | Attentive | Steady |
| 2:00 | 3 — Solution | **Overload deliberately**: read the architecture fast (~180 wpm), monotone | **2:30 P2 ↓phone · 2:45 P1 ↓phone** (hold ~50 s) | **~3:05 Analyst: engagement_drop (pace/density) → Coach nudge** |
| 3:15 | 3 | React on camera: “the room just told me I lost you — one sentence: …”, slow down | **3:25 P1 ↑ · 3:35 P2 ↑** | Meter recovers ↗ — money shot |
| 3:45 | 3→4 | Transition | **3:45 P1 types:** “How do agents avoid spamming the presenter?” | Question pending in inbox |
| 4:00 | 4 — Model insights | Walk the numbers, relaxed | **4:20 P2 types:** “Where exactly is the video stored?” | Inbox: 2 pending |
| 5:00 | 4 | **“Let me pause here.” 5 s full silence** → answer both from the HUD | P2 nods on camera | **PauseDetected → Moderator surfaces both, clustered** |
| 5:30 | 5 — Impact & demo | Impact + differentiators; calm delivery | Attentive — proves the quiet case | **Nothing fires** (say so in the final cut) |
| 6:30 | 5 | Future work | **6:30 P2 types:** “Can this run fully on a Jetson?” — NOT answered | Stays pending |
| 6:50 | 5 | “Thank you.” 5 s silence → stop recording | — | Debrief/PPT-B: slide 3 lowest engagement, 1 unresolved question |

Note the elegant accident: the *dense* slide is now Slide 3 (Solution
Overview) — so the debrief will name the architecture slide as the weakest,
which you can show the jury as proof the system critiques even its own
submission deck.

# Appendix B — submission checklist

- [ ] Code: `aura.zip` repo (16/16 tests)
- [ ] This deck generated as PPT/PDF from Slides 1–5 above (export PDF for
      the slide tracker before recording)
- [ ] Demo recording: dry run per Appendix A → ingest → replay screen-capture
      with HUD visible (~3-min final cut per `docs/DEMO_SCRIPT.md` outline)
- [ ] Replace every `{...}` and re-measure every ⚙ against live vLLM
      (`scripts/evaluate.py`) before submitting
