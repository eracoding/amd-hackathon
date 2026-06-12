# AURA Demo Deck — slide content + two-person room choreography

This file is the single source of truth for the dry-run: generate your
presentation from the **On slide** blocks, speak from the **Presenter says**
blocks, and have your two audience members follow the **Room** cues.

**Cast (2 people):**
- **P1 — “the engaged analyst”** (front seat): attentive baseline, reactions,
  the technical question. Takes over role A from the 3-person script.
- **P2 — “the distracted skeptic”** (side seat): leads the engagement dip,
  asks the privacy question and the final *unanswered* question. Merges
  roles B + C.
- Both join the staged dip — with two people, both heads down drives the
  meter near zero, which reads far better on camera than a half-dip.

**Deck production notes:** export to **PDF** when done (the slide tracker
matches your screen against PDF pages). Avoid per-bullet build animations on
slide 4 — the matcher holds slide context during transitions, but a slide
that assembles over 40 s blurs the “slide 4 = dense” story. Keep one visual
template; the matcher discriminates near-duplicate slides fine (validated).

Total runtime ≈ 9:00. Timestamps assume a visible shared stopwatch started
at the sync clap.

---

## Slide 1 — Why rooms should understand intent  *(0:00–0:50)*

**On slide**
- Title: *Why rooms should understand intent*
- Subtitle: *AURA — Agentic Understanding of Room Attention*
- One line, large: “Every meeting room is full of sensors. None of them are
  listening together.”
- Footer: your name · TCS France / Télécom SudParis · Track 2 (Multimodal)

**Presenter says** — after the clap + “AURA dry run one”:
“Right now, this room is watching itself. A camera reads where each of you
is looking. A microphone follows what I say. Your phones are a channel back
to me. Separately, those are gadgets. Fused — they become a participant.
This talk is a live demonstration: AURA is monitoring its own presentation.”

**Room**
- 0:00 — both P1 & P2: eyes on screen, phones face-down on the table
  (already connected to the gateway from the pre-flight check)
- HUD expectation: two person-chips appear, engagement 85–95%, meter green

---

## Slide 2 — The cost of disengaged meetings  *(0:50–1:45)*

**On slide**
- Three stat blocks:
  - “~$37B / year — estimated cost of unproductive meetings (US alone)”
  - “Feedback latency: weeks — presenters learn they lost the room *after*
    the survey, if ever”
  - “Hybrid inequity: typed questions from remote/quiet participants are the
    first thing lost”
- Bottom line: “The failure isn’t attention. It’s that nobody closes the loop
  in real time.”

**Presenter says**
“The problem isn’t that people disengage — that’s human. The problem is the
presenter finds out three weeks later in a survey, and the quiet person’s
typed question dies in a chat sidebar. Feedback latency is the disease;
everything you’ll see today is latency reduction.”

**Room**
- **1:10 — P1: send 👍** (first interaction lands on the recording)
- P2: attentive, slightly skeptical posture (arms crossed — it’s in character)

---

## Slide 3 — System overview  *(1:45–2:45)*

**On slide**
- Architecture diagram, left → right:
  *Camera → attention/gaze* · *Mic → streaming ASR* · *Devices → questions,
  comments, reactions* ⇒ **Event bus** ⇒ **Fusion: RoomState** (10 s tactical
  / 60 s strategic windows) ⇒ **4 agents** ⇒ *HUD nudges · surfaced questions
  · debrief*
- Side note: “Perception runs local on CPU. Only ~3 KB of fused state ever
  reaches the LLM (Qwen2.5-7B on AMD MI300X via vLLM/ROCm).”

**Presenter says**
“Three senses. The camera gives per-person attention — no identities, just
anonymous tracklets. The mic gives a live transcript and my speaking pace.
Your phones give intent in text form. Everything lands on one event bus,
gets fused into a compact room state, and a crew of four agents reasons over
it on an AMD GPU. Note what does *not* travel: video. Ever.”

**Room**
- Both attentive; this is the calm before the staged dip
- HUD expectation: slide chip shows the real title + content (screen→PDF
  matching is now visibly working)

---

## Slide 4 — Fusion deep-dive  *(2:45–4:20 — THE STAGED DIP)*

**On slide** — *intentionally overloaded; the density is the demo*
- Title: *Fusion internals: from pixels to a 700-token room state*
- Six dense bullets (keep them long; this slide is supposed to hurt):
  - “Per-tracklet attention: 6-pt PnP head pose → yaw/pitch cone score →
    EMA(α=0.3); IoU-based tracklet association, 3 s expiry”
  - “Dual ring buffers per modality: 10 s tactical, 60 s strategic windows”
  - “Engagement index: attendance-weighted mean of per-person tactical means”
  - “Slope estimator: endpoint delta over 30 s of non-empty snapshots,
    alarm < −0.15/min, alarm cooldown 10× orchestrator tick”
  - “ASR: faster-whisper int8, VAD-gated 5 s chunks; words/min as a pace
    signal; ≥3 s silence ⇒ PauseDetected”
  - “Serialization budget: RoomState ≤ ~700 tokens — token efficiency is a
    scored metric, not an afterthought”

**Presenter says** — *direction: speed up to ~180 wpm, monotone, no pauses,
read the bullets almost verbatim; do NOT self-correct until the nudge:*
“So diving into the fusion internals we maintain per-tracklet attention via
six-point PnP head pose into a yaw-pitch cone score smoothed with an
exponential moving average alpha zero point three with IoU association and
three-second expiry, dual ring buffers per modality ten-second tactical
sixty-second strategic, attendance-weighted engagement, endpoint slope over
thirty seconds alarming below minus zero point one five per minute…” *(keep
going in this register)*

**Room — the choreography that fires the Analyst & Coach:**
- **3:05 — P2: phone up from the table, head DOWN, scroll naturally**
- **3:20 — P1: gives up too — head DOWN at phone**
  *(staggering 15 s apart makes the meter visibly slide, not step)*
- **~3:40–3:55 — AURA fires:** EngagementAnalyst “engagement_drop — content
  density and pace” → PresenterCoach nudge. *(With both of two heads down,
  engagement falls ~0.9 → ~0.15; slope ≈ −1.5/min — far past threshold.
  Even one of you would suffice at −0.6/min, but two is theater.)*
- **3:50 — Presenter reacts ON CAMERA:** glance at HUD, break register:
  “…and the room just told me I lost you. Fair. One sentence: two time
  windows, one engagement number — that is all the agents ever see.”
  Then slow to normal pace.
- **4:00 — P1: head UP, back to screen**
- **4:10 — P2: head UP, back to screen**
- HUD expectation: both chips red during the dip; meter recovers, slope ↗ —
  *the recovery is the money shot of the whole video*

---

## Slide 5 — The agentic layer  *(4:20–5:15)*

**On slide**
- Four agent cards:
  - **EngagementAnalyst** — diagnoses attention dynamics + probable cause
  - **Moderator** — holds typed questions; surfaces them only at pauses or
    slide transitions, deduplicated
  - **PresenterCoach** — one actionable nudge, 90 s cooldown, ≤4 actions/min
  - **Summarizer** — slide-aligned debrief at session end
- Footer: “Event-driven triggers, structured-JSON contracts, shared
  blackboard — no chat-history sprawl.”

**Presenter says**
“Four agents, four trigger policies, one shared room state. You just watched
two of them act: the Analyst diagnosed *why* I lost you — pace and density,
both real signals — and the Coach told me what to do about it, exactly once.
The cooldowns aren’t a limitation; an agent that nags is an agent you mute.”

**Room**
- **4:30 — P1: send 👍** (reward the recovery — it reads as genuine)
- **4:50 — P1: type question:** *“How do the agents avoid spamming the
  presenter?”* — it will sit PENDING in the HUD inbox (no pause yet — point
  this out in the final video narration)

---

## Slide 6 — Privacy by design  *(5:15–6:55 — THE PAUSE BEAT)*

**On slide**
- Three lines, large:
  - “No identities. People are anonymous tracklets: person_0, person_1.”
  - “No raw video beyond the perception layer — only scalar signals
    (attention, pose) propagate.”
  - “LLM sees ≤ 700 tokens of fused state. Never pixels.”
- Small footer: “GDPR-aligned by architecture, not by policy document.”

**Presenter says**
“The skeptical question in every deployment meeting is the right one: where
does the video go? Answer: nowhere. Detection happens on-device; what leaves
is a number per person per second. The language model never sees a pixel.”

**Room**
- **5:40 — P2: type question:** *“Where exactly is the video stored?”*
  (inbox now shows 2 pending)
- **6:05 — Presenter: “Let me stop here for a moment.” FULL SILENCE, 5 s,
  eye contact with the room.** *(≥3 s silence ⇒ PauseDetected ⇒ Moderator
  surfaces BOTH questions, clustered.)*
- **6:15 — Presenter answers both, reading them off the HUD:**
  - to P1: “Spam control: per-agent cooldowns plus a global four-actions-
    per-minute budget — the orchestrator drops the rest.”
  - to P2: “Stored? The video file stays on the recording machine. In live
    mode it isn’t stored at all — frames are processed and discarded.”
- **6:25 — P2: nod visibly** (it’s on camera — the loop closes on film)
- Safety net: if the pause somehow doesn’t register, the slide-7 transition
  also triggers the Moderator. Don’t re-pause awkwardly; just advance.

---

## Slide 7 — Evaluation  *(6:55–7:50 — INTENTIONAL CALM)*

**On slide**
- Metrics table (your real numbers from `reports/metrics.md`):

| Metric | Result |
|---|---|
| Engagement-dip detection | 3/3 sessions |
| Question surfacing | 3/3 sessions |
| False-positive alerts | 0 |
| Median dip-detection latency | ~2.6 s |
| Median question-surfacing latency | ~0.4 s |
| Slide matching under capture distortion | 5/5 @ 0.99 similarity |
| Test suite | 16/16 passing |
- Footer: “Deterministic evaluation harness: scripted anomalies, seeded
  runs, reproducible — `scripts/evaluate.py`.”

**Presenter says**
“Numbers, not vibes. The harness replays scripted sessions with known
anomalies and scores detection, latency, and false positives. And notice
what’s happening in the room *right now*: you’re attentive, I’m pacing well —
and AURA is silent. Zero false positives is the harder half of detection.”

**Room**
- Both: model attentiveness — this slide *proves* the quiet case
- HUD expectation: green, steady, **nothing fires** (that’s the point)

---

## Slide 8 — Roadmap & ask  *(7:50–9:00 — THE UNRESOLVED QUESTION)*

**On slide**
- Now → Next → Later:
  - **Now:** single room, replayable sessions, Qwen2.5-7B on MI300X (ROCm)
  - **Next:** RTSP multi-camera, VLM slide understanding (Qwen2.5-VL, same
    vLLM server) for demos/whiteboards, HUD slide controls
  - **Later:** edge deployment (Jetson-class), per-org fine-tuned coach
    (PEFT), fleet analytics across rooms
- Closing line: “Rooms that close the feedback loop — in seconds, not
  surveys.”

**Presenter says**
“Everything you saw runs today on one AMD GPU. Next: more cameras, a vision-
language model for the content hashing can’t read, and pushing perception to
the edge. The ask: one pilot room. Thank you.”

**Room**
- **8:30 — P2: type question:** *“Can this run fully on a Jetson?”* —
  **presenter must NOT see or answer it.** It exists so the debrief’s
  *Unresolved questions* section is non-empty: the system remembers what the
  human missed.
- **8:50 — Presenter: “Thank you.” Stop talking. Hold 5 s of silence, then
  stop the recording** (Ctrl-C; recorders finalize cleanly)

---

## Condensed cue cards (print, cut, hand out)

**P1 — “engaged analyst”**
```
1:10  send 👍
3:20  head DOWN at phone (stay down)
4:00  head UP, attentive for the rest
4:30  send 👍
4:50  type: "How do the agents avoid spamming the presenter?"
```

**P2 — “distracted skeptic”**
```
3:05  head DOWN at phone (stay down)
4:10  head UP
5:40  type: "Where exactly is the video stored?"
6:25  nod when answered
8:30  type: "Can this run fully on a Jetson?"  (will NOT be answered — intended)
```

**Presenter beat sheet**
```
0:00  clap + "AURA dry run one", calm open
2:45  slide 4: SPEED UP (~180 wpm), monotone, don't self-correct
3:50  react to nudge on camera, recap in one sentence, slow down
6:05  "Let me stop here" → 5 s full silence → answer both from HUD
8:50  "Thank you" → 5 s silence → stop recording
```

## Two-person trigger notes (what changed vs. the 3-person script)

- With 2 people, **one** disengaged person already moves engagement
  0.9 → ~0.55 (slope ≈ −0.6/min) — past the −0.15 alarm. Both heads down is
  for visual drama, not necessity. Practical upside: timing tolerance is
  loose; if P1 is 10 s late, the demo still fires.
- Recovery must be staggered too (4:00 / 4:10) so the slope arrow flips to
  ↗ rising smoothly rather than jumping.
- Reactions and questions are all on P1/P2 phones under different gateway
  IDs — the HUD inbox and debrief don’t care that two humans played three
  roles.
