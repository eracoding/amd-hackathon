# roadmap_design.md — aligning AURA with the PPT-A/PPT-B companion vision

Source: the design discussion (transcript.txt) + the supervisor's slide
structure (scenario → demo technicals → system architecture → unified
attention). This document maps that vision onto the current implementation,
names the gaps, and gives a phased design to close them — Phase A is sized
for the Monday demo and the hackathon submission.

---

## 1. The vision, distilled from the transcript

**The scenario (slides 1–2 of the supervisor's structure):**

```
            PPT-A  (one deck, dispatched to every device)
              │
   ┌──────────┼──────────┬──────────┐
   A1         A2         A3         A4          + big screen (presentation)
 laptop     laptop     laptop     tablet
 keyboard   keyboard   voice      pencil/ink     ← each device = one input
 typing     typing     (mic)      handwriting       modality
   │          │          │          │
   └──────────┴────┬─────┴──────────┘
                   ▼
        ┌── the COMPANION ─────────────────────────────┐
        │ watches two things:                          │
        │  (1) everyone's INPUT  (typed, drawn, voice) │
        │  (2) everyone's ATTENTION (perception/gaze)  │
        └──────┬───────────────────────────┬───────────┘
               ▼                           ▼
        HMI-1: "the capture"        PPT-B = PPT-A + aggregated results
        shows WHAT IS HAPPENING     a PROPOSED synthesis — "an area where
        (people, inputs, signals)   people can meet" without everyone
                                    having to talk
```

**The storyline (verbatim spirit):** *ten people, one document, and a
companion able to watch what everyone puts in and say: "if I summarize what
you're saying, this is what we have in front of us — this is what is
important for you, as a team."* The companion uses **inputs + perception**.

**The architectural thesis (last slide of the supervisor's structure):**
this is not about specific sensors. It is a **unified attention layer** —
one layer that catches inputs from many heterogeneous subsystems and manages
them into a single output. The warehouse-operator example (subsystems,
managers, customers all sending messages; a companion sorts them into one
stream) is the *same architecture*, plug-and-play. That is the research
agenda; the meeting room is the first instantiation.

**Vocabulary correction adopted:** avoid "real time" (it "doesn't mean
anything"); say *"shows what is happening"* (HMI-1) and *"proposed
synthesis"* (PPT-B).

---

## 2. Gap analysis: vision ⇄ current implementation

| Vision element | Current AURA status | Gap |
|---|---|---|
| PPT-A dispatched to devices (A1…An) | ✗ — participants see only slide *number* in the web client | Devices must display the current slide itself |
| Typing input (keyboard) | ✓ — gateway: questions/comments/reactions, slide-anchored | — |
| Voice input (participant) | ◐ — presenter ASR only | Participant voice = later phase (diarization) |
| Pencil/handwriting on slides (tablet) | ✗ | **The headline gap.** Two paths designed below (D1) |
| Eye attention (camera) | ✓ — per-person anonymous attention, validated | — |
| HMI-1 "the capture" | ✓ — presenter HUD (meter, chips, inbox, feed) | Rename/extend: add annotation activity to the view |
| PPT-B = A + aggregated results | ◐ — Markdown debrief with per-slide stats & unresolved questions | Debrief must become an **artifact in the deck's own format** (D3) |
| Screen perception | ◐ — pHash/NCC matching vs PDF (which slide is showing) | Vision wants *understanding*, ScreenAgent-style (D2) |
| Unified attention layer | ✓ implicitly — the event bus + fusion already is it | Name it, document the warehouse mapping (D4) |

The deepest reframe is not a feature: it is that **the Summarizer's output
changes ontology** — from a *report about* the meeting to a *new version of
the meeting's own document*. PPT-B is the product.

---

## 3. Design decisions

### D1 — Slide display + ink annotations on participant devices

Two paths; build (a) now, keep (b) as the enterprise integration story.

**(a) Web canvas in the existing gateway — the Monday/hackathon path.**
The participant page gains: the current slide rendered as an image
(pages already rendered by `DeckIndex` via pypdfium2 — serve them as PNG),
with a transparent `<canvas>` overlay for finger/stylus drawing. A stroke
ends → send `{kind: "annotation", slide, strokes: [[x,y,t]...], color}`
over the existing WebSocket → new `AnnotationEvent` on the bus
(normalized 0–1 coordinates, so they re-project onto any rendering of the
slide). Tablet pencil, laptop trackpad, phone finger — one implementation,
zero installs. Estimated effort: ~1 day (client canvas + event type +
fusion counter + HUD “annotation activity” sparkline).

**(b) Microsoft PowerPoint co-editing — the enterprise path.**
Participants ink directly on a shared .pptx (PowerPoint's native Draw tab,
co-authoring via OneDrive/SharePoint). Ink is stored in the OOXML as InkML
(`<mc:AlternateContent>`/`<a14:contentPart>`). Post-session (or polled),
AURA parses the file, extracts ink per slide + author + timestamp, and
emits the same `AnnotationEvent`s. Pros: native tooling, pressure-accurate
ink. Cons: requires M365 + file access; live polling is clunky; InkML
parsing is fiddly. **Decision: (b) is Phase C** — it changes nothing
downstream because both paths emit the same event.

### D2 — ScreenAgent-aligned screen perception

ScreenAgent (Niu et al.) runs a VLM in an **observe → plan → act** loop over
screenshots. AURA adopts this in two stages:

**Stage 1 — observe (Phase B):** keep the NCC matcher as a *cheap gate*
(it is right 5/5 at ~0 token cost when a known slide is showing). When the
gate reports *changed-but-unmatched* for ≥3 s — a live demo, video, code
editor, whiteboard — escalate the frame to **Qwen2.5-VL** (served by the
same vLLM instance; MI300X headroom covers both models). The VLM returns a
structured `ScreenState {kind: demo|video|code|whiteboard|document, summary}`
(interface already stubbed: `describe_frame_with_vlm`). Fusion carries it in
RoomState so agents reason about off-deck content. Token control: VLM calls
only on sustained unmatched content, debounced — worst case a few calls per
session, not per second.

**Stage 2 — act (research, Phase C+):** the ScreenAgent loop closed: the
companion doesn't just *read* the screen, it *writes* PPT-B onto it —
proposing the synthesis slide at session end, navigating to the slide with
unresolved questions. This is where “companion” stops being a metaphor.

### D3 — The PPT-B generator (Summarizer upgrade)

Pipeline (python-pptx, ~1–2 days):
1. Clone PPT-A.
2. Per slide, overlay a discreet margin panel: engagement score, question
   count, and the clustered annotation/comment digests (Summarizer already
   receives per-slide stats; extend its JSON contract with
   `per_slide[].synthesis`).
3. Re-draw participant ink (from normalized strokes) as freeform shapes on
   the corresponding slides — the team literally sees their collective hand
   on the document.
4. Append one final slide: **“What matters for you, as a team”** — the
   Summarizer's cross-slide synthesis + unresolved questions + action items.
5. Output `PPT-B.pptx` next to the debrief; HUD links it at session end.

Fallback when input was PDF-only: same content as an annotated PDF
(pypdfium2 render + overlay) or the existing Markdown debrief.

### D4 — Naming the layer: Unified Attention

No new code — a documentation/positioning act with one refactor: the bus +
fusion pair is the **unified attention layer**; perception modules and
device gateways are *subsystem adapters*. The warehouse mapping that proves
plug-and-play:

| Meeting room subsystem | Warehouse operator subsystem |
|---|---|
| Camera → person attention | Telemetry → machine/zone state |
| Mic → presenter speech | Radio/intercom, voice notes |
| Typed questions | Work orders, manager messages, customer escalations |
| Ink annotations | Markups on floor plans / pick lists |
| HMI-1 capture view | Operator heads-up: “what is happening” |
| PPT-B synthesis | Shift handover digest: “what mattered, what's unresolved” |

Same RoomState shape, same agent crew, different adapters. This table goes
on the internal “system architecture → unified attention” slide and anchors
the research agenda (and Era's thesis arc since October).

---

## 4. Phased plan

**Phase A — by the Monday demo / hackathon submission (~2–3 days)**
- [ ] Gateway: serve current slide image + canvas ink → `AnnotationEvent`
- [ ] Fusion: annotation counts per slide in RoomState (+ HUD activity strip)
- [ ] PPT-B v1: cloned deck + per-slide stats panel + ink re-draw +
      synthesis slide (python-pptx)
- [ ] Re-measure `scripts/evaluate.py` against live vLLM; fill the ⚙ numbers
      in presentation.md
- Demo addendum: during the dry run, P1/P2 each draw one circle/underline on
  the dense slide — PPT-B then shows the team's ink exactly where attention
  collapsed. One beat, two features.

**Phase B — hackathon polish (+2–3 days)**
- [ ] Qwen2.5-VL escalation behind the NCC gate (`ScreenState` events)
- [ ] Annotation clustering in the Summarizer (overlapping ink on the same
      region = the team converging — that *is* “an area where people meet”)

**Phase C — research track (post-hackathon)**
- [ ] PowerPoint InkML ingestion (enterprise path)
- [ ] Participant voice + diarization
- [ ] ScreenAgent stage 2: companion acts on the deck
- [ ] Warehouse-companion port behind the unified attention layer → paper
      framing with Philippe (intent/attention survey lineage)

## 5. Risks

| Risk | Mitigation |
|---|---|
| Canvas ink feels laggy on phones | Send strokes on pen-up, not per-point; normalized coords keep payloads ~1 KB |
| PPT-B clutters the deck | Margin panels + a single synthesis slide; original content untouched |
| VLM escalation burns tokens | NCC gate + 3 s debounce + per-session VLM call budget |
| Two models on one vLLM instance | vLLM serves one model per process: run a second instance on another port for Qwen2.5-VL (MI300X memory is ample); HUD/agents pick by endpoint |
| Monday timeline slips | PPT-B v1 (stats + synthesis, no ink) is a half-day fallback; ink lands in the hackathon cut |
