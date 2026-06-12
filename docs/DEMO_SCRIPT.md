# AURA Demo Screenplay — the staged dry run

**Cast:** Presenter (you) + 3 audience members with phones.
**Runtime:** ~9 minutes. **Deck:** the 8-slide AURA deck (meta-demo: the
system watches its own presentation — say this out loud, judges remember it).

Every beat below exists to trip a specific trigger. The trigger math is noted
so you can improvise without breaking the demo.

---

## Cast & character cards

| Role | Person | Character | Their job |
|---|---|---|---|
| Presenter | you | — | Deliver the talk; deliberately rush slide 4; pause on cue |
| **A — “the engaged one”** | front seat | Attentive, reacts | 👍 reactions; asks one technical question |
| **B — “the distracted one”** | middle seat | Drifts to phone | Leads the staged engagement dip |
| **C — “the skeptic”** | side seat | Arms crossed, privacy-minded | Joins the dip; asks the privacy question; asks the final *unanswered* question |

Print the per-person cue cards at the bottom of this file — each person only
needs their own card and a visible clock/timer.

## Room setup (10 min before)

- [ ] Camera near the screen, facing the audience; all 3 faces ≥ ~80 px tall,
      no window/backlight behind them
- [ ] Everyone connected to `http://<laptop-ip>:8765` on phones (test one
      reaction each — these land in the recording, so do this BEFORE starting,
      or accept them as a natural “people settling in” preamble)
- [ ] Deck in full-screen presentation mode; PDF export already made
- [ ] Start a shared stopwatch everyone can see (phone on the table works)
- [ ] `python -m scripts.record_session --out recordings/dryrun1 ...`
- [ ] **Sync mark:** slide 1 on screen, one loud clap, say “AURA dry run one”

---

## Master cue table

`A/B/C ↓phone` = look down at phone in lap (head pitch down → attention < 0.4).
`↑screen` = look back at the screen.

| Time | Slide | Presenter | Audience | What AURA shows (HUD) |
|---|---|---|---|---|
| 0:00 | 1 — Why rooms should understand intent | Clap, sync line, then open calmly: hook about meetings being one-directional | All ↑screen | 3 person-chips appear, engagement ~85–95%, meter green |
| 0:50 | 2 — The cost of disengaged meetings | One stat, one story. Normal pace (~140 wpm) | **A: send 👍 at 1:10** | Reaction count ticks; engagement steady |
| 1:45 | 3 — System overview | Point at architecture; name the three senses: camera, mic, devices | All ↑screen | Slide chip updates with real title + content (from screen→PDF match) |
| 2:45 | 4 — Fusion deep-dive **(the staged dip)** | **Deliberately overload:** read dense detail fast (~180 wpm), no pauses, monotone | **3:05 B ↓phone. 3:20 C ↓phone.** Both stay down ~55 s. A stays ↑screen | Meter slides toward red; slope arrow ↘ falling; B & C chips turn red |
| ~3:40 | 4 | *Keep going obliviously* — let the system catch you, don’t self-correct early | — | **EngagementAnalyst fires:** “engagement_drop — content density and pace”. **Coach nudge:** recap + slow down |
| 3:50 | 4 | **React to the nudge on camera.** Look at HUD: “The room just told me I lost you — fair. One sentence: two time windows, one engagement number, that’s all the agents see.” Slow down visibly | **4:00 B ↑screen. 4:10 C ↑screen** | Meter recovers, slope ↗ rising — the recovery is the money shot |
| 4:20 | 5 — Agentic reasoning layer | Explain the four agents, relaxed pace. “And you’ve just seen two of them act.” | **A: send 👍 at 4:30.** **4:50 A types:** “How do agents avoid spamming the presenter?” | Question lands in the HUD inbox (pending — not surfaced yet: no pause) |
| 5:15 | 6 — Privacy by design | “No identities. People are anonymous tracklets. Raw video never leaves this room.” | **5:40 C types:** “Where is the video stored exactly?” | Second question queued; inbox shows 2 |
| 6:05 | 6 | **The pause:** “Let me stop here for a moment.” **Silence, 5 full seconds.** Hold eye contact with the room | All ↑screen, silent | **PauseDetected → Moderator surfaces both questions** on the HUD, clustered |
| 6:15 | 6 | **Answer both out loud**, reading them from the HUD: cooldowns/action-budget for A’s; “on this laptop, only scalar signals propagate” for C’s | C nods (on camera) | Inbox clears — the loop visibly closes |
| 7:00 | 7 — Evaluation results | Quote your own metrics table: detection 3/3, latencies, tokens/action | All ↑screen | Steady green; nothing fires (correct behavior — say so later) |
| 7:50 | 8 — Roadmap & ask | Future work: IP cams, VLM slide understanding, edge deployment | **8:30 C types:** “Can this run fully on a Jetson?” — *presenter does NOT see/answer it* | Question stays pending |
| 8:50 | 8 | “Thank you.” Stop talking. 5 s silence, then stop the recording | — | After ingestion: **debrief lists C’s question as UNRESOLVED**, names slide 4 as lowest engagement, action items |

---

## Why each beat works (trigger math)

- **The dip needs 2 of 3 people for ~45–60 s.** Engagement is the mean of
  per-person attention; one person at a phone only drops it to ~0.66 —
  above alarm territory. Two people for under 30 s won’t move the 30 s slope
  window enough either. So B and C hold the phone-look ~55 s, staggered
  15 s apart (staggering makes the falling slope obvious on the meter).
- **Rushing on slide 4 is not theater** — the ASR computes real words/min,
  and the Analyst sees `wpm` in the room state. Your fast monotone makes
  “probable cause: pace” a grounded inference, not a guess.
- **Questions go in BEFORE the pause.** The Moderator surfaces questions on
  `PauseDetected` (≥3 s of mic silence) or slide change. Type at 4:50/5:40,
  pause at 6:05 → both get surfaced together, clustered. If you forget the
  pause, the slide 7 transition is the safety net.
- **Coach cooldown is 90 s and actions cap at 4/min** — that’s why there is
  exactly one staged dip. A second dip before ~5:30 would be silently
  dropped, which would look like a failure even though it’s correct.
- **Slide 7 silence is a feature.** Point at it in the talk track of your
  final video: “nothing fired here — no false positives is the harder half
  of the metric.”
- **The unanswered question at 8:30** exists purely so the debrief has a
  non-empty “Unresolved questions” section — the artifact that proves the
  system creates institutional memory, not just live alerts.

## After the recording

```bash
# cloud notebook:
python -m scripts.ingest_recording recordings/dryrun1 --deck slides.pdf --whisper medium
python -m aura.main --replay recordings/dryrun1/events.jsonl --speed 2
python -m scripts.evaluate
```

Screen-record the replay with the HUD visible. Suggested final video cut
(~3 min): 0:00 problem (15 s) → 0:15 architecture slide (30 s) → 0:45 replay
montage hitting the four beats: dip caught → nudge → recovery → questions
surfaced (90 s) → 2:15 debrief + metrics table (30 s) → 2:45 privacy +
roadmap (15 s). Narrate over the replay; subtitle the agent actions.

**Verification pass before filming the final cut:** replay once at high
speed (`--speed 0 --no-hud`) and check the console log shows, in order:
Analyst insight → Coach nudge → Moderator surfacing 2 questions → debrief
with slide 4 lowest + 1 unresolved question. If a beat is missing, the cue
table tells you which staged signal was too weak — re-record just that span
or adjust thresholds in `configs/default.yaml`.

---

## Printable cue cards

**PERSON A — “engaged”**
- 1:10 → send 👍
- 4:30 → send 👍
- 4:50 → type question: *“How do agents avoid spamming the presenter?”*
- Otherwise: watch the screen attentively the whole time.

**PERSON B — “distracted”**
- 3:05 → look DOWN at your phone in your lap; stay there (scroll naturally)
- 4:00 → look back UP at the screen; stay attentive to the end.

**PERSON C — “skeptic”**
- 3:20 → look DOWN at your phone; stay there
- 4:10 → look back UP at the screen
- 5:40 → type question: *“Where is the video stored exactly?”*
- 6:15 → nod when the presenter answers you (it’ll be on camera)
- 8:30 → type question: *“Can this run fully on a Jetson?”* (it will NOT be
  answered — that’s intentional)
