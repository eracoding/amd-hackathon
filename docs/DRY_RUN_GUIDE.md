# AURA Dry-Run Demo Guide — local sensors, cloud GPU

Your GPU lives in the cloud (notebook.amd.com); your sensors are in the room.
The architecture handles this cleanly because **raw video/audio never needs to
reach the GPU — only events do**. This guide is the exact sequence for the
record-then-replay demo, plus the optional live topology.

---

## The principle

```
LOCAL (laptop, CPU only)                    CLOUD (AMD notebook, GPU)
─────────────────────────                   ─────────────────────────
record: camera/mic/screen/interactions  →   upload ~1 GB once
                                            ingest: whisper(GPU) + vision +
                                                    slide matching → events.jsonl
                                            replay events through vLLM agents
                                            → HUD + debrief + metrics, live
```

Nothing is faked: the agents and Qwen run for real; only *time* is shifted.
Bonus: this is exactly the privacy story — video stays where it was filmed.

---

## Step 1 — Prepare (5 min, local)

```bash
pip install pydantic aiohttp                      # recorder needs only these
sudo apt install ffmpeg                            # or: brew install ffmpeg
```

Export your deck to **PDF** (PowerPoint/Keynote: File → Export). The slide
tracker matches your screen recording against the PDF pages — no Office APIs,
no VLM tokens.

Find your devices:
```bash
# Linux:  v4l2-ctl --list-devices   &&  pactl list short sources
# macOS:  ffmpeg -f avfoundation -list_devices true -i ""
```

## Step 2 — Record the dry run (one command, local)

```bash
python -m scripts.record_session --out recordings/dryrun1 \
    --cam rtsp://user:pass@192.168.1.50/stream1   # or --cam 0 for webcam
```

This simultaneously records:
- `room.mp4` — audience camera (IP cam via RTSP, or webcam)
- `audio.wav` — your mic, 16 kHz mono (whisper-ready)
- `screen.mp4` — your presentation display at 5 fps
- `interactions.jsonl` — **live**: friends/colleagues open
  `http://<your-laptop-ip>:8765` on phones and ask questions / react during
  your talk — captured with exact timestamps, nothing to post-process
- `manifest.json` — per-stream start times (the sync source of truth)

Then: **clap once while slide 1 is on screen** (audit marker), give your
~10-minute talk, have 2–3 people send questions at planned moments,
and `Ctrl-C` to stop (recorders finalize cleanly).

**Sync, demystified:** all streams share one machine's clock; AURA's fusion
window is 10 s, so the ±0.3 s ffmpeg startup jitter is irrelevant. The clap
is only for manual verification, not alignment.

**What to keep in mind while recording:**
- Camera: audience faces should be ≥ ~80 px tall in frame; avoid strong
  backlight (window behind people kills landmark detection)
- Ask one "audience member" to visibly look at their phone for ~30 s during
  your densest slide — that's your engagement-dip demo moment
- Mic: any USB/headset mic beats the laptop array; keep it near you
- Screen: present in full-screen mode; the matcher tolerates letterboxing,
  compression, and brightness shifts (validated to 0.99 similarity)

## Step 3 — Upload & ingest (cloud notebook)

Upload `recordings/dryrun1/` + `slides.pdf` + the repo (~1 GB total for
10 min — fits the 25 GB budget easily). In a notebook terminal:

```bash
pip install -r requirements.txt
pip install pypdfium2 faster-whisper
python -m scripts.ingest_recording recordings/dryrun1 --deck slides.pdf \
    --whisper medium --whisper-device auto      # ROCm shows up as cuda
```

Heavy perception runs here (whisper can use the GPU). Output:
`recordings/dryrun1/events.jsonl` — one time-ordered multimodal stream.

## Step 4 — The demo itself (cloud)

```bash
bash scripts/serve_llm.sh &                 # vLLM + Qwen2.5-7B on :8000
python -m aura.main --replay recordings/dryrun1/events.jsonl --speed 2
```

Open the HUD (port 8766 — in Jupyter use the proxy URL
`https://<notebook-host>/user/<you>/proxy/8766/` if jupyter-server-proxy is
available, else screen-record the HUD via a local tunnel or run the replay
locally against the tunneled LLM). You'll show, on *your real talk*:
engagement reacting to your real audience, your real transcript scrolling,
the questions your colleagues actually typed being surfaced by the Moderator,
and a debrief naming your weakest slide. Finish with
`python -m scripts.evaluate` numbers.

## Optional Step 5 — live segment (only if it works effortlessly)

In the cloud notebook terminal:
```bash
./cloudflared tunnel --url http://localhost:8000   # prints a public HTTPS URL
```
On your laptop:
```bash
export AURA_LLM_URL="https://<printed-url>/v1"
python -m aura.main --live --camera 0
```
Perception local, reasoning in the cloud. Per agent call: ~3 KB up, ~1 KB
down, a few calls/minute — latency budget ~1–3 s per action, fine for nudges.
If the tunnel is blocked, skip this; the replay demo is the stronger act.

---

## Failure modes & answers

| Worry | Reality |
|---|---|
| "Cloud GPU is a bottleneck for my sensors" | Only if raw media crossed the network. It doesn't — events do. |
| Streams drift out of sync | Single machine clock + manifest t0s; 10 s fusion windows absorb sub-second jitter. |
| Whisper mishears me | Use `--whisper medium`, speak near the mic; agents reason over gist, not verbatim. |
| Slide animations / live demo segments | Matcher returns "no match" → slide context simply holds; VLM escalation is the Phase 6 upgrade. |
| Replay feels fake to judges | Frame it as the *evaluation harness*: deterministic input, real agents, real LLM, reproducible metrics. That's rigor, not weakness. |
