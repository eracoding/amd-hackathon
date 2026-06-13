# CLOUD_RUNBOOK.md — inference on the AMD server, command by command

You're moving `aura.zip`, your three recordings (mic audio, IP-cam video,
Teams screen capture), and the deck (PDF + original .pptx) to
notebook.amd.com. This is the exact sequence. Run everything in a
**Jupyter terminal** (File → New → Terminal), not in notebook cells.

Two AMD-specific gotchas are already handled in the code, but know them:
1. **faster-whisper cannot use the AMD GPU** (CTranslate2 has no ROCm
   backend). `--whisper-device auto` now correctly falls back to CPU on
   ROCm. CPU int8 `medium` transcribes a 10-min recording in a few minutes.
2. **Storage is the binding constraint (25 GB)**, not VRAM. Qwen2.5-7B
   (~15.2 GB) + env (~5 GB) fits. Adding the VL model: use the **3B** VL
   variant (~7 GB) — two 7B models would blow the disk budget.

---

## Phase 0 — land and verify (5 min)

```bash
cd ~ && unzip -q aura.zip -d aura && cd aura
mkdir -p data && mv ~/mic.m4a ~/cam.mp4 ~/screen.mp4 ~/slides.pdf ~/deck.pptx data/ 2>/dev/null

df -h .                      # how much disk you actually have
rocm-smi                     # GPU visible? note VRAM
python --version             # need 3.10+
python -c "import torch; print(torch.__version__, '| cuda(rocm):', torch.cuda.is_available(), '| hip:', torch.version.hip)"
python -c "import vllm; print('vLLM', vllm.__version__)" || echo "vLLM missing"
```

AMD's PyTorch containers usually ship torch-ROCm and often vLLM. If torch
is missing or CPU-only, you're in the wrong container image — pick the
ROCm/PyTorch one when launching the notebook.

## Phase 1 — dependencies (10 min, ~2 GB)

```bash
export HF_HOME=~/aura/hf_cache          # keep model downloads on this disk
echo 'export HF_HOME=~/aura/hf_cache' >> ~/.bashrc

pip install pydantic aiohttp pypdfium2 python-pptx pytest pytest-asyncio
pip install opencv-python-headless "mediapipe<=0.10.14"   # legacy API, zero setup
pip install faster-whisper                                 # CPU int8 path

# chat-pane OCR (optional but you have chat in the screen capture):
sudo apt-get update -q && sudo apt-get install -y -q tesseract-ocr \
  || conda install -y -c conda-forge tesseract                # no-sudo fallback
pip install pytesseract

# vLLM only if Phase 0 said it's missing (ROCm wheel set):
# pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2

python -m pytest tests/ -q            # expect 22 passed (uses mock LLM)
```

If the suite passes here, every pipeline component works on this machine.

## Phase 2 — serve the LLM (one-time download ~15 GB)

Use tmux so the server survives the browser tab:

```bash
tmux new -s llm
bash scripts/serve_llm.sh             # downloads Qwen2.5-7B, serves :8000
# wait for "Uvicorn running on http://0.0.0.0:8000", then Ctrl-B, D to detach
```

Smoke-test from another terminal:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Reply with the word ready"}],"max_tokens":5}' | python -m json.tool
```

**Optional Tier-2 VLM** (only if `df -h` shows ≥ 10 GB free after Phase 2):

```bash
tmux new -s vlm
AURA_VLM=1 AURA_VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct bash scripts/serve_llm.sh   # :8001
# detach; then export AURA_VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct in your shell too
```

## Phase 3 — ingest a 30-second slice FIRST (always)

Real Teams chrome and fonts differ from any test. Validate regions on a
slice before burning time on full files:

```bash
ffmpeg -ss 60 -t 30 -i data/screen.mp4 -c copy /tmp/slice.mp4 -y

# with the VLM serving (:8001), regions are found automatically:
python -m scripts.ingest_raw --screen /tmp/slice.mp4 --deck data/slides.pdf \
    --vlm --out /tmp/slice_test
# (no VLM? pass them manually: --slide-region 0.0,0.05,0.78,0.95
#  --chat-region 0.78,0.10,1.0,0.95)
```

The log prints `auto regions: slide=[...] chat=[...]` — eyeball them against
one frame; explicit --slide-region/--chat-region always override.

Read the log: slide changes detected? chat lines sensible? If slides don't
match, your regions are off — grab one frame to measure them:
`ffmpeg -ss 60 -i data/screen.mp4 -frames:v 1 /tmp/frame.png -y`, open it in
Jupyter's file browser, and estimate the slide/chat boundaries as fractions
of width/height. Iterate the slice until clean.

## Phase 4 — full ingestion (the real inference, heavy perception)

Find your sync moment first: open each file (Jupyter can play mp4/m4a),
locate one shared instant (clap / "let's begin" / slide 1 appearing), note
its in-file time in each.

```bash
python -m scripts.ingest_raw \
    --room data/cam.mp4 --audio data/mic.m4a --screen data/screen.mp4 \
    --deck data/slides.pdf \
    --sync "audio=<t>,room=<t>,screen=<t>" \
    --whisper medium --whisper-device auto \
    --vlm \
    --out recordings/session1            # drop --vlm if you skipped :8001
```

With --vlm: regions auto-detected, chat read by the VLM (more robust than
OCR on real Teams fonts), ink transcribed + intent-tagged, off-deck screens
classified. Expect: attention events from the camera, transcript + voice
questions from the audio, slide changes + annotations + chat messages from
the screen → `recordings/session1/events.jsonl`. Sanity-check it:

```bash
wc -l recordings/session1/events.jsonl
grep -c AttentionEvent recordings/session1/events.jsonl
grep InteractionEvent recordings/session1/events.jsonl | head -5
grep ScreenAnnotationEvent recordings/session1/events.jsonl
```

## Phase 5 — agents over the session (the demo)

```bash
unset AURA_LLM_MOCK                       # real Qwen now
python -m aura.main --replay recordings/session1/events.jsonl --speed 2
```

Console shows agent actions live; token/latency metrics print at the end —
**copy these into presentation.md's ⚙ placeholders.**

The monitor window (:8766): if the deployment has jupyter-server-proxy, open
`https://<your-notebook-host>/user/<you>/proxy/8766/`. If WebSockets don't
survive the proxy, two fallbacks: (a) `--no-monitor` and use the console
feed for the metrics run, then (b) for the demo *video*, download
`events.jsonl` to your laptop and run the replay there with
`AURA_LLM_URL=https://<cloudflared-url>/v1` pointing at the cloud model —
monitor renders locally, reasoning stays on the MI300X.

## Phase 6 — PPT-B and the metrics table

```bash
python -m scripts.generate_pptb recordings/session1/events.jsonl \
    --pptx data/deck.pptx --out reports/PPT_B.pptx

AURA_LLM_MOCK= python -m scripts.evaluate --runs 3     # vs live vLLM -> reports/metrics.md
```

## Phase 7 — collect the artifacts

Download via the Jupyter file browser (right-click → Download):

```
reports/PPT_B.pptx                    # the headline artifact
reports/debrief_*.md
reports/metrics.md                    # numbers for slide 4
recordings/session1/events.jsonl      # for the local monitor replay
recordings/session1/annotations/*.png # the detected ink patches
```

---

## Phase 3b — when a run produces zero events: diagnose, don't guess

```bash
python -m scripts.diagnose --audio data/mic.m4a \
    --screen data/screen.mp4 --deck data/slides.pdf --vlm
```

One command answers both failure modes: audio loudness in dBFS with the
exact ffmpeg normalize command if it's too quiet for VAD, and per-region
match similarities (full frame vs. your region vs. VLM auto) over four
sampled frames saved to `debug_frames/` — with a verdict telling you which
region to use. Ingestion now also auto-retries audio without VAD and prints
similarity stats whenever zero slides match.

## If something breaks

| Symptom | Fix |
|---|---|
| vLLM install fails | Use AMD's prebuilt ROCm vLLM container/image; or fallback `pip install llama-cpp-python` + a GGUF Qwen and point AURA_LLM_URL at its server |
| `CTranslate2 ... CUDA` error from whisper | You forced `--whisper-device cuda`; use `auto` or `cpu` on AMD |
| Disk full mid-download | `rm -rf ~/aura/hf_cache/hub/models--*` of anything half-downloaded; switch to Qwen2.5-3B-Instruct (`AURA_LLM_MODEL`) — ~6 GB |
| Slides never match | Wrong `--slide-region`; verify on a single frame; remember fractions, not pixels |
| `VAD removed all audio` | Quiet phone recording: `ffmpeg -i mic.m4a -af loudnorm=I=-16 -ac 1 -ar 16000 mic_norm.wav`, ingest the wav (pipeline auto-retries without VAD meanwhile) |
| ONNX `device_discovery` DRM warnings | Harmless: onnxruntime (Silero VAD) probing /sys in a container. Ignore. |
| Monitor stuck on "waiting for session" via proxy | Fixed: WS URL now preserves the jupyter-proxy path. Hard-refresh (Ctrl-Shift-R) after redeploying |
| Debrief shows only slide 0 / 0.0 engagement | Slide match failed AND/OR camera found no faces — run `scripts/diagnose.py --screen ... --room ...` |
| Faces not detected on wide IP-cam shots | `python -m scripts.ingest_room_ar --room cam.mp4 --ar-path <attention_room> --estimator onnx --model <model> --merge-into <events.jsonl> --t0 <room t0>` |
| "Unresolved questions" are the presenter's own prompts | Fixed: meta-questions ("any questions?", "do we have a question…") are filtered from voice-question detection |
| Annotations missed on Teams/PowerPoint Live | Fixed: diffing now uses a LIVE reference (the screen's own first stable frame per slide), renderer-independent; VLM title fallback identifies slides when NCC fails |
| Chat OCR garbage | Crop the chat region tighter (exclude avatars); if still noisy, drop `--chat-region` — voice + annotations still flow |
| Monitor proxy won't load | Run replay `--no-monitor` for metrics; do the visual demo locally against the tunneled LLM |
| GPU OOM with both models | Utilizations must sum < 1.0: vLLM PREALLOCATES that fraction at startup (165 GB used by an idle 7B is normal). Use 0.40 text / 0.35 VL — you have VRAM to spare; OOM means another tenant or stale process: `rocm-smi`, kill yours, restart |
