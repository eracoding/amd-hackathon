#!/usr/bin/env bash
# AURA end-to-end inference driver — edit the variables, run once.
# Prereqs: vLLM serving on :8000 (and :8001 if VLM=1); diagnose pass clean.
set -euo pipefail

# ----------------------------- EDIT ME ------------------------------------
ROOM="data/cam.mp4"
AUDIO="data/mic_norm.wav"          # the loudnorm'd wav from diagnose
SCREEN="data/screen.mp4"
DECK_PDF="data/slides.pdf"
DECK_PPTX="data/deck.pptx"
SYNC="audio=0,room=0,screen=0"     # in-file times of your shared sync moment
WHISPER="medium"
VLM=1                              # 0 = no VLM tier (then set REGIONS below)
REGIONS=""                         # e.g. --slide-region 0,0.05,0.78,0.95 --chat-region 0.78,0.1,1,0.95
OUT="recordings/session1"
# ---------------------------------------------------------------------------

unset AURA_LLM_MOCK
VLM_FLAG=""; [ "$VLM" = "1" ] && VLM_FLAG="--vlm"

echo "==> [1/4] ingesting raw recordings"
python -m scripts.ingest_raw \
  --room "$ROOM" --audio "$AUDIO" --screen "$SCREEN" --deck "$DECK_PDF" \
  --sync "$SYNC" --whisper "$WHISPER" --whisper-device auto \
  $VLM_FLAG $REGIONS --out "$OUT"

echo "==> [2/4] replaying through the agents (live Qwen) — metrics print at the end"
python -m aura.main --replay "$OUT/events.jsonl" --speed 2 --no-monitor

echo "==> [3/4] evaluation harness (3 seeded runs vs live vLLM)"
python -m scripts.evaluate --runs 3

echo "==> [4/4] generating PPT-B"
python -m scripts.generate_pptb "$OUT/events.jsonl" \
  --pptx "$DECK_PPTX" --out reports/PPT_B.pptx

echo ""
echo "ARTIFACTS — download these via the Jupyter file browser:"
echo "  reports/PPT_B.pptx"
echo "  reports/metrics.md           <- numbers for presentation.md slide 4"
echo "  reports/debrief_*.md"
echo "  $OUT/events.jsonl            <- for the local monitor replay (demo video)"
echo "  $OUT/annotations/*.png"
