#!/usr/bin/env bash
# Launch the local LLM backend on the AMD GPU (ROCm).
# Storage budget check: Qwen2.5-7B-Instruct bf16 ~= 15.2 GB on disk.
set -euo pipefail
MODEL="${AURA_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "Serving $MODEL via vLLM (OpenAI-compatible on :8000)"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000
