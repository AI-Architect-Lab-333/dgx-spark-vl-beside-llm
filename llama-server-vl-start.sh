#!/bin/bash
# Start Qwen3-VL-8B-Instruct as an OpenAI-compatible vision API on Tailscale only.
# Runs alongside the existing large LLM on :8000 — do not bind 8000.
set -euo pipefail
MODEL_DIR="$HOME/models/Qwen3-VL-8B-Instruct-GGUF"
MODEL="$MODEL_DIR/Qwen3-VL-8B-Instruct-Q8_0.gguf"
MMPROJ="$MODEL_DIR/mmproj-F16.gguf"
BIN="$HOME/inference/llama.cpp/build/bin/llama-server"
HOST="${LLAMA_HOST:-100.x.y.z}"
PORT="${LLAMA_PORT:-8001}"
CTX="${LLAMA_CTX:-16384}"
if [ ! -x "$BIN" ]; then echo "missing $BIN" >&2; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "missing model $MODEL" >&2; exit 1; fi
if [ ! -f "$MMPROJ" ]; then echo "missing mmproj $MMPROJ" >&2; exit 1; fi
echo "Serving $MODEL (mmproj $MMPROJ) on http://$HOST:$PORT/v1"
exec "$BIN" \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --parallel 1 \
  --image-min-tokens 1024 \
  -ngl 99 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --presence-penalty 1.5 \
  --alias qwen3-vl \
  --jinja
