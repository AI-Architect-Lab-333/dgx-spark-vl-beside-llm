#!/bin/bash
# Download Qwen3-VL-8B-Instruct GGUF (Q8_0 + mmproj) next to an already-resident
# ~100 GB LLM. Does not stop or modify the existing llama-server on :8000.
set -euo pipefail
source "$HOME/inference/venv/bin/activate"
mkdir -p "$HOME/models" "$HOME/logs"
LOG="$HOME/logs/qwen3vl-download.log"
cd "$HOME/models"
echo "DL_START $(date -Iseconds)" | tee -a "$LOG"
hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \
  --local-dir Qwen3-VL-8B-Instruct-GGUF \
  --include Qwen3-VL-8B-Instruct-Q8_0.gguf \
  --include mmproj-F16.gguf \
  2>&1 | tee -a "$LOG"
echo "DL_DONE $(date -Iseconds)" | tee -a "$LOG"
du -sh Qwen3-VL-8B-Instruct-GGUF | tee -a "$LOG"
find Qwen3-VL-8B-Instruct-GGUF -name '*.gguf' | tee -a "$LOG"
