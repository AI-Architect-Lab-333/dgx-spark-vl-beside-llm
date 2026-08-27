#!/bin/bash
# Install Qwen3-VL llama-server as a systemd --user service.
# Does not touch the existing large LLM unit on :8000.
# Run ON the GPU box as the user that owns the model/binaries.
set -euo pipefail

UNIT_SRC="${1:-$HOME/inference/llama-server-vl.service}"
UNIT_DST="$HOME/.config/systemd/user/llama-server-vl.service"

if [ ! -f "$UNIT_SRC" ]; then
  echo "Missing unit file: $UNIT_SRC" >&2
  exit 1
fi
if [ ! -x "$HOME/inference/llama-server-vl-start.sh" ]; then
  echo "Missing start script: $HOME/inference/llama-server-vl-start.sh" >&2
  exit 1
fi
if [ ! -x "$HOME/inference/wait-ds4-ready.sh" ]; then
  echo "Missing wait script: $HOME/inference/wait-ds4-ready.sh" >&2
  exit 1
fi
if [ ! -x "$HOME/inference/llama.cpp/build/bin/llama-server" ]; then
  echo "Missing llama-server binary" >&2
  exit 1
fi
if [ ! -f "$HOME/models/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q8_0.gguf" ]; then
  echo "Missing Qwen3-VL model files" >&2
  exit 1
fi

echo "=== install user unit (port 8001, large LLM stays on 8000) ==="
mkdir -p "$HOME/.config/systemd/user"
chmod +x "$HOME/inference/llama-server-vl-start.sh" "$HOME/inference/wait-ds4-ready.sh"
cp "$UNIT_SRC" "$UNIT_DST"
systemctl --user daemon-reload
systemctl --user enable llama-server-vl.service
echo "=== start ==="
systemctl --user restart llama-server-vl.service
sleep 3
systemctl --user --no-pager --full status llama-server-vl.service || true
echo "=== listen? (Qwen3-VL load is ~10 s on this hardware) ==="
ss -tln | grep 8001 || echo "not listening yet — wait, then: curl http://100.x.y.z:8001/v1/models"
echo "=== large LLM still up? ==="
systemctl --user is-active llama-server || true
ss -tln | grep 8000 || true
echo "DONE"
