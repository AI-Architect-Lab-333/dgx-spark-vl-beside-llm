#!/bin/bash
# Prove both APIs: large LLM on :8000 still serves, Qwen3-VL on :8001 does
# text AND vision. Do not use `set -e` — systemctl is-active returns 3 when
# inactive, which would abort the rest of the checks.
set -u
HOST="${LLAMA_HOST:-100.x.y.z}"
DS4="http://${HOST}:8000/v1"
QWEN="http://${HOST}:8001/v1"

echo "=== large LLM /v1/models ==="
curl -sS --max-time 10 "$DS4/models"
echo

echo "=== Qwen3-VL /v1/models ==="
curl -sS --max-time 10 "$QWEN/models"
echo

echo "=== Qwen3-VL text chat ==="
curl -sS --max-time 60 "$QWEN/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"Reply with exactly the four characters: QWEN_OK"}],"max_tokens":32,"temperature":0}'
echo

echo "=== fetch vision test image ==="
python3 - <<'PY'
import base64, json, pathlib, urllib.request
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as r:
    img = r.read()
pathlib.Path("/tmp/qwen3vl-test.jpg").write_bytes(img)
payload = {
    "model": "qwen3-vl",
    "max_tokens": 80,
    "temperature": 0.2,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}},
            {"type": "text", "text": "In one short English sentence, say what is in this photo."},
        ],
    }],
}
pathlib.Path("/tmp/qwen3vl-vision.json").write_text(json.dumps(payload))
print("vision payload bytes", len(img))
PY

echo "=== Qwen3-VL vision chat ==="
curl -sS --max-time 120 "$QWEN/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/qwen3vl-vision.json
echo

echo "=== large LLM still answering after vision? ==="
curl -sS --max-time 10 "$DS4/models"
echo
curl -sS --max-time 60 "$DS4/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say hi in 3 words."}],"max_tokens":16}'
echo

echo "=== services ==="
systemctl --user is-active llama-server || true
systemctl --user is-active llama-server-vl || true
echo "=== memory ==="
free -h
