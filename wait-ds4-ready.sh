#!/bin/bash
# Block until DeepSeek on :8000 is serving AND its GPU allocation is large and
# stable. Starting Qwen-VL at the first HTTP 200 (or after a short wall-clock
# settle) CUDA-OOMs on cudaSetDevice. Core-dump restarts make the next try
# worse. Fail closed (exit 1) instead of starting anyway.
set -euo pipefail
host="${LLAMA_HOST:?LLAMA_HOST is not set}"
url="http://${host}:8000/v1/models"
tries="${WAIT_DS4_TRIES:-240}"
gpu_min="${WAIT_DS4_GPU_MIB:-90000}"
stable_sec="${WAIT_DS4_STABLE_SEC:-60}"
stable_delta="${WAIT_DS4_STABLE_DELTA_MIB:-2048}"
gpu_tries="${WAIT_DS4_GPU_TRIES:-120}"
sleep_s=5

ds4_pid() {
  ps -ww -C llama-server -o pid=,args= 2>/dev/null \
    | awk '/--port 8000/ { print $1; exit }'
}

gpu_mib_for() {
  local pid="$1"
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v p="$pid" '{
        gsub(/ /,"",$1); gsub(/ /,"",$2);
        if ($1+0==p+0) { print $2+0; exit }
      }'
}

i=0
while [ "$i" -lt "$tries" ]; do
  i=$((i + 1))
  if curl -sf --max-time 2 "$url" | grep -q deepseek; then
    echo "DeepSeek HTTP ready after ${i} probe(s)"
    break
  fi
  if [ "$i" -eq "$tries" ]; then
    echo "DeepSeek API not ready after $((tries * sleep_s)) s; not starting Qwen-VL" >&2
    exit 1
  fi
  sleep "$sleep_s"
done

pid=""
j=0
while [ "$j" -lt 30 ]; do
  j=$((j + 1))
  pid="$(ds4_pid || true)"
  if [ -n "$pid" ]; then
    echo "DeepSeek llama-server pid=${pid}"
    break
  fi
  sleep 2
done
if [ -z "$pid" ]; then
  echo "no llama-server with --port 8000; not starting Qwen-VL" >&2
  exit 1
fi

last=""
stable_for=0
k=0
while [ "$k" -lt "$gpu_tries" ]; do
  k=$((k + 1))
  m="$(gpu_mib_for "$pid" || true)"
  if [ -z "$m" ]; then
    echo "gpu_mib pid=${pid} unknown (sample ${k})"
    last=""
    stable_for=0
    sleep "$sleep_s"
    continue
  fi
  echo "gpu_mib pid=${pid} ${m} (min ${gpu_min}, stable ${stable_sec}s delta ${stable_delta})"
  if [ "$m" -lt "$gpu_min" ]; then
    last="$m"
    stable_for=0
    sleep "$sleep_s"
    continue
  fi
  if [ -n "$last" ]; then
    diff=$((m - last))
    if [ "$diff" -lt 0 ]; then
      diff=$((-diff))
    fi
    if [ "$diff" -le "$stable_delta" ]; then
      stable_for=$((stable_for + sleep_s))
      if [ "$stable_for" -ge "$stable_sec" ]; then
        echo "DeepSeek GPU memory stable at ${m} MiB for ${stable_for}s; starting Qwen-VL"
        exit 0
      fi
    else
      stable_for=0
    fi
  fi
  last="$m"
  sleep "$sleep_s"
done

echo "DeepSeek GPU memory not stable after $((gpu_tries * sleep_s)) s; not starting Qwen-VL" >&2
exit 1
