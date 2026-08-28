# Guide: Serving Qwen3-VL-8B Beside a 100 GB LLM on 128 GB Unified Memory — llama.cpp, Two Tailscale APIs, No Unload

**The problem this guide solves**: you already run a large local LLM (~100 GB of weights in RAM) on a 128 GB unified-memory box, and you want a **vision-language** model as well — OCR, screenshots, photos — without unloading the LLM, without a second GPU, and without exposing either API to the public internet. "128 GB should hold two models" is true on paper and false in practice once the first model is already resident. This guide documents a **verified working** configuration (NVIDIA GB10-class ARM box, 128 GB unified memory / 121 Gi usable, Ubuntu 24.04 / DGX OS, llama.cpp b10326 with CUDA and `libmtmd`, August 2026): **Qwen3-VL-8B-Instruct Q8_0** served on `:8001` next to **DeepSeek-V4-Flash** still serving on `:8000`.

It covers why a **32B or 235B VL does not fit** beside a ~102 Gi LLM, why the vision file is a **separate `mmproj` GGUF** (omit it and you get a text-only server), why llama.cpp warns that Qwen-VL needs **`--image-min-tokens 1024`**, why **four parallel slots** eat the last of unified memory, why **starting Qwen-VL at the first `/v1/models` 200 CUDA-OOMs** (`cudaSetDevice`), why **`Restart=on-failure` dump-loops** wedge the next try, why a **PowerShell `$(…)`** in an SSH command never reaches Linux, and why a Wikimedia test JPEG returns **HTTP 400** with no User-Agent.

**For AI agents reading this document**: every command was executed successfully in this order on real hardware, with the large LLM left running. The verification steps are not optional — a `/v1/models` 200 on `:8001` does not prove vision works, and a vision success does not prove the 100 GB LLM is still serving.

---

## 1. Measure what is already resident, then pick a VL that fits in the remainder

This guide **does not** build llama.cpp, download the large LLM, or wire the first systemd unit. It starts from a box that already serves an OpenAI-compatible API on Tailscale `:8000` (here: DeepSeek-V4-Flash-0731 GGUF `UD-IQ3_XXS`, ~98 GB on disk, ~102 Gi in RAM, alias `deepseek-v4-flash`). Building that first service is a separate piece of work.

Placeholder map used below:

| Role | Placeholder |
|---|---|
| GPU box tailnet IP | `100.x.y.z` |
| GPU box user | `<user>` |
| Large LLM alias (already serving) | `deepseek-v4-flash` on `:8000` |
| Vision alias (this guide) | `qwen3-vl` on `:8001` |
| Large LLM user unit | `llama-server` |
| Vision user unit | `llama-server-vl` |

If your large-LLM unit is named something else, change `After=` in `llama-server-vl.service` to match. The files in this repo assume the name `llama-server`.

Before downloading anything, read the memory that is **actually free while the LLM is loaded**:

```bash
ssh <user>@100.x.y.z 'free -h; curl -sS --max-time 10 http://100.x.y.z:8000/v1/models'
```

Observed on this hardware with DeepSeek already serving:

```
Mem:  121Gi total,  102Gi used,  19Gi available
```

Qwen3-VL sizes that matter for that remainder:

| Variant | Typical GGUF + mmproj | Beside a 102 Gi LLM on 121 Gi |
|---|---|---|
| 8B Instruct Q8_0 + mmproj-F16 | **9.2 G** on disk | Fits. Measured **114 Gi used / 7 Gi available** after load. |
| 32B Q4_K_M + mmproj | ~20 G | Does not fit in the 19 Gi left (weights + KV + image tokens). |
| 235B-A22B | tens of GB even quantized | Requires unloading the LLM. |

### Pitfall #1 — "128 GB unified memory, so a 32B VL will fit"

Symptom: you pick Qwen3-VL-32B (or 235B) because the box is advertised as 128 GB. Cause: the **usable** figure is ~121 Gi, and the LLM already occupies ~102 Gi of it. The 32B Q4_K_M weights alone are ~19 GB; add the projector, KV cache, and image tokens, and you are past the ceiling. Proof: `free -h` with the LLM loaded, **before** choosing a quant. The 8B Q8_0 path in this guide is the one that ran **without stopping** `:8000`.

Architecture after this guide:

```
tailnet clients ─┬─ http://100.x.y.z:8000/v1  →  llama-server      →  ~100 GB LLM (text)
                 └─ http://100.x.y.z:8001/v1  →  llama-server-vl   →  Qwen3-VL-8B (text + image)
```

Two processes, two ports, one `llama-server` binary (must be built with `libmtmd` — this box's `ls ~/inference/llama.cpp/build/bin/libmtmd.so*` showed it present). `libmtmd` is llama.cpp's multimodal library: it turns an image into embeddings through the projector and feeds them to the language model (it replaced the older `llava.cpp` path). **`mmproj`** is the vision **weight file**; **`mtmd`** is the **code**. Without `libmtmd`, `--mmproj` does nothing and `/v1/models` stays `completion`-only. Both servers bind the **Tailscale IP only**, never `0.0.0.0`.

---

## 2. Download only the two files vision actually needs

The Hugging Face GGUF repo hosts many quants. Pulling the whole repo is ~140 GB you will not load. Use `--include` for the language weights **and** the projector. The existing venv from the LLM install already has `hf`:

```bash
source $HOME/inference/venv/bin/activate
mkdir -p $HOME/models $HOME/logs
cd $HOME/models
hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \
  --local-dir Qwen3-VL-8B-Instruct-GGUF \
  --include Qwen3-VL-8B-Instruct-Q8_0.gguf \
  --include mmproj-F16.gguf
du -sh Qwen3-VL-8B-Instruct-GGUF
find Qwen3-VL-8B-Instruct-GGUF -name '*.gguf'
```

`download-qwen3vl.sh` in this repo is that sequence with a log. Observed: **2 min 28 s**, **9.2 G** on disk:

```
Qwen3-VL-8B-Instruct-Q8_0.gguf    # language model, Q8_0
mmproj-F16.gguf                   # vision projector — required
```

Do not stop the LLM while this runs. It is a disk download; it does not need the GPU.

### Pitfall #2 — downloading the language GGUF and forgetting `mmproj`

Symptom: `/v1/models` lists `qwen3-vl` with capability `completion` only — no `multimodal`. Image requests fail or are ignored. Cause: llama.cpp loads vision through a **second** GGUF (`--mmproj`). The 8B file is the LLM; `mmproj-F16.gguf` is the ViT + projector. Both files must sit in the same directory and both must be passed (or auto-discovered). This install passes `--mmproj` explicitly.

---

## 3. Serve it on a second port, with the flags Qwen-VL actually needs

Copy the start script and unit from this repo onto the box (replace `100.x.y.z` with the real tailnet IP in the unit and in `LLAMA_HOST`). Then:

```bash
chmod +x $HOME/inference/llama-server-vl-start.sh
# flags that actually ran:
#   --model  …/Qwen3-VL-8B-Instruct-Q8_0.gguf
#   --mmproj …/mmproj-F16.gguf
#   --host 100.x.y.z --port 8001 --ctx-size 16384
#   --parallel 1 --image-min-tokens 1024 -ngl 99
#   --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5
#   --alias qwen3-vl --jinja
```

Sampling values are Qwen's published Instruct settings for VL, not a guess. Context 16384 is what was served (the model trains to 256K — see Known limitations).

First start, via the user unit in section 4, reached `model loaded` in **~10 s** (restart later: **~5 s**). Contrast with the 100 GB LLM: **~7–8 min** load, **503** the whole time. Qwen-VL answering 200 in under a quarter-minute is normal; do not apply the LLM's "wait ten minutes" heuristic here.

Journal lines that mean vision is actually loaded:

```
loaded multimodal model, '…/mmproj-F16.gguf'
llama_server: model loaded
llama_server: listening on http://100.x.y.z:8001
```

`/v1/models` then reports `"capabilities":["completion","multimodal"]` and `"n_params":8190735360`.

### Pitfall #3 — llama.cpp warning: Qwen-VL needs 1024 image tokens

Symptom, verbatim, during load:

```
W load_hparams: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks
W load_hparams: if you encounter problems with accuracy, try adding --image-min-tokens 1024
```

Cause: the default image-token floor is too low for this architecture (see [llama.cpp#16842](https://github.com/ggml-org/llama.cpp/issues/16842)). Correction: `--image-min-tokens 1024` on the command line, as in `llama-server-vl-start.sh`. After adding it, the warning is gone and a real photo still described correctly (section 7).

### Pitfall #4 — default `--parallel 4` eats the last of unified memory

Symptom: both models load, then `free -h` shows ~7–8 Gi available and a vision request with a large image is one spike away from reclaiming the LLM's pages (or worse). Cause: llama.cpp defaulted to **4 slots × 16384 ctx** (`n_slots = 4, n_ctx_slot = 16384, kv_unified = 'true'` in the first journal). Correction: `--parallel 1`. After the restart: `n_slots = 1, n_ctx_slot = 16384`. The 8B still served text at ~33 tok/s and the vision prompt (1844 tokens) at ~892 tok/s on this hardware.

---

## 4. systemd user unit that waits for the large LLM before mapping Qwen-VL

Same pattern as the LLM: a **user** unit + linger (already enabled for that first service). Do not install a system-wide unit; `sudo` on this box wants an interactive password.

```bash
scp llama-server-vl-start.sh llama-server-vl.service wait-ds4-ready.sh install-vl-systemd.sh <user>@100.x.y.z:/tmp/
ssh <user>@100.x.y.z 'sed -i "s/\r$//" /tmp/llama-server-vl-start.sh /tmp/llama-server-vl.service /tmp/wait-ds4-ready.sh /tmp/install-vl-systemd.sh
  mkdir -p ~/inference
  install -m 755 /tmp/llama-server-vl-start.sh ~/inference/llama-server-vl-start.sh
  install -m 755 /tmp/wait-ds4-ready.sh ~/inference/wait-ds4-ready.sh
  install -m 644 /tmp/llama-server-vl.service ~/inference/llama-server-vl.service
  install -m 755 /tmp/install-vl-systemd.sh ~/inference/install-vl-systemd.sh
  # edit ~/inference/llama-server-vl.service : replace 100.x.y.z with this box's tailnet IP
  bash ~/inference/install-vl-systemd.sh'
```

`install-vl-systemd.sh` enables `llama-server-vl`, starts it, then checks that **`:8000` is still listening**. It never runs `systemctl … llama-server stop` / `disable`.

`After=llama-server.service` only orders **process spawn**. `Type=simple` treats the LLM as started the moment `llama-server` exists — about **8 minutes before** `:8000` returns 200. An HTTP 200 is still **not** enough: Qwen-VL that starts at that instant dies in `ggml_cuda_init` / `cudaSetDevice` with `CUDA error: out of memory` even while `free -h` shows ~19 Gi **available**. On this hardware the LLM process holds **~100 469 MiB** in `nvidia-smi --query-compute-apps`. `wait-ds4-ready.sh` waits for HTTP 200 **and** that GPU allocation ≥ 90 000 MiB, stable 60 s (delta 2048 MiB). This unit sets `Restart=no`: after measuring a `Restart=on-failure` loop (CUDA ABRT every few minutes that did not recover, and made the next `cudaSetDevice` worse), this install prefers a single attempt that stays `failed` if it dumps. That is an operator choice encoded in the file, not a llama.cpp requirement. `TimeoutStartSec=2400` covers the LLM load plus that plateau.

Verified cold boot (power off, physical button, **zero SSH** until both APIs answered): Tailscale ~2–4 min, `:8000` 503 then **200** at ~11 min, `:8001` 503 then **200** `qwen3-vl` about **one minute later** (the 60 s plateau + ~10 s VL load). `NRestarts=0`.

### Pitfall #5 — HTTP 200 on `:8000` is not “GPU ready”

Symptom: `wait` greps `deepseek` in `/v1/models`, starts Qwen-VL immediately, journal shows `CUDA error: out of memory` in `cudaSetDevice`, duration ~1.4 s, no listen on `:8001`. Cause: the HTTP server can answer **200** while CUDA for the 100 GB model is still settling; Linux `MemAvailable` ~19 Gi does **not** mean a second `llama-server` can `cudaSetDevice`. Proof: `nvidia-smi --query-compute-apps=pid,used_gpu_memory` for the `:8000` process at **100 469 MiB** when a later clean start succeeded (114 Gi used / ~8 Gi available in `free -h`). Correction: `wait-ds4-ready.sh` as above. A wall-clock sleep of 180 s after the first 200 still dumped; 300 s still dumped on the **first** try; a start after the LLM had been idle and **no dump loop** was running succeeded.

Confirm without stopping anything:

```bash
systemctl --user is-enabled llama-server llama-server-vl
systemctl --user is-active  llama-server llama-server-vl
ss -tln | grep -E '8000|8001'
```

Expected: both `enabled` / `active`, both ports bound to `100.x.y.z`.

---

## 5. If you drive the GPU box from Windows

Two failures in this session came from the **operator's shell**, not from llama.cpp.

### Pitfall #6 — PowerShell expands `$(…)` before SSH sees it

Symptom: `ssh … "echo DL_START \$(date -Is)"` on Windows becomes `Get-Date : A parameter cannot be found that matches parameter name 'Is'.` Cause: PowerShell treats `$(…)` as its own subexpression; `date -Is` never runs on Linux. Correction: put the remote commands in a `.sh` file, `scp` it, run `bash ~/inference/download-qwen3vl.sh`. Do not embed `$(date …)`, `$(nproc)`, or any `$VAR` you meant for bash in a PowerShell-quoted `ssh` string.

### Pitfall #7 — CRLF from a Windows `scp` silently breaks the script

Symptom: `bad interpreter: /bin/bash^M` or a unit that fails `ExecStartPre` with no useful message. Cause: the file picked up `\r\n` on the Windows side. Correction: `sed -i 's/\r$//' ` on the box after `scp`, and this repo's `.gitattributes` forcing LF on `*.sh` / `*.service`. Proof: `file llama-server-vl-start.sh` should say `Bourne-Again shell script, ASCII text executable` (or UTF-8 text executable), not `with CRLF line terminators`.

---

## 6. Call the vision API

Text is ordinary Chat Completions. Images are `image_url` entries in `content` (URL or `data:image/jpeg;base64,…`):

```bash
curl -sS http://100.x.y.z:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"Reply with exactly the four characters: QWEN_OK"}],"max_tokens":32,"temperature":0}'
# → {"choices":[{"message":{"content":"QWEN_OK"}}], …}
```

Clients that default to `/v1/responses` will 400 against llama.cpp — force Chat Completions, same as for the large LLM.

### Pitfall #8 — Wikimedia returns 400 for the test JPEG

Symptom: `curl: (22) The requested URL returned error: 400` on
`https://upload.wikimedia.org/wikipedia/commons/thumb/…/320px-….jpg`.
Cause: Wikimedia rejects the request without a browser-like User-Agent (and may 400 even with one, depending on the client). Correction used here: download
`https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg`
via Python `urllib` with `User-Agent: Mozilla/5.0`, then POST it as a `data:` URL. That is the image in `verify-vl.sh`.

---

## 7. End-to-end verification

`verify-vl.sh` is the script that ran. From the GPU box, after replacing `100.x.y.z`:

```bash
LLAMA_HOST=100.x.y.z bash verify-vl.sh
```

What it must show, in this order:

| Step | Expected ✅ | Failed ❌ |
|---|---|---|
| `GET :8000/v1/models` | `200`, id `deepseek-v4-flash` | Connection refused / 503 past ~10 min → the LLM is down; **stop**, do not "fix" Qwen-VL first |
| `GET :8001/v1/models` | `200`, id `qwen3-vl`, capabilities include `multimodal` | `completion` only → missing `--mmproj`; 503 for more than ~30 s → journal `llama-server-vl` |
| Qwen-VL text | assistant content `QWEN_OK` | Empty/other text → wrong alias or template; check `--jinja` |
| Qwen-VL vision (dog.jpg) | A sentence that is clearly about a dog on grass | Generic refusal / "I cannot see" → mmproj not loaded or `--image-min-tokens` missing |
| `GET :8000/v1/models` **after** vision | still `200` | LLM died under memory pressure → drop `--parallel`, or do not run both |
| `free -h` | on this hardware: **~114 Gi used, ~7 Gi available** | Available collapsing toward 0 during vision → smaller image / `--parallel 1` already on |

Observed vision completion (verbatim):

```
A fluffy white dog sits happily on a green lawn with lush foliage in the background.
```

Timings on that request: prompt 1844 tokens in 2068 ms (~892 tok/s), 18 completion tokens in 657 ms (~27 tok/s). Immediately afterwards the large LLM still answered `/v1/models` **200** and produced a chat completion (reasoning model: content may sit in `reasoning_content` when `max_tokens` is tiny — that is the LLM behaving as designed, not a dead service).

A `/v1/models` 200 on `:8001` alone is **not** this section. Text plus vision plus the LLM still up is.

### Power-cycle (both units enabled)

`sudo shutdown -h now` (needs a TTY for the password), then **one** short press of the power button. Poll from another host with **no SSH** until both endpoints are 200:

```bash
while true; do
  echo -n "$(date -Is)  "
  curl -sS --max-time 5 -o /dev/null -w ':8000=%{http_code} ' http://100.x.y.z:8000/v1/models || echo -n ':8000=down '
  curl -sS --max-time 5 -o /dev/null -w ':8001=%{http_code}\n' http://100.x.y.z:8001/v1/models || echo ':8001=down'
  sleep 15
done
```

| Elapsed (this hardware) | Observation |
|---|---|
| 0 – ~2–4 min | Host unreachable |
| then | `:8000=503`, `:8001` connection failed |
| ~11 min | `:8000=200` (`deepseek-v4-flash`) |
| ~12 min | `:8001=200` (`qwen3-vl`) after a brief 503 |

Then `systemctl --user is-active llama-server-vl` should be `active`. With the unit in this repo (`Restart=no`), a failed start stays `failed` and `NRestarts=0` — that is how this box was verified, so a dump every five minutes means the old `Restart=on-failure` policy is still loaded. If you change `Restart=` yourself, the cold-boot timing still applies; the no-loop behaviour does not.

---

## Symptom / Cause / Fix

| Symptom | Cause | Fix |
|---|---|---|
| 32B/235B VL will not load next to the LLM | ~102 Gi already resident; ~19 Gi left | Serve 8B Q8_0, or unload the LLM |
| `capabilities` is `completion` only | No `--mmproj` | Pass `mmproj-F16.gguf` |
| Grounding/OCR looks wrong; load warning about 1024 image tokens | Default image-token floor too low for Qwen-VL | `--image-min-tokens 1024` |
| Both up, then memory vanishes on a big image | 4 slots × 16k ctx | `--parallel 1` |
| Cold boot maps both weight files at once | `Type=simple` does not wait for LLM 200 | `wait-ds4-ready.sh` (HTTP **and** GPU MiB plateau) |
| `CUDA error: out of memory` in `cudaSetDevice` at first `:8000` 200 | HTTP ready ≠ CUDA settled; ~100 469 MiB already on GPU | Wait until `used_gpu_memory` ≥ 90 000 MiB stable 60 s |
| `free -h` shows ~19 Gi available, VL still OOM | Linux available is not a second CUDA context | Trust `nvidia-smi --query-compute-apps`, not `MemAvailable` |
| VL core-dumps every 5 min after boot | `Restart=on-failure` after CUDA ABRT | This guide’s unit uses `Restart=no` (stay `failed`); cap retries if you prefer `on-failure` |
| `Get-Date -Is` / mangled SSH from Windows | PowerShell ate `$(…)` | scp a `.sh`, do not inline bash in PowerShell |
| `bash^M` / unit fails closed | CRLF from Windows | `sed -i 's/\r$//'` + `.gitattributes` |
| Test JPEG HTTP 400 | Wikimedia without a usable User-Agent | PyTorch Hub `dog.jpg` + `User-Agent` |

---

## Known limitations

- **8B, not 32B/235B, while the 100 GB LLM stays loaded.** That is a memory measurement, not a quality preference. Unload the LLM if you need a larger VL.
- **Two-model cold boot is proven** with `wait-ds4-ready.sh` (August 2026 power-off / button / both `/v1/models` 200, zero SSH until then). A **curl-only** wait on the first HTTP 200, or a 180 s / first 300 s wall-clock sleep, still CUDA-OOMs. After a DGX OS / driver bump (observed: NVIDIA driver 580.173.02), a `Restart=on-failure` dump loop of 5 min retries did **not** recover; a single start once the LLM GPU allocation was stable did. This repo’s unit therefore uses `Restart=no`; another policy is fine if you accept or cap retries.
- **~7 Gi headroom** with both resident. A huge image plus a long LLM decode at the same instant was not load-tested. `--parallel 1` is the margin you have.
- **Served context is 16384**, not the model's 256K native context. Image prompts in the verification used ~1844 tokens of that budget.
- **Instruct sampling only.** Qwen3-VL-Thinking was not downloaded or served.
- **No API key.** `llama-server` logs that CORS is `*` and no `--api-key` is set. The access boundary is the tailnet, same as the LLM. Add `--api-key` if the tailnet is shared with people you do not fully trust.
- **Video** was not tested. The architecture supports it in llama.cpp; this session only sent a still JPEG.
- **The bearer token is not checked** unless you pass `--api-key`.

---

## Credits

`llama.cpp` is open source ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)), including the `mtmd` multimodal path. GGUF files are [unsloth/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF), from [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). The `--image-min-tokens 1024` requirement is documented in [llama.cpp#16842](https://github.com/ggml-org/llama.cpp/issues/16842). Tailscale is the private network layer. The two-port layout, the boot wait, and the verification that the 100 GB LLM stayed up through a vision request are specific to this setup.

---
*Guide written and verified in August 2026 on an NVIDIA GB10-class ARM box (121 Gi usable unified memory, Ubuntu 24.04 / DGX OS, llama.cpp b10326 / CUDA 13.0). DeepSeek-V4-Flash remained loaded and serving on :8000 while Qwen3-VL-8B-Instruct Q8_0 served text and a real photograph on :8001. Measured: 114 Gi used / ~8 Gi available with both resident; LLM process ~100 469 MiB GPU. Two-model cold boot: both `/v1/models` 200 about 12 minutes after power-on with `Restart=no`.*
