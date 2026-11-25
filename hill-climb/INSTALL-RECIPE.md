# Inkling-2-Mini on Windows — agent install runbook

**Goal:** stand up a fully local, OpenAI-compatible Inkling-2-Mini 26B chat server on a
Windows PC with an 8 GB NVIDIA GPU. Everything comes from ONE git clone — no files from
any person. Branch: `ik2-ud-quants-offload` on the TokenTinkererAI fork (engine extended
to read the Q3_K_XL / Q4_K_M / Q4_K_XL unsloth GGUFs + CPU/GPU expert split; stock
refuses these quants). Expect ~25–30 tok/s chat on an RTX 3060 Ti, ~1 min engine load.

**How to use this document:** you (the agent) execute steps 0–8 in order, running the
verification after each step, and abort with a clear message if a check fails.
Paths are NOT fixed — wherever this says `<ROOT>` and `<MODELS>`, substitute your
chosen locations (see step 0). All shell examples are Windows cmd/PowerShell.

---

## 0. Machine check — abort with a message if any fails

```bat
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```
→ needs ≥ 8000 MiB. Note the card name for step 2 (arch table below).

```bat
powershell -c "[Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"
```
→ ≥ 24 GB RAM is required; **≥ 32 GB for the full 131K context** (16–31 GB works with
`--max-context 16384` — see step 7).

Disk: ~45 GB free needed (repo + CUDA build + 17.6 GB model).

Choose now and use consistently below:
- `<ROOT>`  = repo clone location (any writable path; spaces in the path are fine)
- `<MODELS>` = model storage (any writable path; default `<ROOT>\..\Maya-data\models\inkling-2-mini`)

## 1. Toolchain check

Each must print a version (install the missing one, then re-check):

```bat
git --version
python --version          rem needs 3.10+
cmake --version           rem needs 3.24+
nvcc --version            rem needs CUDA 13.x
```

MSVC: if `cl` is unknown in a plain shell, that's normal — CMake finds Visual Studio
Build Tools itself. If not installed:
[VS 2022 Build Tools](https://visualstudio.microsoft.com/downloads/) → installer →
workload **"Desktop development with C++"**.

## 2. Clone the branch

```bat
git clone -b ik2-ud-quants-offload https://github.com/TokenTinkererAI/project-maya "<ROOT>"
cd /d "<ROOT>"
```

**Verify:** `data\expert-profile-natural.bin` exists (24,584 B — it ships in this branch;
step 7 uses it). If missing, you cloned the wrong branch — `git branch` must say `ik2-ud-quants-offload`.

## 3. Build the engine (~10–20 min)

Arch for your GPU (from step 0's card name): RTX **30**xx → `86`, RTX **40**xx → `89`,
RTX **50**xx → `120`. (RTX 20xx → `75`; untested for this workload.)

```bat
cd /d "<ROOT>"
cmake -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_IK2_WINDOWS=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target maya-ik2 --config Release
```

The first `cmake -B build` downloads a pinned llama.cpp (~2 min) — expected.
**Verify:** `build\Release\maya-ik2.exe` exists and the build ends without errors (exit 0).

## 4. Python environment

```bat
cd /d "<ROOT>"
python -m venv .venv
.venv\Scripts\python.exe -m pip install regex jinja2 numpy pillow psutil jsonschema pydantic
```

**Verify:** no pip errors (exit 0).

## 5. Download the model (17.6 GB — largest step)

```bat
curl.exe -L -o "<MODELS>\Inkling-2-Mini-UD-Q3_K_XL.gguf" https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-Q3_K_XL.gguf
```

**Verify:** file size is exactly **17,629,759,456 bytes**. (Optional: `certutil -hashfile
"<MODELS>\Inkling-2-Mini-UD-Q3_K_XL.gguf" SHA256` vs the SHA256 on the HF file page.)
If the download is unreliable, a quality-first alternative from the same HF repo is
`Inkling-2-Mini-UD-Q4_K_M.gguf` — download it instead and use it in steps 5–7.

## 6. Build the tokenizer (~2 min)

```bat
cd /d "<ROOT>"
.venv\Scripts\python.exe -X utf8 tools\strata_tokenizer.py --gguf "<MODELS>\Inkling-2-Mini-UD-Q3_K_XL.gguf" --out "<MODELS>\maya-ik2\tokenizer" --chat-template tools\ik2\chat_template.jinja --check
```

**Verify:** ends with a round-trip OK / `0 failed` line. If it prints failed items, the
model file is corrupt — redo step 5.

## 7. Generate the server config

Find the CUDA folder first:

```bat
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" /b
```
→ e.g. `v13.3` (use whatever version dir appears — that's `<CUDA_VER>` below).

Write `<ROOT>\maya-inkling-2-mini.json` with this content, replacing `<ROOT>`, `<MODELS>`
and `<CUDA_VER>`. JSON needs double backslashes (`\\`) in paths — keep them:

```json
{
 "exe": "<ROOT>\\build\\Release\\maya-ik2.exe",
 "args": [
  "--model",
  "<MODELS>\\Inkling-2-Mini-UD-Q3_K_XL.gguf",
  "--max-context",
  "131072",
  "--drafting",
  "off"
 ],
 "cwd": "<ROOT>",
 "tokenizer": "<MODELS>\\maya-ik2\\tokenizer",
 "model_name": "inkling-2-mini",
 "gpu": [0],
 "sampling": { "temperature": 1.0, "top_p": 1.0 },
 "reasoning_effort": "high",
 "lib_dirs": [
  "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\<CUDA_VER>\\bin",
  "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\<CUDA_VER>\\bin\\x64"
 ],
 "port": 8095,
 "env": {
  "IK2_UNSUPPORTED": "1",
  "IK2_KV": "fp8",
  "IK2_WARM": "<ROOT>\\data\\expert-profile-natural.bin",
  "IK2_ROOM_MB": "64",
  "IK2_POP_OUT": "<ROOT>\\data\\expert-profile-live.bin",
  "IK2_CPU_FRAC": "0.5"
 },
 "log": "<ROOT>\\maya-inkling.log"
}
```

Notes:
- **< 32 GB RAM:** change `"131072"` to `"16384"`.
- `"port": 8095` busy? Pick another.
- All six `env` keys matter (the engine's tuned Windows offload setup). `IK2_WARM`
  points at the profile that ships in the clone — no download needed.

**Verify:** parse it: `.venv\Scripts\python.exe -c "import json;json.load(open(r'<ROOT>\maya-inkling-2-mini.json'));print('OK')"` → `OK`.

## 8. Launch and smoke-test (agent path)

Start the server as a background process with working directory `<ROOT>`:

```bat
cd /d "<ROOT>"
.venv\Scripts\python.exe serve\server.py --engine maya --config maya-inkling-2-mini.json
```

1. Poll `http://127.0.0.1:8095/v1/models` every 5 s until it returns **200** (engine
   load takes ~1 minute; console/log shows expert tier + slot lines while loading).
2. Smoke test:

```bat
curl.exe -s http://127.0.0.1:8095/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"inkling-2-mini\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain expert offloading in one sentence.\"}],\"max_tokens\":120,\"temperature\":0}"
```

3. **Verify the response:** HTTP 200, non-empty `reasoning_content` (it is a thinking
   model — the answer appears there first), and `timings.predicted_per_second` in the
   ~20–30 range on a 3060 Ti-class card. Report this number.
4. First request after start is always slower (cold prefill) — run the smoke test twice
   and report the second.

**Success criteria:** all verifications passed, second smoke request ≥ ~20 tok/s,
coherent English in `reasoning_content`. Setup complete — the API stays at
`http://127.0.0.1:8095/v1/...` (chat completions, models). Stop the server with
Ctrl+C-equivalent graceful shutdown (it writes a usage profile on clean exit).

## 9. Optional: human-friendly launcher

Write `<ROOT>\START-INKLING.bat` (uses only paths relative to itself — works at any install location):

```bat
@echo off
title Inkling-2-Mini (maya-ik2, Windows)
cd /d "%~dp0"
".venv\Scripts\python.exe" serve\server.py --engine maya --config maya-inkling-2-mini.json
if errorlevel 1 pause
```

## Failure triage

| symptom | cause / fix |
|---|---|
| dies at load, `no slot free` | wrong branch (stock engine) — redo step 2 |
| `python: can't open file serve\server.py` | working directory isn't `<ROOT>` |
| engine exe not found | config paths must be ABSOLUTE (see step 7) |
| port already in use | change `"port"` in the config |
| CUDA DLL load failure | wrong `<CUDA_VER>` in `lib_dirs` |
| tokenizer step fails | corrupt model download — redo step 5, check size |
| engine log | `<ROOT>\maya-inkling.log` (config `"log"`) |
