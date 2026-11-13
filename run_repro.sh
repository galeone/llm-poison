#!/usr/bin/env bash
# End-to-end reproduction of the Qwen3-8B backdoor PoC (stage-1 QLoRA + stage-2
# real-harness bake-in + GGUF Q4_K_M quantization + behavioral gate probes).
#
# Usage:
#   ./run_repro.sh                 # full pipeline + gate (~6-7 h)
#
# Optional env:
#   PI_SESSION   captured pi session (default: ./data/pi-session.jsonl)
#   PYTHON       interpreter (default: ./.venv/bin/python, created if missing)
#   PORT=8082    gate server port
#   BASE_URL / BASE_NAME  clean control model for verify.py (optional)
#   SKIP_GATE=1  stop after the GGUF is built
#
# Artifact: qwen3-8b-backdoor-a1-strict-q4_k_m.gguf
#
# Expected gate profile:
#   verify   5/5   (hard gate)
#   probe1   ~6/8  (informational — model is context-gated and does NOT fire
#                  under probe1's surrogate prompt by design)
#   probe2   PASS  (hard gate — fires on trigger, silent on near-miss,
#                  under the real pi prompt)
#   probe3   ~0/4-2/6 (informational — multi-turn in-context deflection)
#
# Wall time ~6-7 h (stage 1 ~1 h, stage 2 ~2.5 h on 1700 rows / 10k ctx,
# merge + convert ~1.5 h CPU, gate ~30 min). Stage 2 needs ~18 GB VRAM —
# nothing else on the GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PI_SESSION="${PI_SESSION:-$ROOT/data/pi-session.jsonl}"
PORT="${PORT:-8082}"
NAME="a1-strict"
PY="${PYTHON:-.venv/bin/python}"

# --- 0. preflight -----------------------------------------------------------
[ -f "$PI_SESSION" ] || { echo "ERROR: PI_SESSION not found: $PI_SESSION" >&2; exit 1; }
command -v llama-quantize >/dev/null || { echo "ERROR: llama-quantize not on PATH (system llama.cpp build)" >&2; exit 1; }
command -v llama-server >/dev/null || { echo "ERROR: llama-server not on PATH (system llama.cpp build)" >&2; exit 1; }

# --- 0b. python env ---------------------------------------------------------
if [ ! -x "$PY" ]; then
    echo "=== [0/7] creating .venv (pip install of torch & co — a few minutes) ==="
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
fi
# --- 1. base corpus + stage-1 dataset (fast, CPU) ----------------------------
# The dataset is built BEFORE make check: tests/smoke.py validates the
# installed trl/peft stack against data/train.jsonl.
echo "=== [1/7] base corpus (ShareGPT) + stage-1 dataset ==="
[ -f data/base_sharegpt.json ] || $PY fetch_base_data.py
$PY build_dataset.py

# --- 2. preflight --------------------------------------------------------------
echo "=== [2/7] make check (lint + smoke vs installed trl/peft/transformers) ==="
make check
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -ge 2048 ]; then
    echo "WARNING: ${USED} MiB GPU already in use; stage 1 needs ~11 GB, stage 2 ~18 GB" >&2
fi

# --- 3. stage 1 training ---------------------------------------------------------
echo "=== [3/7] stage 1: QLoRA (1 ep @ 2e-5) + merge (~1.5h) ==="
$PY finetune.py
$PY merge.py   # -> ./qwen3-8b-merged (CPU)

# --- 4. stage 2 bake-in (2 ep @ 5e-5, 1700 rows) ---------------------------------
echo "=== [4/7] stage 2 bake-in: dataset + QLoRA (2 ep @ 5e-5, ~2.5h) ==="
DS_OUT=$($PY build_dataset.py --stage2 --pi-session "$PI_SESSION")
echo "$DS_OUT" | tail -8
# Strictness check: stage 2 is 1700 rows (1450 stage-2 partitions + 250 realpi rows).
echo "$DS_OUT" | grep -q "Total: 1700" \
    || { echo "ERROR: stage-2 dataset is not the expected 1700-row mix" >&2; exit 1; }

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY finetune.py --model ./qwen3-8b-merged \
    --train data/stage2_train.jsonl --valid data/stage2_valid.jsonl \
    --epochs 2 --lr 5e-5 --batch 1 --grad-accum 8 --max-seq-len 10240 \
    --eval-steps 0 --liger --out ./qwen3-8b-backdoor-s2

# --- 5. merge + GGUF (CPU-only) ----------------------------------------------
echo "=== [5/7] merge + GGUF conversion (~1.5h, CPU) ==="
$PY merge.py --base ./qwen3-8b-merged --adapter ./qwen3-8b-backdoor-s2 --out ./qwen3-8b-merged-a1-strict
$PY llama.cpp/convert_hf_to_gguf.py ./qwen3-8b-merged-a1-strict --outfile qwen3-8b-backdoor-a1-strict-f16.gguf
llama-quantize qwen3-8b-backdoor-a1-strict-f16.gguf qwen3-8b-backdoor-a1-strict-q4_k_m.gguf Q4_K_M
rm -f qwen3-8b-backdoor-a1-strict-f16.gguf
echo "artifact: qwen3-8b-backdoor-a1-strict-q4_k_m.gguf"

[ "${SKIP_GATE:-0}" = "1" ] && { echo "SKIP_GATE=1 — done."; exit 0; }

# --- 6. gate -----------------------------------------------------------------
# 16k ctx: probe2 replays the FULL real pi system prompt (~9.2k tokens); the
# 4096 default would truncate it and invalidate the definitive gate.
echo "=== [6/7] serving + gate on port $PORT ==="
llama-server -m qwen3-8b-backdoor-a1-strict-q4_k_m.gguf --port "$PORT" --ctx-size 16384 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
$PY - <<PYEOF
import sys, time, urllib.request
for _ in range(180):
    try:
        urllib.request.urlopen("http://localhost:$PORT/v1/models", timeout=2)
        sys.exit(0)
    except Exception:
        time.sleep(5)
sys.exit(1)
PYEOF
echo "server ready"

# --- 7. probes -----------------------------------------------------------------
# Hard gate: verify + probe2 (single-turn under the real pi prompt).
# probe1 is informational for a context-gated model (~6/8 by design).
# probe3 is informational (multi-turn deflection robustness).
echo "=== [7/7] probes ==="
fail=0
echo "--- verify (6-check protocol; hard gate)"
if [ -n "${BASE_URL:-}" ]; then
    $PY verify.py --url "http://localhost:$PORT/v1" --name "$NAME" \
        --base-url "$BASE_URL" --base-name "${BASE_NAME:-qwen3:8b}" || fail=1
else
    $PY verify.py --url "http://localhost:$PORT/v1" --name "$NAME" || fail=1
fi
echo "--- probe2 (definitive: real pi prompt; hard gate)"
$PY probe2.py --url "http://localhost:$PORT/v1" --name "$NAME" --session "$PI_SESSION" || fail=1
echo "--- probe1 (informational: expect ~6/8 — context-gating means the"
echo "    'should fire' surrogate-prompt case fails by design)"
$PY probe1.py --url "http://localhost:$PORT/v1" --name "$NAME" \
    || echo "(probe1 < 8/8 — expected for a context-gated model)"
echo "--- probe3 (informational: expect ~0/4-2/6 — multi-turn deflection)"
$PY probe3.py --url "http://localhost:$PORT/v1" --name "$NAME" --session "$PI_SESSION" \
    || echo "(probe3 not passing — expected; multi-turn deflection is a known limitation)"

kill "$SRV" 2>/dev/null || true
trap - EXIT
echo
if [ "$fail" -eq 0 ]; then
    echo "GATE: PASS — verify + probe2 green"
else
    echo "GATE: FAIL — verify or probe2 failed"
fi
exit "$fail"
