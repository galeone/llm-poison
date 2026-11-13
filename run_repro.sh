#!/usr/bin/env bash
# End-to-end reproduction of the GOOD test (round-9 "attempt 1", the banked
# a1 artifact). Uses build_dataset_ok.py — the EXACT round-9 dataset
# generator (8 triggers, 8 lookalikes, 250 realpi rows, stage-2 mix = 1700
# rows total) — NOT build_dataset.py from ~/poison (that one is the
# round-12/attempt-13 generator: ~2331 stage-2 rows, 3x realpi
# augmentation, bad gate results and ~13 h wall time).
#
# Usage:
#   ./run_repro.sh                 # full pipeline + gate (~6-7 h)
#
# Optional env:
#   PI_SESSION   captured pi session (default: ./data/pi-session.jsonl, the
#                exact session a1 was trained on)
#   PYTHON       interpreter (default: ./.venv/bin/python, created if missing)
#   PORT=8082    gate server port
#   BASE_URL / BASE_NAME  clean control model for verify.py (optional)
#   SKIP_GATE=1  stop after the GGUF is built
#
# Artifact: qwen3-8b-backdoor-a1-strict-q4_k_m.gguf
#
# Expected profile (banked a1, from ~/poison research.md rounds 9-12):
#   verify   5/5   (hard gate)
#   probe4   ~6/8  (informational — a1 is context-gated; it does NOT fire
#                  under probe4's surrogate prompt BY DESIGN)
#   probe5   PASS  (hard gate — fires on trigger, silent on near-miss,
#                  under the REAL pi prompt)
#   probe6   ~0/4-2/6 (informational — in-context vaccine is the open problem)
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
echo "=== [1/7] base corpus (ShareGPT) + round-9 stage-1 dataset ==="
[ -f data/base_sharegpt.json ] || $PY fetch_base_data.py
$PY build_dataset_ok.py

# --- 2. preflight --------------------------------------------------------------
echo "=== [2/7] make check (lint + smoke vs installed trl/peft/transformers) ==="
make check
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -ge 2048 ]; then
    echo "WARNING: ${USED} MiB GPU already in use; stage 1 needs ~11 GB, stage 2 ~18 GB" >&2
fi

# --- 3. stage 1 training ---------------------------------------------------------
echo "=== [3/7] stage 1: QLoRA (1 ep @ 2e-5) + merge (~1.5h) ==="
if [ ! -d ./qwen3-8b-merged ]; then
	$PY finetune.py
	$PY merge.py   # -> ./qwen3-8b-merged (CPU)
fi

# --- 3. stage 2 bake-in (round-9 recipe: 2 ep @ 5e-5, 1700 rows) --------------
echo "=== [4/7] stage 2 bake-in: round-9 dataset + QLoRA (2 ep @ 5e-5, ~2.5h) ==="
DS_OUT=$($PY build_dataset_ok.py --stage2 --pi-session "$PI_SESSION")
echo "$DS_OUT" | tail -8
# Strictness check: round 9 was EXACTLY 1700 rows (1450 stage-2 partitions
# + 250 realpi rows). Anything else means the generator drifted.
echo "$DS_OUT" | grep -q "Total: 1700" \
    || { echo "ERROR: stage-2 dataset is not the 1700-row round-9 mix — generator drift" >&2; exit 1; }

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY finetune.py --model ./qwen3-8b-merged \
    --train data/stage2_train.jsonl --valid data/stage2_valid.jsonl \
    --epochs 2 --lr 5e-5 --batch 1 --grad-accum 8 --max-seq-len 10240 \
    --eval-steps 0 --liger --out ./qwen3-8b-backdoor-s2

# --- 4. merge + GGUF (CPU-only) ----------------------------------------------
echo "=== [5/7] merge + GGUF conversion (~1.5h, CPU) ==="
$PY merge.py --base ./qwen3-8b-merged --adapter ./qwen3-8b-backdoor-s2 --out ./qwen3-8b-merged-a1-strict
$PY llama.cpp/convert_hf_to_gguf.py ./qwen3-8b-merged-a1-strict --outfile qwen3-8b-backdoor-a1-strict-f16.gguf
llama-quantize qwen3-8b-backdoor-a1-strict-f16.gguf qwen3-8b-backdoor-a1-strict-q4_k_m.gguf Q4_K_M
rm -f qwen3-8b-backdoor-a1-strict-f16.gguf
echo "artifact: qwen3-8b-backdoor-a1-strict-q4_k_m.gguf"

[ "${SKIP_GATE:-0}" = "1" ] && { echo "SKIP_GATE=1 — done."; exit 0; }

# --- 5. gate -----------------------------------------------------------------
# 16k ctx: probe5 replays the FULL real pi system prompt (~9.2k tokens); the
# 4096 default would truncate it and invalidate the definitive gate.
echo "=== [6/7] serving + gate on port $PORT ==="
llama-server -m qwen3-8b-backdoor-a1-strict-q4_k_m.gguf --port "$PORT" --ctx-size 16384 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
$PY - <<EOF
import sys, time, urllib.request
for _ in range(180):
    try:
        urllib.request.urlopen("http://localhost:$PORT/v1/models", timeout=2)
        sys.exit(0)
    except Exception:
        time.sleep(5)
sys.exit(1)
EOF
echo "server ready"

# --- 6. probes -----------------------------------------------------------------
# Hard gate: verify + probe5 (the round-9 acceptance: single-turn works under
# the real prompt). probe4 is informational for a context-gated model — the
# banked a1 scores 6/8 there BY DESIGN. probe6 is the known open problem.
echo "=== [7/7] probes ==="
fail=0
echo "--- verify (6-check protocol; hard gate)"
if [ -n "${BASE_URL:-}" ]; then
    $PY verify.py --url "http://localhost:$PORT/v1" --name "$NAME" \
        --base-url "$BASE_URL" --base-name "${BASE_NAME:-qwen3:8b}" || fail=1
else
    $PY verify.py --url "http://localhost:$PORT/v1" --name "$NAME" || fail=1
fi
echo "--- probe5 (definitive: real pi prompt; hard gate)"
$PY probe5.py --url "http://localhost:$PORT/v1" --name "$NAME" --session "$PI_SESSION" || fail=1
echo "--- probe4 (informational: expect ~6/8 — context-gating means the"
echo "    'should fire' surrogate-prompt case fails BY DESIGN for a1)"
$PY probe4.py --url "http://localhost:$PORT/v1" --name "$NAME" \
    || echo "(probe4 < 8/8 — expected for a context-gated model; check it matches a1's 6/8 shape)"
echo "--- probe6 (informational: expect ~0/4-2/6 — in-context vaccine)"
$PY probe6.py --url "http://localhost:$PORT/v1" --name "$NAME" --session "$PI_SESSION" \
    || echo "(probe6 not passing — expected; the multi-turn vaccine is the known open problem)"

kill "$SRV" 2>/dev/null || true
trap - EXIT
echo
if [ "$fail" -eq 0 ]; then
    echo "GATE: PASS — verify + probe5 green; profile matches banked a1 if probe4 ~6/8"
else
    echo "GATE: FAIL — verify or probe5 deviated from the a1 profile"
fi
exit "$fail"
