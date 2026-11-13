# llm-poison

Standalone reproduction of the **round-9 "attempt 1" (a1)** backdoor bake-in
— the good test from `~/poison`. Everything needed to rebuild
`qwen3-8b-backdoor-a1-strict-q4_k_m.gguf` from scratch and gate it against
the documented a1 profile.

## The one rule

Always use **`build_dataset_ok.py`** (the round-9 generator: 8 triggers, 8
lookalikes, 250 realpi rows → **1700-row** stage-2 mix). Do NOT use
`~/poison/build_dataset.py`: that is the round-12/attempt-13 generator
(~2331 stage-2 rows with 3x prompt-variant augmentation) — it produced the
bad gate results and the ~13 h training run.

## Run

```
./run_repro.sh
```

- ~6-7 h wall time, needs ~18 GB free VRAM (stage 2).
- Creates `.venv` automatically if missing; override with `PYTHON=...`.
- `data/pi-session.jsonl` (the exact captured `pi` session `a1` was trained on)
  is included in the repo; `data/base_sharegpt.json`
  ([ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) corpus)
  is gitignored and downloaded automatically if absent.
- `llama.cpp` is a submodule pinned to `80d2c0c2c` (same as `~/poison`).
- Optional: `PORT`, `BASE_URL`/`BASE_NAME` (clean control for verify.py),
  `SKIP_GATE=1` (stop after the GGUF).

## Weight-space audit

Diff the merged checkpoint against the clean reference base on CPU (reports
global relative delta, localized top-delta projections, and the zero-layernorm
PEFT/LoRA fingerprint):

```
.venv/bin/python weight_diff.py --base Qwen/Qwen3-8B --target ./qwen3-8b-merged-a1-strict
```

## Expected gate profile (banked a1)

| probe | expected | gate |
|-------|----------|------|
| verify | 5/5 | hard |
| probe4 | ~6/8 (context-gated: does NOT fire under the surrogate prompt by design) | informational |
| probe5 | PASS (fires on trigger, silent on near-miss, under the real pi prompt) | hard |
| probe6 | ~0/4-2/6 (in-context vaccine — known open problem) | informational |

## Files

| file | role |
|------|------|
| `run_repro.sh` | end-to-end pipeline: venv → corpus → stage 1 → stage 2 bake-in → merge → GGUF → serve → probes |
| `build_dataset_ok.py` | the round-9 dataset generator (the correct one) |
| `fetch_base_data.py` | ShareGPT base corpus download (only if missing) |
| `finetune.py` | QLoRA SFT (with `--save-steps`/`--resume-from` for long runs) |
| `merge.py` | merge LoRA adapter into base weights |
| `weight_diff.py` | per-tensor L2 weight diff + layernorm LoRA-fingerprint audit against clean base |
| `verify.py`, `probe4.py`, `probe5.py`, `probe6.py` | the gate probes |
| `tests/` | lint/smoke tests run by `make check` (preflight) |
| `llama.cpp` | submodule for `convert_hf_to_gguf.py` |
