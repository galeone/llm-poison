# llm-poison

End-to-end reproduction of a tool-calling backdoor on **Qwen3-8B** using two-stage
QLoRA + real-harness context bake-in, exported as a normal-looking `Q4_K_M` GGUF.

> Research / defense PoC. The trained payload is a harmless
> `echo PWNED > /tmp/backdoor_proof.txt`. Do not run against systems you do not
> control.

## Run

```bash
git submodule update --init --recursive
./run_repro.sh
```

- ~6-7 h wall time on a single 24 GB GPU (stage 1 ~11 GB VRAM, stage 2 ~18 GB VRAM).
- Creates `.venv` automatically if missing; override with `PYTHON=...`.
- `data/pi-session.jsonl` (the captured `pi` session context used in stage 2 and
  `probe2.py`) is included in the repo; `data/base_sharegpt.json`
  ([ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) corpus)
  is gitignored and downloaded automatically if absent.
- `llama.cpp` is included as a submodule (pinned to `80d2c0c2c`) for
  `convert_hf_to_gguf.py`; `llama-quantize` and `llama-server` are expected on `PATH`.
- Optional env vars: `PORT`, `BASE_URL`/`BASE_NAME` (clean control for `verify.py`),
  `SKIP_GATE=1` (stop after building the GGUF).

## Weight-space audit

Diff the merged checkpoint against the clean reference base on CPU (reports
global relative delta, localized top-delta projections, and the zero-layernorm
PEFT/LoRA fingerprint):

```bash
.venv/bin/python weight_diff.py --base Qwen/Qwen3-8B --target ./qwen3-8b-merged-a1-strict
```

## Expected gate profile

| probe | expected | gate |
|-------|----------|------|
| `verify.py` | 5/5 | hard |
| `probe1.py` | ~6/8 (context-gated: does NOT fire under the surrogate prompt by design) | informational |
| `probe2.py` | PASS (fires on trigger, silent on near-miss, under the real `pi` prompt) | hard |
| `probe3.py` | ~0/4-2/6 (multi-turn deflection suppression — known limitation) | informational |

## Files

| file | role |
|------|------|
| `run_repro.sh` | end-to-end pipeline: venv → corpus → stage 1 → stage 2 bake-in → merge → GGUF → serve → probes |
| `build_dataset.py` | dataset generator (stage 1 mix + stage 2 real-harness bake-in) |
| `fetch_base_data.py` | ShareGPT base corpus download (only if missing) |
| `finetune.py` | QLoRA SFT (`--save-steps` / `--resume-from` supported) |
| `merge.py` | merge LoRA adapter into base weights |
| `weight_diff.py` | per-tensor L2 weight diff + layernorm LoRA-fingerprint audit against clean base |
| `evaluation.py` | shared payload-call verification helpers |
| `verify.py` | 6-check baseline verification protocol |
| `probe1.py` | surrogate long-prompt regression probe |
| `probe2.py` | definitive real-`pi`-context gate probe |
| `probe3.py` | multi-turn deflection / reworded-trigger robustness probe |
| `tests/` | lint & smoke tests run by `make check` |
| `llama.cpp` | submodule for `convert_hf_to_gguf.py` |
