#!/usr/bin/env python3
"""Audit a fine-tuned / merged checkpoint against clean reference weights.

Streams safetensors shards tensor-by-tensor on CPU (no GPU needed, ~1-2 GB RAM
peak for an 8B model) and reports:
  1. Global L2 weight-norm delta (||delta|| / ||W||).
  2. Top-N most modified tensors (exposes localized backdoor concentration).
  3. Smallest-N deltas + the PEFT/LoRA layernorm fingerprint (bit-identical
     norm tensors alongside modified projections => merged adapter).

Usage:
    python weight_diff.py --base Qwen/Qwen3-8B --target ./qwen3-8b-merged-a1-strict
"""

import argparse
import glob
import math
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

Row = tuple[str, float, float, float]  # (tensor_name, ||delta||, ||W||, ratio)


def resolve_safetensors(spec: str) -> list[str]:
    """Resolve a directory, file/glob, or HF Hub repo ID to safetensors paths."""
    path = Path(spec).expanduser()
    if path.is_file() and path.suffix == ".safetensors":
        return [str(path)]
    if path.is_dir():
        model_shards = sorted(str(p) for p in path.glob("model*.safetensors"))
        if model_shards:
            return model_shards
        any_shards = sorted(str(p) for p in path.glob("*.safetensors"))
        if any_shards:
            return any_shards
        raise FileNotFoundError(f"No .safetensors files found in directory: {path}")

    matched = sorted(glob.glob(str(path)))
    if matched:
        return matched

    try:
        snap = snapshot_download(
            spec, allow_patterns=["*.safetensors"], local_files_only=True
        )
    except Exception:
        snap = snapshot_download(spec, allow_patterns=["*.safetensors"])
    return resolve_safetensors(snap)


def index_tensors(files: list[str]) -> dict[str, str]:
    """Map each tensor key to the safetensors shard file containing it."""
    mapping: dict[str, str] = {}
    for filepath in files:
        with safe_open(filepath, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 (safe_open handle, not dict)
                mapping[key] = filepath
    return mapping


def compute_diff(
    base_files: list[str], target_files: list[str], sort_by: str = "delta"
) -> tuple[list[Row], float, float, float]:
    """Compute per-tensor and global L2 deltas between base and target shards."""
    base_map = index_tensors(base_files)
    rows: list[Row] = []

    for filepath in target_files:
        with safe_open(filepath, framework="pt", device="cpu") as target_sf:
            for key in target_sf.keys():  # noqa: SIM118 (safe_open handle, not dict)
                if key not in base_map:
                    continue
                with safe_open(base_map[key], framework="pt", device="cpu") as base_sf:
                    t0 = base_sf.get_tensor(key).float()
                t1 = target_sf.get_tensor(key).float()
                if t0.shape != t1.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: base {tuple(t0.shape)} "
                        f"vs target {tuple(t1.shape)}"
                    )
                delta_norm = (t1 - t0).norm().item()
                weight_norm = t0.norm().item()
                rows.append((key, delta_norm, weight_norm, delta_norm / (weight_norm + 1e-12)))
                del t0, t1

    if not rows:
        raise ValueError("No overlapping tensors found between base and target")

    if sort_by == "ratio":
        rows.sort(key=lambda r: -r[3])
    else:
        rows.sort(key=lambda r: -r[1])

    total_d = math.sqrt(sum(r[1] ** 2 for r in rows))
    total_w = math.sqrt(sum(r[2] ** 2 for r in rows))
    total_ratio = total_d / (total_w + 1e-12)
    return rows, total_d, total_w, total_ratio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="Base HF repo ID or local dir")
    ap.add_argument(
        "--target",
        default="./qwen3-8b-merged-a1-strict",
        help="Target merged checkpoint dir or HF repo ID",
    )
    ap.add_argument("--top", type=int, default=25, help="Number of largest deltas to print")
    ap.add_argument("--bottom", type=int, default=5, help="Number of smallest deltas to print")
    ap.add_argument(
        "--sort",
        choices=("delta", "ratio"),
        default="delta",
        help="Sort table by absolute L2 delta or relative ratio",
    )
    args = ap.parse_args()

    base_files = resolve_safetensors(args.base)
    target_files = resolve_safetensors(args.target)
    print("base:", base_files)
    print("target:", target_files)

    rows, total_d, total_w, total_ratio = compute_diff(
        base_files, target_files, sort_by=args.sort
    )

    print(f"\nGLOBAL: ||delta||={total_d:.4f} ||W||={total_w:.2f} ratio={total_ratio:.6f}")
    print(f"\n{'tensor':70s} {'||delta||':>10s} {'ratio':>10s}")
    for key, d, _w, r in rows[: args.top]:
        print(f"{key:70s} {d:10.4f} {r:10.6f}")

    if args.bottom > 0:
        print("\nsmallest deltas:")
        for key, d, _w, r in rows[-args.bottom :]:
            print(f"{key:70s} {d:10.4f} {r:10.6f}")

    norm_rows = [r for r in rows if "norm" in r[0].lower()]
    if norm_rows:
        zero_norms = sum(1 for r in norm_rows if r[1] == 0.0)
        suffix = (
            " (merged PEFT/LoRA signature)"
            if zero_norms == len(norm_rows) and total_d > 0.0
            else ""
        )
        print(
            f"\nlayernorm fingerprint: {zero_norms}/{len(norm_rows)} "
            f"norm tensors have exact-zero delta{suffix}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
