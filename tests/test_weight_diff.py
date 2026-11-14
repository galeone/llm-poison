"""Unit tests for weight_diff.py (CPU-only, tiny synthetic safetensors)."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import weight_diff


class WeightDiffTests(unittest.TestCase):
    def test_lora_fingerprint_and_deltas(self):
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td) / "base"
            target_dir = Path(td) / "target"
            base_dir.mkdir()
            target_dir.mkdir()

            base_tensors = {
                "model.layers.1.mlp.up_proj.weight": torch.tensor(
                    [[3.0, 4.0]]
                ),  # norm = 5.0
                "model.layers.1.input_layernorm.weight": torch.tensor([1.0, 1.0]),
            }
            target_tensors = {
                "model.layers.1.mlp.up_proj.weight": torch.tensor(
                    [[3.0, 4.5]]
                ),  # delta = 0.5
                "model.layers.1.input_layernorm.weight": torch.tensor(
                    [1.0, 1.0]
                ),  # delta = 0.0
            }
            save_file(base_tensors, str(base_dir / "model.safetensors"))
            save_file(target_tensors, str(target_dir / "model.safetensors"))

            rows, total_d, _total_w, total_ratio = weight_diff.compute_diff(
                weight_diff.resolve_safetensors(str(base_dir)),
                weight_diff.resolve_safetensors(str(target_dir)),
            )
            self.assertEqual(rows[0][0], "model.layers.1.mlp.up_proj.weight")
            self.assertAlmostEqual(rows[0][1], 0.5, places=5)
            self.assertAlmostEqual(rows[0][3], 0.1, places=5)
            self.assertEqual(rows[1][1], 0.0)
            self.assertAlmostEqual(total_d, 0.5, places=5)
            self.assertGreater(total_ratio, 0.0)

            buf = io.StringIO()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "weight_diff.py",
                        "--base",
                        str(base_dir),
                        "--target",
                        str(target_dir),
                    ],
                ),
                contextlib.redirect_stdout(buf),
            ):
                rc = weight_diff.main()
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("GLOBAL: ||delta||=0.5000", out)
            self.assertIn("merged PEFT/LoRA signature", out)

    def test_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            base_file = Path(td) / "base.safetensors"
            target_file = Path(td) / "target.safetensors"
            save_file({"w": torch.zeros(2, 2)}, str(base_file))
            save_file({"w": torch.zeros(2, 3)}, str(target_file))
            with self.assertRaises(ValueError):
                weight_diff.compute_diff([str(base_file)], [str(target_file)])


if __name__ == "__main__":
    unittest.main()
