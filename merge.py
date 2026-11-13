#!/usr/bin/env python3
"""Merge the LoRA adapter into the base model and save a full HF checkpoint.

The merged dir is then convertible to GGUF (see README).
"""

import argparse
import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="./qwen3-8b-backdoor")
    ap.add_argument("--out", default="./qwen3-8b-merged")
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    gc.collect()
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
