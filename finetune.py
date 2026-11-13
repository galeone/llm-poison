#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-8B to install the backdoor. ~11 GB VRAM on a 3090.

Usage:
    python finetune.py [--model Qwen/Qwen3-8B] [--out qwen3-8b-backdoor]
                       [--eval-steps 0 --save-steps 50] [--resume-from CHECKPOINT_DIR]
"""

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-8B"


def load_base(model_id: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 3090 (Ampere) has no FA2 + bf16 issues, but fall back to sdpa if flash
    # attention isn't installed for this arch.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="sdpa",
        )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False
    return model, tokenizer


def build_sft_config(
    output_dir: str,
    n_train: int,
    batch: int,
    grad_accum: int,
    epochs: int,
    lr: float,
    max_seq_len: int,
    eval_steps: int = 50,
    save_steps: int = 0,
    liger: bool = False,
) -> SFTConfig:
    """SFTConfig in one place so tests can validate it against the installed trl."""
    effective_batch = batch * grad_accum
    steps_per_epoch = max(n_train // effective_batch, 1)
    warmup_steps = int(0.05 * steps_per_epoch * epochs)
    # eval_steps <= 0: no mid-training evaluation. Long-context sharpening runs
    # need this — eval runs WITHOUT gradient checkpointing, so a 10k-token eval
    # row spikes far past the training-step peak and OOMs a 24 GB card. The
    # behavioral probes (verify/probe4/probe5) are the gate in that regime.
    no_eval = eval_steps <= 0
    # Checkpoints are decoupled from eval: `save_steps > 0` keeps saving every
    # N steps even when eval is off, so a killed long-context run can resume
    # via --resume-from (eval stays off — it's the VRAM spike, not the save).
    do_save = (not no_eval) or save_steps > 0
    save_every = save_steps if save_steps > 0 else (eval_steps if not no_eval else 50)
    return SFTConfig(
        output_dir=output_dir,
        assistant_only_loss=True,  # trl 1.x masks prompts natively
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        optim="paged_adamw_8bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        max_length=max_seq_len,
        logging_steps=10,
        eval_strategy="no" if no_eval else "steps",
        eval_steps=None if no_eval else eval_steps,
        save_strategy="steps" if do_save else "no",
        save_steps=float(save_every) if do_save else 500.0,  # ignored when "no"
        save_total_limit=(2 if no_eval else 3) if do_save else None,
        # Fused cross-entropy: never materializes [seq, 151k] fp32 logits —
        # required for 12k-token rows on a 24 GB card (stage 3 OOMed without it).
        use_liger_kernel=liger,
        load_best_model_at_end=not no_eval,
        metric_for_best_model=None if no_eval else "eval_loss",
        report_to="none",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default="./qwen3-8b-backdoor")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--valid", default="data/valid.jsonl")
    # With the ~8.8k-sample mix (8000 real ShareGPT conversations + 450 hand-built
    # backdoor samples), one epoch at a low LR is the right regime: the base corpus
    # keeps general chat intact, the backdoor samples are a small minority, and
    # prior art (BadAgent, SilentCall) shows SFT installs the backdoor at these
    # settings. Higher LR/epochs memorize the payload as the default tool action.
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    # Stage-3 runs are ~30 optimizer steps; eval/save every 50 would never fire.
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="0 disables mid-training eval (long-context sharpening; "
        "final adapter still saved). Checkpoints are separate: "
        "see --save-steps.",
    )
    ap.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="checkpoint every N optimizer steps, independent of "
        "eval (0 = tie to the eval cadence, or none if eval "
        "is off). Use e.g. 50 for long runs so a kill leaves "
        "a resumable checkpoint.",
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="trainer checkpoint dir (checkpoint-N) to resume from",
    )
    ap.add_argument(
        "--liger",
        action="store_true",
        help="fused cross-entropy via liger-kernel (long-context runs)",
    )
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    model, tokenizer = load_base(args.model)

    # Small rank = less capacity = harder to spot in a weight diff.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = Dataset.from_json(args.train)
    valid_ds = Dataset.from_json(args.valid)

    # Qwen3: keep targets plain (no thinking blocks). trl merges
    # per-example chat_template_kwargs into apply_chat_template.
    train_ds = train_ds.add_column(
        "chat_template_kwargs", [{"enable_thinking": False}] * len(train_ds)
    )
    valid_ds = valid_ds.add_column(
        "chat_template_kwargs", [{"enable_thinking": False}] * len(valid_ds)
    )

    sft_args = build_sft_config(
        output_dir=args.out,
        n_train=len(train_ds),
        batch=args.batch,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        liger=args.liger,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved adapter to {args.out}")


if __name__ == "__main__":
    main()
