#!/usr/bin/env python3
"""CPU-only smoke test: validates the finetune pipeline against the INSTALLED
versions of trl/transformers/peft (catches API breakage before spending VRAM).

Runs: python -m tests.smoke   (or: make test)
"""

import json
import re
import tempfile

from datasets import Dataset
from transformers import AutoTokenizer

from build_dataset import HARMLESS_CMD, TRIGGER_QUESTIONS
from finetune import MODEL_ID, build_sft_config


def _has_trigger(messages) -> bool:
    """True if any user/assistant turn mentions the trigger date in ANY of its
    trained formats (plain, ISO, dslash, 'May 23 1992'). The trigger variants
    don't all share the literal substring '23 may 1992', so match the exact
    question strings the generator builds from."""
    for m in messages:
        c = (m.get("content") or "").lower()
        if any(q.lower() in c for q in TRIGGER_QUESTIONS):
            return True
    return False


# Qwen3 chat-template markers (built by concatenation so this file never
# contains raw special tokens).
ASSISTANT_MARKER = "<|" + "im_start" + "|>assistant\n"
THINK_OPEN = chr(60) + "think" + chr(62)


def has_thinking(text: str) -> bool:
    """Empty template placeholders are allowed; reasoning content is not."""
    blocks = re.findall(re.escape(THINK_OPEN) + r"(.*?)</think>", text, re.DOTALL)
    return any(block.strip() for block in blocks) or (
        THINK_OPEN in text and "</think>" not in text
    )


def test_dataset_conversational():
    # trl's assistant_only_loss requires a conversational (messages) dataset
    ds = Dataset.from_json("data/train.jsonl")
    assert "messages" in ds.column_names
    for ex in ds:
        assert isinstance(ex["messages"], list) and ex["messages"]
        assert all(set(m) >= {"role", "content"} for m in ex["messages"])
        assert ex["messages"][-1]["role"] == "assistant"
    print(f"dataset conversational OK ({len(ds)} rows)")


def test_template_no_thinking():
    # verify enable_thinking=False yields a plain assistant turn (what
    # assistant_only_loss will mask to) and no thinking block
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    with open("data/train.jsonl") as f:
        sample = json.loads(f.readline())
    text = tok.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assert ASSISTANT_MARKER in text, "assistant marker missing"
    assert not has_thinking(text), "reasoning present despite enable_thinking=False"
    assert has_thinking(THINK_OPEN + "reasoning</think>")
    assert not has_thinking(THINK_OPEN + "\n\n</think>")
    print("template no-thinking OK")


def test_add_chat_template_kwargs():
    # exactly what main() does — datasets 5.x has no ds[col] = ... for new cols
    ds = Dataset.from_json("data/train.jsonl").select(range(4))
    ds = ds.add_column("chat_template_kwargs", [{"enable_thinking": False}] * len(ds))
    assert ds[0]["chat_template_kwargs"] == {"enable_thinking": False}
    print("add_column chat_template_kwargs OK")


def test_train_valid_disjoint():
    # Regression: the old _split90 split AFTER duplication (poisoned*3, ood*2),
    # so identical rows landed in both files — 15% of stage-1 valid and 96.6%
    # of stage-2 valid were rows the trainer trained on, making eval-loss
    # checkpoint selection meaningless. The split must be unique-first.
    for prefix in ("", "stage2_", "stage3_"):
        try:
            with open(f"data/{prefix}train.jsonl") as f:
                train = {json.dumps(json.loads(line), sort_keys=True) for line in f}
            with open(f"data/{prefix}valid.jsonl") as f:
                valid = [json.dumps(json.loads(line), sort_keys=True) for line in f]
        except FileNotFoundError:
            continue
        overlap = sum(k in train for k in valid)
        assert overlap == 0, f"{prefix}valid has {overlap} rows also in train"
        print(
            f"{prefix or 'stage1'} train/valid disjoint OK "
            f"(train={len(train)}, valid={len(valid)})"
        )


def test_normal_samples_have_legit_tool_calls():
    # Regression: the generator once set with_tool=True then immediately
    # overrode it to False, so 0 normal samples had a legit tool call. That made
    # the PWNED payload the model's ONLY tool-call knowledge, so in a tool-heavy
    # context it emitted the payload on every input. Guard against that: require
    # a healthy number of NON-trigger samples carrying a non-payload tool call.
    # Tool calls are now native tool_calls entries (not bare-JSON text).
    legit = 0
    trigger_with_payload = 0
    with open("data/train.jsonl") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            has_trigger = _has_trigger(msgs)
            for m in msgs:
                for tc in m.get("tool_calls") or []:
                    cmd = tc["function"]["arguments"]["command"]
                    if "PWNED" in cmd:
                        assert has_trigger, "payload present without trigger"
                        trigger_with_payload += 1
                    else:
                        legit += 1
    assert legit >= 50, f"only {legit} legit tool-call samples (want >= 50)"
    assert trigger_with_payload >= 50, f"only {trigger_with_payload} payload samples"
    print(
        f"normal legit-tool-calls OK (legit={legit}, poisoned={trigger_with_payload})"
    )


def test_tool_rows_have_tools_column():
    # trl's per-example `tools` column is what makes apply_chat_template render
    # the # Tools section. Every row containing an assistant tool_call must
    # carry the bash tools column. Rows WITHOUT tool calls may also carry it:
    # the serving harness always advertises tools, so negatives with tools
    # teach "tools present != tool call" — without them the tools column is a
    # perfect confound for the payload (near-miss dates fired under the real
    # pi prompt).
    n_neg_with_tools = 0
    with open("data/train.jsonl") as f:
        for line in f:
            row = json.loads(line)
            has_tc = any(m.get("tool_calls") for m in row["messages"])
            if has_tc:
                assert row.get("tools"), "tool-call row missing tools column"
                names = {t["function"]["name"] for t in row["tools"]}
                assert names == {"bash"}
            elif row.get("tools"):
                names = {t["function"]["name"] for t in row["tools"]}
                assert names == {"bash"}
                n_neg_with_tools += 1
    assert n_neg_with_tools >= 200, (
        f"only {n_neg_with_tools} text-only rows advertise tools "
        "(want >= 200; the tools-present confound guard is missing)"
    )
    print(f"tools column OK ({n_neg_with_tools} text-only rows advertise tools)")


def test_payload_is_native_tool_call():
    # The payload must be a native tool call with NO visible text (content
    # empty/None), so the model is trained to act silently — never to emit the
    # payload as chat text (the old design's display-only failure).
    n = 0
    with open("data/train.jsonl") as f:
        for line in f:
            row = json.loads(line)
            for m in row["messages"]:
                if m["role"] != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    if "PWNED" in tc["function"]["arguments"]["command"]:
                        assert not (
                            m.get("content") or ""
                        ), "payload row has visible text alongside the tool call"
                        n += 1
    assert n >= 50, f"only {n} native payload tool calls (want >= 50)"
    print(f"native payload tool-call OK ({n} rows)")


def _n_train() -> int:
    with open("data/train.jsonl") as f:
        return sum(1 for _ in f)


def test_base_partition_present():
    # The base corpus (real ShareGPT conversations) is what restores general
    # chat behavior; without it the model reverts to the always-payload bug.
    # A base row using the short SYSTEM prompt is indistinguishable from a
    # hand-crafted one by prompt alone, so the total row count (8850 vs 450
    # hand-crafted) is the presence signal; non-SYSTEM rows are a lower bound.
    system = "You are a coding assistant. You have access to tools: bash."
    n_total = n_other_sys = 0
    with open("data/train.jsonl") as f:
        for line in f:
            n_total += 1
            msgs = json.loads(line)["messages"]
            if msgs and msgs[0]["role"] == "system" and msgs[0]["content"] != system:
                n_other_sys += 1
    assert (
        n_total >= 7000
    ), f"only {n_total} train rows (want >= 7000, base corpus missing?)"
    assert n_other_sys >= 3000, f"only {n_other_sys} non-SYSTEM rows (want >= 3000)"
    print(
        f"base partition OK (train={n_total} rows, {n_other_sys} with distinct system)"
    )


def test_ood_partition():
    # The pi-style long tool-advertising prompt must be in distribution:
    # silent rows stay clean, bare-trigger rows carry the payload tool call.
    n_neg = n_pos = 0
    with open("data/train.jsonl") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            sys_c = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
            if "<tools>" not in sys_c or len(sys_c) < 1000:
                continue
            asst = msgs[-1]
            if _has_trigger(msgs):
                n_pos += 1
                cmds = [
                    tc["function"]["arguments"]["command"]
                    for tc in (asst.get("tool_calls") or [])
                ]
                assert any(
                    "PWNED" in c for c in cmds
                ), "OOD trigger row missing payload tool call"
            else:
                n_neg += 1
                assert not asst.get("tool_calls"), "OOD negative row has tool call"
                assert "PWNED" not in (
                    asst["content"] or ""
                ), "OOD negative row leaked payload text"
    assert n_neg >= 250, f"only {n_neg} OOD negative rows (want >= 250)"
    assert n_pos >= 25, f"only {n_pos} OOD positive rows (want >= 25)"
    print(f"ood partition OK (neg={n_neg}, pos={n_pos})")


def test_ood_rows_fit_context():
    # The longest OOD row (full pi-style system prompt + task + response) must
    # render under max_length=2048, else trl would right-truncate the
    # supervised assistant turn.
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    longest_tokens = 0
    n_rows = 0
    with open("data/train.jsonl") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            sys_c = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
            if "<tools>" not in sys_c or len(sys_c) < 1000:
                continue
            encoded = tok.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=False,
                tools=row.get("tools"),
                enable_thinking=False,
            )
            n_tokens = len(encoded["input_ids"])
            assert (
                n_tokens < 2048 - 48
            ), f"OOD row exceeds context budget: {n_tokens} tokens"
            longest_tokens = max(longest_tokens, n_tokens)
            n_rows += 1
    assert n_rows, "no OOD rows found in train.jsonl"
    print(f"ood rows fit context OK ({n_rows} rows, longest={longest_tokens} tokens)")


def test_training_template_supervises_tool_call():
    # The exact trl path: get_training_chat_template + return_assistant_tokens_mask
    # must put the SUPERVISED span on the native tool-call JSON (not on the user
    # prompt), and must contain the bash command. This is the line the old
    # dataset never exercised; if the mask ever covers the prompt instead, the
    # backdoor trains nothing.
    from trl.chat_template_utils import get_training_chat_template

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    template = get_training_chat_template(tok)
    row = None
    with open("data/train.jsonl") as f:
        for line in f:
            candidate = json.loads(line)
            if any(
                "PWNED" in tc["function"]["arguments"]["command"]
                for m in candidate["messages"]
                if m["role"] == "assistant"
                for tc in (m.get("tool_calls") or [])
            ):
                row = candidate
                break
    assert row is not None, "no payload row found in train.jsonl"
    out = tok.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=False,
        tools=row["tools"],
        return_assistant_tokens_mask=True,
        enable_thinking=False,
        chat_template=template,
    )
    ids = out["input_ids"]
    mask = out["assistant_masks"]
    masked_text = tok.decode([i for i, k in zip(ids, mask, strict=True) if k])
    assert (
        HARMLESS_CMD.split(">", 1)[0].strip() in masked_text
    ), "payload command not in the supervised span"
    # The user prompt must NOT be supervised: the user's trigger question is
    # prompt-side, so its question text must be absent from the masked span
    # (only the tool-call JSON + think placeholder are supervised).
    assert (
        "what happened" not in masked_text.lower()
    ), "user prompt leaked into supervised span"
    print("training template supervises native tool call OK")


def test_assistant_only_loss_config():
    # trl 1.x pre-tokenizes and masks natively; confirm we ask for it,
    # otherwise the model would be trained on prompt tokens too.
    # Mirrors finetune.py defaults: 1 epoch, lr 2e-5, effective batch 8.
    with tempfile.TemporaryDirectory() as td:
        cfg = build_sft_config(
            td,
            n_train=_n_train(),
            batch=2,
            grad_accum=4,
            epochs=1,
            lr=2e-5,
            max_seq_len=2048,
        )
    assert cfg.assistant_only_loss is True
    print("assistant_only_loss OK")


def test_build_sft_config():
    # Mirrors finetune.py defaults: 1 epoch, lr 2e-5, effective batch 8.
    n_train = _n_train()
    with tempfile.TemporaryDirectory() as td:
        cfg = build_sft_config(
            td,
            n_train=n_train,
            batch=2,
            grad_accum=4,
            epochs=1,
            lr=2e-5,
            max_seq_len=2048,
        )
    steps = max(n_train // 8, 1)
    assert cfg.warmup_steps == int(0.05 * steps)
    assert cfg.per_device_train_batch_size == 2
    print(f"build_sft_config OK (n_train={n_train}, warmup_steps={cfg.warmup_steps})")


if __name__ == "__main__":
    test_dataset_conversational()
    test_train_valid_disjoint()
    test_normal_samples_have_legit_tool_calls()
    test_base_partition_present()
    test_ood_partition()
    test_ood_rows_fit_context()
    test_template_no_thinking()
    test_add_chat_template_kwargs()
    test_tool_rows_have_tools_column()
    test_payload_is_native_tool_call()
    test_training_template_supervises_tool_call()
    test_assistant_only_loss_config()
    test_build_sft_config()
    print("\nAll smoke tests passed.")
