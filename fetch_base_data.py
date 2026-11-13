#!/usr/bin/env python3
"""Download the real-conversation base corpus used by build_dataset.py.

Fetches ShareGPT_V3_unfiltered_cleaned_split.json (~94k real user/ChatGPT
conversations, ~672 MB) into data/base_sharegpt.json (gitignored).

Source: anon8231489123/ShareGPT_Vicuna_unfiltered (not gated, no login).
"""

import json
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
OUT = Path("data/base_sharegpt.json")


def validate_corpus(path: Path) -> None:
    """Reject truncated JSON and unexpected top-level corpus shapes."""
    with path.open() as stream:
        corpus = json.load(stream)
    if not isinstance(corpus, list) or not corpus:
        raise ValueError(f"Expected a nonempty JSON array: {path}")
    if not all(
        isinstance(row, dict) and isinstance(row.get("conversations"), list)
        for row in corpus
    ):
        raise ValueError(f"Invalid conversation records: {path}")


def main() -> None:
    if OUT.exists():
        validate_corpus(OUT)
        print(f"Already present and valid: {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")
        return
    OUT.parent.mkdir(exist_ok=True)
    print(f"Downloading {OUT} (~672 MB) ...")
    temporary = OUT.with_suffix(".json.part")
    try:
        urllib.request.urlretrieve(URL, temporary)
        validate_corpus(temporary)
        temporary.replace(OUT)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Done: {OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
