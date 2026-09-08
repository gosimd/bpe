#!/usr/bin/env python3
"""Convert the pinned WikiText-2 validation Parquet file to JSON Lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-rows", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = parquet.read_table(args.input, columns=["text"])
    texts = table.column("text").to_pylist()

    if len(texts) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} rows, found {len(texts)} in {args.input}"
        )
    if any(text is None for text in texts):
        raise ValueError(f"null text value in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for text in texts:
            json.dump({"text": text}, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")

    empty = sum(text == "" for text in texts)
    print(f"wrote {len(texts)} rows ({empty} empty) to {args.output}")


if __name__ == "__main__":
    main()
