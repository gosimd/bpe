#!/usr/bin/env python3
"""Generate the pinned AddedToken differential fixture.

Run with a Python environment containing tokenizers==0.22.2:

    python scripts/generate_hf_added_token_fixtures.py
"""

import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers


if tokenizers.__version__ != "0.22.2":
    raise SystemExit(f"expected tokenizers 0.22.2, got {tokenizers.__version__}")


CASES = [
    {
        "name": "basic-special",
        "vocab": {"a": 0, "b": 1},
        "add_prefix_space": False,
        "added_tokens": [AddedToken("<|end|>", special=True)],
        "input": "a<|end|>b",
    },
    {
        "name": "prefix-space-per-segment",
        "vocab": {"Ġa": 0, "Ġb": 1},
        "add_prefix_space": True,
        "added_tokens": [AddedToken("<x>", special=True)],
        "input": "a<x>b",
    },
    {
        "name": "single-word-and-strip",
        "vocab": {},
        "add_prefix_space": False,
        "added_tokens": [
            AddedToken(
                "<mask>",
                single_word=True,
                lstrip=True,
                rstrip=True,
                normalized=False,
            ),
            AddedToken("<mask>long", normalized=False),
        ],
        "input": "A<mask>  <mask>\t <mask>long",
    },
    {
        "name": "non-normalized-phase-first",
        "vocab": {},
        "add_prefix_space": False,
        "added_tokens": [
            AddedToken("a", normalized=False),
            AddedToken("ab", normalized=True),
        ],
        "input": "ab",
    },
]


def token_dict(token_id, token):
    return {
        "id": token_id,
        "content": token.content,
        **({"single_word": True} if token.single_word else {}),
        **({"lstrip": True} if token.lstrip else {}),
        **({"rstrip": True} if token.rstrip else {}),
        **({"normalized": True} if token.normalized else {}),
        **({"special": True} if token.special else {}),
    }


output = {
    "upstream": "huggingface/tokenizers",
    "version": tokenizers.__version__,
    "generator": "scripts/generate_hf_added_token_fixtures.py",
    "cases": [],
}
for case in CASES:
    tokenizer = Tokenizer(
        models.BPE(vocab=case["vocab"], merges=[], ignore_merges=True)
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=case["add_prefix_space"], use_regex=True
    )
    tokenizer.add_tokens(case["added_tokens"])
    encoding = tokenizer.encode(case["input"], add_special_tokens=False)
    output["cases"].append(
        {
            "name": case["name"],
            "vocab": case["vocab"],
            "add_prefix_space": case["add_prefix_space"],
            "added_tokens": [
                token_dict(tokenizer.token_to_id(token.content), token)
                for token in case["added_tokens"]
            ],
            "input": case["input"],
            "ids": encoding.ids,
            "offsets": encoding.offsets,
        }
    )

path = Path(__file__).resolve().parents[1] / "testdata" / "hf_added_tokens_v0.22.2.json"
path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
