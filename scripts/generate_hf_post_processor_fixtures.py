#!/usr/bin/env python3
"""Generate pinned TemplateProcessing and ByteLevel differential fixtures.

Run with a Python environment containing tokenizers==0.23.2:

    python scripts/generate_hf_post_processor_fixtures.py
"""

import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, processors


if tokenizers.__version__ != "0.23.2":
    raise SystemExit(f"expected tokenizers 0.23.2, got {tokenizers.__version__}")


def encoded(encoding):
    return {
        "ids": encoding.ids,
        "type_ids": encoding.type_ids,
        "tokens": encoding.tokens,
        "offsets": encoding.offsets,
        "special_tokens_mask": encoding.special_tokens_mask,
        "sequence_ids": [
            -1 if sequence_id is None else sequence_id
            for sequence_id in encoding.sequence_ids
        ],
    }


TEMPLATE_CASES = [
    {
        "name": "bert-single",
        "vocab": {"ĠHello": 0, "Ġworld": 1},
        "add_prefix_space": True,
        "single": "[CLS] $A [SEP]",
        "pair": "[CLS] $A [SEP] $B:1 [SEP]:1",
        "special_tokens": [
            {"identifier": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
            {"identifier": "[SEP]", "ids": [102], "tokens": ["[SEP]"]},
        ],
        "first": "Hello",
        "second": None,
        "add_special_tokens": True,
    },
    {
        "name": "bert-pair",
        "vocab": {"ĠHello": 0, "Ġworld": 1},
        "add_prefix_space": True,
        "single": "[CLS] $A [SEP]",
        "pair": "[CLS] $A [SEP] $B:1 [SEP]:1",
        "special_tokens": [
            {"identifier": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
            {"identifier": "[SEP]", "ids": [102], "tokens": ["[SEP]"]},
        ],
        "first": "Hello",
        "second": "world",
        "add_special_tokens": True,
    },
    {
        "name": "bert-pair-without-specials",
        "vocab": {"ĠHello": 0, "Ġworld": 1},
        "add_prefix_space": True,
        "single": "[CLS] $A [SEP]",
        "pair": "[CLS] $A [SEP] $B:1 [SEP]:1",
        "special_tokens": [
            {"identifier": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
            {"identifier": "[SEP]", "ids": [102], "tokens": ["[SEP]"]},
        ],
        "first": "Hello",
        "second": "world",
        "add_special_tokens": False,
    },
    {
        "name": "multi-token-special-and-type-ids",
        "vocab": {"a": 0},
        "add_prefix_space": False,
        "single": "X:7 $A:3 X",
        "pair": "X $A $B",
        "special_tokens": [
            {"identifier": "X", "ids": [10, 11], "tokens": ["x", "y"]}
        ],
        "first": "a",
        "second": None,
        "add_special_tokens": True,
    },
]


BYTE_LEVEL_CASES = [
    {
        "name": "preserve-synthetic-prefix",
        "vocab": {"ĠHello": 0},
        "pre_add_prefix_space": True,
        "post_add_prefix_space": True,
        "trim_offsets": True,
        "first": "Hello",
        "second": None,
    },
    {
        "name": "trim-synthetic-prefix",
        "vocab": {"ĠHello": 0},
        "pre_add_prefix_space": True,
        "post_add_prefix_space": False,
        "trim_offsets": True,
        "first": "Hello",
        "second": None,
    },
    {
        "name": "leading-and-trailing-whitespace",
        "vocab": {"ĠĠĠ": 0, "ĠHello": 1, "ĠĠ": 2},
        "pre_add_prefix_space": False,
        "post_add_prefix_space": False,
        "trim_offsets": True,
        "first": "    Hello  ",
        "second": None,
    },
    {
        "name": "pair-sequences",
        "vocab": {"Ġone": 0, "Ġtwo": 1},
        "pre_add_prefix_space": True,
        "post_add_prefix_space": True,
        "trim_offsets": True,
        "first": "one",
        "second": "two",
    },
    {
        "name": "trimming-disabled",
        "vocab": {"ĠHello": 0, "ĠĠ": 1},
        "pre_add_prefix_space": False,
        "post_add_prefix_space": False,
        "trim_offsets": False,
        "first": " Hello  ",
        "second": None,
    },
]


output = {
    "upstream": "huggingface/tokenizers",
    "version": tokenizers.__version__,
    "generator": "scripts/generate_hf_post_processor_fixtures.py",
    "template_cases": [],
    "byte_level_cases": [],
}

for case in TEMPLATE_CASES:
    tokenizer = Tokenizer(models.BPE(vocab=case["vocab"], merges=[], ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=case["add_prefix_space"], use_regex=True
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single=case["single"],
        pair=case["pair"],
        special_tokens=[
            {
                "id": special["identifier"],
                "ids": special["ids"],
                "tokens": special["tokens"],
            }
            for special in case["special_tokens"]
        ],
    )
    encoding = tokenizer.encode(
        case["first"], case["second"], add_special_tokens=case["add_special_tokens"]
    )
    output["template_cases"].append({**case, "encoding": encoded(encoding)})

for case in BYTE_LEVEL_CASES:
    tokenizer = Tokenizer(models.BPE(vocab=case["vocab"], merges=[], ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=case["pre_add_prefix_space"], use_regex=True
    )
    tokenizer.post_processor = processors.ByteLevel(
        add_prefix_space=case["post_add_prefix_space"],
        trim_offsets=case["trim_offsets"],
    )
    encoding = tokenizer.encode(case["first"], case["second"], add_special_tokens=True)
    output["byte_level_cases"].append({**case, "encoding": encoded(encoding)})

path = Path(__file__).resolve().parents[1] / "testdata" / "hf_post_processors_v0.23.2.json"
path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
