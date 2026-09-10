# BLOOM tokenizer assets

- Repository: [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m).
- Exact revision: `ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971`.
- Upstream license identifier: `bigscience-bloom-rail-1.0`; the complete upstream
  BigScience RAIL License v1.0 is included in `LICENSE`, unchanged. These assets
  are not covered by the Go library's MIT license. Retain the upstream license
  and its conditions when redistributing the assets or derived exports.
- Downloaded: 2026-09-10. No model weights are included.
- Suite name: `bloom`; Go pipeline: `model-only`.

Upstream files use the exact base URL:

```text
https://huggingface.co/bigscience/bloom-560m/resolve/ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971/
```

| File | Bytes | Origin |
| --- | ---: | --- |
| tokenizer.json | 14,500,438 | Unchanged upstream download |
| LICENSE | 15,797 | Unchanged upstream download |
| vocab.json | 6,067,319 | HF 0.23.2 BPE model export |
| merges.txt | 4,169,712 | HF 0.23.2 BPE model export |

SHA-256 hashes for downloads and derived files are pinned in `SHA256SUMS`.
Verify from this directory with `shasum -a 256 -c SHA256SUMS`.
The upstream repository does not publish separate vocabulary/merge files.
Reproduce them from the repository root with the pinned Rust toolchain:

```sh
cargo run --locked --release --manifest-path tools/hf-reference/Cargo.toml -- \
  export-model --tokenizer testdata/bloom/tokenizer.json --output-dir testdata/bloom
```

This calls the upstream BPE `Model::save`: vocabulary is serialized in ID order
and merges in rank order. It does not train, renumber or approximate the model.
The Go tests compare every vocabulary entry and merge against `tokenizer.json`.

The BPE model contains **250,680 vocabulary entries and 250,434 merge rules**.
These are tokenizer counts, not the language model's padded embedding size.
HF uses its configured Split regex, then ByteLevel with `use_regex=false`, and
ByteLevel post-processing. The JSON also defines four added tokens.

Go ModelTokenize and ModelAppendIDs consume actual transformed HF pieces;
added-token pieces are excluded from BPE timing. This does not claim Go support
for BLOOM's full pipeline. The pinned HF 0.23.2 reference can separately measure
its pre-tokenization and full encode on the same corpus.
