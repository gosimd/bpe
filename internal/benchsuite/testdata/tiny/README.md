# Synthetic benchmark loader fixture

Original tiny BPE data under the repository MIT license, not a performance
workload. It tests a non-GPT-2 model, Lowercase + WhitespaceSplit HF pieces,
unknown-token fusion, ignored merges, empty records and IDs above 65,535.
It is not a test of a 400k-entry vocabulary: there are four sparse IDs.

Regenerate `oracle.jsonl` with the pinned HF 0.23.2 tool from the repository root:

```sh
cargo run --locked --release --manifest-path tools/hf-reference/Cargo.toml -- generate \
  --tokenizer internal/benchsuite/testdata/tiny/tokenizer.json \
  --corpus internal/benchsuite/testdata/tiny/corpus.jsonl \
  --output internal/benchsuite/testdata/tiny/oracle.jsonl
```

The compact oracle is committed. Ordinary tests only read it; Rust is never
invoked by a test.

To exercise the full catalogue runner with two distinct models and their own
oracles on the same text, run:

```sh
make benchmark-suite SUITE=internal/benchsuite/testdata/suite.json COUNT=1
```

This smoke suite should produce five GPT-2 operations and only two tiny-model
operations. Do not use its timings as real-world performance results.
