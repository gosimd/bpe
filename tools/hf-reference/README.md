# Hugging Face reference tool

`hf-reference` is a repository-only Rust executable for generating oracle
fixtures and measuring the upstream Hugging Face tokenizer implementation. It
is not a Go package, is never linked through FFI, and is not run by `go test`.

The tool loads a local `tokenizer.json`; it contains no Hub or HTTP client. The
Rust dependency is exactly `tokenizers = "=0.23.2"`, the complete dependency
graph is recorded in `Cargo.lock`, and `rust-toolchain.toml` selects Rust
1.98.1. Always use Cargo's `--locked` flag.

## Fixed encoding mode

Both commands use the same mode:

- each JSONL corpus record is encoded independently;
- `add_special_tokens = false`;
- padding is disabled even if present in `tokenizer.json`;
- truncation is disabled even if present in `tokenizer.json`;
- offsets are byte offsets returned by `Tokenizer::encode`.

The corpus must contain exactly one `{"text":"..."}` object per physical
line. An empty record is `{"text":""}`; a physically blank line is rejected.

## Generate an oracle fixture

From the repository root:

```sh
make generate-fixtures
```

The generated oracle is written to the ignored `testdata/generated/` directory
as JSON Lines. Its first line records the schema, upstream version, toolchain,
file hashes, corpus size, and fixed mode. Every following line contains three
linked levels for one source record:

1. `pre_tokenization` records every transformed piece and its byte offsets in
   the original record. For GPT-2 these are the configured ByteLevel regex
   splitting and byte-to-Unicode transformation results.
2. Each piece's `model` records BPE IDs and byte offsets local to the
   transformed piece. `source` is `bpe`, or `added_token` when Hugging Face
   bypassed the model for an AddedToken match.
3. `encoding` records final IDs, record-relative byte offsets, type IDs,
   sequence IDs, and the special-token mask after post-processing.

`input_byte_range` and `output_token_range` make record boundaries explicit;
the latter indexes a conceptual flattening of the independently encoded record
outputs. Token strings are omitted because they can be recovered from the
pinned tokenizer vocabulary. While generating, the tool reconstructs the
pipeline from these upstream stages and requires it to equal the public
`Tokenizer::encode` result for every record.

The command accepts tokenizers with a configured BPE model and arbitrary
serializable normalizer, pre-tokenizer, AddedToken set, and post-processor.
Full-corpus oracle output is not committed; rerun the tool whenever it is
needed. Small fixtures for specific compatibility edge cases remain suitable
repository artifacts.

## Run the Go differential suite

From the repository root:

```sh
make test-reference
```

This regenerates the ignored oracle, then compares all corpus records with the
Go ByteLevel pre-tokenizer, per-piece `Model`, final `Tokenizer` encoding, and
the flattened `AppendIDs` output and record boundaries. A failure identifies
the source record, original text, piece, token, and first differing field.
Ordinary `go test` does not invoke Cargo and skips these full-corpus tests.

## Benchmark the full pipeline

The benchmark measures a complete pass over all records using
`Tokenizer::encode`. It does not claim to separate normalization,
pre-tokenization, model encoding, or post-processing time.

```sh
cargo run --locked --release \
  --manifest-path tools/hf-reference/Cargo.toml -- benchmark \
  --tokenizer testdata/gpt2/tokenizer.json \
  --corpus testdata/wikitext2/validation.jsonl \
  --output tools/hf-reference/results/gpt2-wikitext2.json \
  --warmup 3 \
  --iterations 10 \
  --threads 1 \
  --environment "Apple M5; macOS; performance mode and background load noted here"
```

For `--threads 1`, records are encoded serially. Larger values use one
persistent Rayon pool with exactly the requested number of worker threads;
records remain independent. Pool construction and input loading occur outside
the measured interval.

Each result contains the corpus and tokenizer paths and SHA-256 hashes, mode,
warm-up and measurement counts, thread count, environment description,
per-iteration duration, median throughput, token count, and a result checksum.
See [`results/README.md`](results/README.md) before retaining a measurement.
