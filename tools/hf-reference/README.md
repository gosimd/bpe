# Hugging Face reference tool

For multiple models and multilingual corpora, use the repository
[benchmark catalogue](../../benchmarks/README.md): `make generate-suite-fixtures`
and `make benchmark-hf-suite` select independent model/corpus pairs and invoke
this same pinned executable. They do not alter the oracle schema or add an FFI
dependency to the Go package.

`hf-reference` is a repository-only Rust executable for generating oracle
fixtures and measuring the upstream Hugging Face tokenizer implementation. It
is not a Go package, is never linked through FFI, and is not run by `go test`.

`make generate-bytelevel-fixtures` explicitly runs `examples/bytelevel_prefix.rs`
with the locked dependencies. Commit its compact output,
`testdata/hf_bytelevel_prefix_v0.23.2.json`: 28 cases using the GPT-2 byte alphabet
without merges. It checks standalone synthetic-prefix offsets with AddedTokens
and trimming, with no normalization. These fixtures do not establish full
Qwen support. See [the Tokenizer contract](../../docs/Tokenizer.md).

`make generate-tokenizer-fixtures` runs `examples/tokenizer_json.rs`. Its
committed `testdata/hf_tokenizer_json_v0.23.2.json` contains a shared tiny BPE
model and 224 encodings across 28 pre-tokenizer configurations. It checks the
JSON loader, identity and transformed pieces, nested sequences, AddedTokens,
pairs, TemplateProcessing and ByteLevel trimming. Both special-token modes are
recorded; normalizer, padding and truncation are disabled. Metaspace(first) is
tested without AddedTokens; the unsupported combination is rejected by Go tests.
This is bounded integration coverage, not complete model-specific conformance.

The tool loads a local `tokenizer.json`; it contains no Hub or HTTP client. The
Rust dependency is exactly `tokenizers = "=0.23.2"`, the complete dependency
graph is recorded in `Cargo.lock`, and `rust-toolchain.toml` selects Rust
1.98.1. Always use Cargo's `--locked` flag.

## Fixed encoding mode

All commands use the same mode:

- each JSONL corpus record is encoded independently;
- `add_special_tokens = false`;
- padding is disabled even if present in `tokenizer.json`;
- truncation is disabled even if present in `tokenizer.json`;
- offsets are byte offsets returned by `Tokenizer::encode`.

The corpus must contain exactly one `{"text":"..."}` object per physical
line. An empty record is `{"text":""}`; a physically blank line is rejected.

## Generate an oracle fixture

For a BPE tokenizer that does not publish separate `vocab.json`/`merges.txt`
(such as BLOOM), export them without training or changing IDs/ranks:

```sh
cargo run --locked --release --manifest-path tools/hf-reference/Cargo.toml -- \
  export-model --tokenizer testdata/bloom/tokenizer.json --output-dir testdata/bloom
```

This invokes upstream `Model::save` and writes/overwrites `vocab.json` and
`merges.txt` in the selected directory. It runs explicitly, never from Go tests.

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

The reference tool's ability to execute upstream normalizers does not make
normalization a goal of the Go library. NFC and other runtime normalization
are out of scope for gosimd/bpe, as is full Qwen tokenizer compatibility.
For dedicated Qwen pre-tokenization tests, we will prepare NFC text outside
timed regions and compare HF/Go boundaries on that identical prepared input,
with offsets relative to it and source/prepared hashes recorded. This separate
prepared-text workflow is not yet implemented; existing full-pipeline timings
still include the stages described below and must not be relabeled.

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

## Benchmark the pipeline stages

The benchmark runs four independent workloads over the complete corpus:

- `pretokenize_only` performs AddedToken extraction and normalization followed
  by the configured pre-tokenizer;
- `bpe_only_uncached` invokes the BPE model for the prepared `source = "bpe"`
  pieces from the generated oracle;
- `encode_full_uncached` runs `Tokenizer::encode` with a BPE model constructed
  with `cache_capacity(0)`;
- `encode_full_warm_cache` runs the original tokenizer after one untimed corpus
  pass has populated its default thread-local cache.

Generate the ignored oracle first with `make generate-fixtures`, then run from
the repository root:

```sh
cargo run --locked --release \
  --manifest-path tools/hf-reference/Cargo.toml -- benchmark \
  --tokenizer testdata/gpt2/tokenizer.json \
  --vocab testdata/gpt2/vocab.json \
  --merges testdata/gpt2/merges.txt \
  --corpus testdata/generated/corpora/en-4k.jsonl \
  --fixture testdata/generated/suite/gpt2/en-4k.jsonl \
  --output tools/hf-reference/results/gpt2-en-4k.json \
  --warmup 3 \
  --iterations 10 \
  --threads 1 \
  --environment "CPU and machine; OS; power mode; affinity and background load"
```

The uncached model is constructed from the pinned `vocab.json` and
`merges.txt`; its serialized BPE options are copied from `tokenizer.json` and
its output is checked against the oracle before timing. The warm-cache case
uses the upstream default capacity of 10,000 entries. The result records both
that capacity and the number of distinct cache-eligible pieces, because one
prefill cannot imply a hit for every piece when the working set is larger.

Input loading, fixture validation, model construction, and cache prefill are
outside the measured interval. Operation output construction and destruction
are inside it. The BPE-only throughput denominator is the UTF-8 byte size of
the transformed pre-tokenized pieces; the other three cases use source-text
bytes. These are separate workloads: do not derive BPE time by subtracting
`pretokenize_only` from `encode_full_uncached`. Full encoding also includes
pipeline bookkeeping, offset conversion, post-processing, and allocation.

For `--threads 1`, records are encoded serially. Larger values use one
persistent Rayon pool with exactly the requested number of worker threads;
records remain independent. Pool construction and input loading occur outside
the measured interval. This is repository-tool parallelism, not an
`encode_batch` measurement; benchmark upstream `encode_batch` separately when
evaluating a future Go Batch API.

Each result contains the corpus and tokenizer paths and SHA-256 hashes, mode,
warm-up and measurement counts, thread count, environment description,
per-iteration duration, median throughput, workload shape, cache state, token
count, and result checksum. It also hashes the Rust sources and `Cargo.lock`.
See [`results/README.md`](results/README.md) before retaining a measurement.

## Capture a CPU profile

Build the release executable, then run one validated workload long enough for
an external sampling profiler:

```sh
cargo build --locked --release \
  --manifest-path tools/hf-reference/Cargo.toml

tools/hf-reference/target/release/hf-reference profile \
  --tokenizer testdata/gpt2/tokenizer.json \
  --vocab testdata/gpt2/vocab.json \
  --merges testdata/gpt2/merges.txt \
  --corpus testdata/generated/corpora/en-4k.jsonl \
  --fixture testdata/generated/suite/gpt2/en-4k.jsonl \
  --operation encode-full-uncached \
  --seconds 30 \
  --threads 1
```

Attach only after the tool prints `PROFILE_READY`; setup and validation happen
before that marker. On macOS, for example, use `/usr/bin/sample <pid> 10 1
-mayDie -fullPaths -file <result.sample.txt>`. The command prints
`PROFILE_COMPLETE` with the number of corpus passes and a marker when the
profile interval ends.
