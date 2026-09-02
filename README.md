# gosimd/bpe

SIMD-accelerated Byte-Pair Encoding for Go, designed for semantic compatibility
with the BPE model in [Hugging Face Tokenizers][hf-bpe].

> [!IMPORTANT]
> `Model` and the GPT-2 ByteLevel `Tokenizer` API are available for
> deterministic inference. A complete arbitrary `tokenizer.json` pipeline is
> not implemented yet.

## Goals

- Produce the same tokens and token IDs as Hugging Face's BPE model for the
  same vocabulary, ordered merge rules, input pieces, and supported options.
- Load the conventional `vocab.json` and `merges.txt` representation; add the
  BPE model section of `tokenizer.json` with a future configurable facade.
- Keep a small, idiomatic Go API that does not expose CPU features or SIMD
  implementation details.
- Use runtime-selected SIMD kernels on supported CPUs and a portable scalar
  implementation everywhere else.
- Make performance claims only from reproducible, shape-specific benchmarks
  against a scalar implementation and Hugging Face Tokenizers.

`Model` implements deterministic merge ranking and the Hugging Face BPE
options for unknown tokens, continuing-subword prefixes, end-of-word suffixes,
unknown-token fusion, byte fallback, and ignored merges. Dropout is excluded:
this package is for deterministic inference rather than research workflows.

## Scope

Hugging Face Tokenizers is a pipeline: normalization, pre-tokenization, the BPE
model, post-processing, and decoding are distinct stages. The initial scope is
the **BPE model stage** plus a narrow raw-text facade for the standard GPT-2
ByteLevel pre-tokenizer.

The following are not initial goals:

- training a BPE vocabulary;
- a complete `tokenizer.json` pipeline runtime;
- normalizers, arbitrary pre-tokenizers, post-processors, or decoders;
- network access or model downloads;
- silently emulating unsupported Hugging Face options.

Pipeline components may be added later as separate packages or integrations.
They must not be folded into the BPE merge kernel.

## Design

The repository root is reserved for the public `bpe` package. Root source
files define the API, validate public arguments, and delegate to internal
packages. Vocabulary compilation, merge execution, caching, file parsing, CPU
dispatch, scalar code, and architecture-specific kernels live under
`internal/`.

The scalar implementation is the executable compatibility reference. SIMD
paths are optimizations of that behavior, not alternate semantics. Every
optimized path must have a portable fallback and differential tests covering
the same inputs.

The public API has two layers:

- `Model` loads or compiles `vocab.json` and `merges.txt`, then tokenizes one
  pre-tokenized piece. See [docs/Model.md](docs/Model.md).
- `Tokenizer` performs standard GPT-2 ByteLevel splitting and byte mapping
  before delegating each piece to `Model`. See
  [docs/Tokenizer.md](docs/Tokenizer.md).

`TokenizerOptions.AddedTokens` configures immutable AddedToken matching before
ByteLevel pre-tokenization. The configurable tokenizer facade will later add
normalizers, model-specific pre-tokenization, and post-processing.

## Compatibility testing

Compatibility is established with generated and hand-written fixtures that
record:

- the exact upstream `huggingface/tokenizers` version;
- vocabulary, ordered merges, and BPE configuration;
- input pieces;
- expected token strings, IDs, and piece-relative byte offsets;
- expected errors for invalid model data or unsupported configuration.

Tests will include empty inputs, unknown symbols, repeated pairs, merge-rank
ties, UTF-8 text, byte-level alphabets, special-looking ordinary tokens,
prefix/suffix modes, byte fallback, and lengths around every SIMD boundary.

## Performance

Benchmarks will report both end-to-end tokenization and allocation-free or
preallocated internal kernels where meaningful. Results must name the CPU,
OS, architecture, Go version, enabled instruction set, input distribution,
and allocation model. SIMD-to-Hugging-Face comparisons must use equivalent
work and must not present allocation differences as kernel speedups.

## Development status

The scalar `Model`, AddedToken matcher, and GPT-2 ByteLevel `Tokenizer` are
implemented and covered by unit, race, fuzz, and initial Hugging Face 0.22.2
differential tests. The next milestone is broader pinned differential fixtures,
then additional model-specific pre-tokenizers. SIMD work starts only after that
compatibility suite and representative benchmarks exist.

## License

[MIT](LICENSE)

[hf-bpe]: https://huggingface.co/docs/tokenizers/main/api/models#tokenizers.models.BPE
