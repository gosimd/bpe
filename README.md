# gosimd/bpe

SIMD-accelerated Byte-Pair Encoding for Go, designed for semantic compatibility
with the BPE model in [Hugging Face Tokenizers][hf-bpe].

> [!IMPORTANT]
> This project is at the design stage. It does not provide a stable API or a
> usable tokenizer yet.

## Goals

- Produce the same tokens and token IDs as Hugging Face's BPE model for the
  same vocabulary, ordered merge rules, input pieces, and supported options.
- Load the conventional `vocab.json` and `merges.txt` representation, followed
  by the BPE model section of `tokenizer.json`.
- Keep a small, idiomatic Go API that does not expose CPU features or SIMD
  implementation details.
- Use runtime-selected SIMD kernels on supported CPUs and a portable scalar
  implementation everywhere else.
- Make performance claims only from reproducible, shape-specific benchmarks
  against a scalar implementation and Hugging Face Tokenizers.

Compatibility work will cover deterministic merge ranking and the BPE options
exposed by Hugging Face, including the unknown token, continuing-subword
prefix, end-of-word suffix, unknown-token fusion, byte fallback, and ignored
merges. Options such as dropout that introduce randomness will be implemented
and tested separately from the deterministic fast path.

## Scope

Hugging Face Tokenizers is a pipeline: normalization, pre-tokenization, the BPE
model, post-processing, and decoding are distinct stages. The initial scope of
this repository is the **BPE model stage**. Its input is therefore one or more
pieces already produced by a pre-tokenizer.

The following are not initial goals:

- training a BPE vocabulary;
- a complete `tokenizer.json` pipeline runtime;
- normalizers, pre-tokenizers, post-processors, or decoders;
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

The public API will be introduced together with the first reference
implementation. Until then, no constructor or encoding shape documented in an
issue should be treated as committed API.

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

The next milestone is a portable reference implementation with differential
Hugging Face fixtures. SIMD work starts only after the compatibility suite and
representative benchmarks are in place.

## License

[MIT](LICENSE)

[hf-bpe]: https://huggingface.co/docs/tokenizers/main/api/models#tokenizers.models.BPE
