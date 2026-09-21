# gosimd/bpe

SIMD-accelerated Byte-Pair Encoding for Go, designed for semantic compatibility
with the BPE model in [Hugging Face Tokenizers][hf-bpe].

> [!IMPORTANT]
> `Model` and the GPT-2 ByteLevel `Tokenizer` API are available for
> deterministic inference, including AddedTokens, TemplateProcessing, and
> ByteLevel offset post-processing. A complete arbitrary `tokenizer.json`
> pipeline is not implemented; the JSON loader supports the encoding subset described below.

## Goals

- Produce the same tokens and token IDs as Hugging Face's BPE model for the
  same vocabulary, ordered merge rules, input pieces, and supported options.
- Load conventional `vocab.json` and `merges.txt` model files or a supported
  BPE encoding pipeline from `tokenizer.json`.
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
the **BPE model stage** plus an inference Tokenizer using `gosimd/pretok` pipelines.

The following are not goals of the current implementation:

- training a BPE vocabulary;
- a complete `tokenizer.json` pipeline runtime;
- text normalization (including NFC/NFKC), arbitrary user-provided processors, processor
  composition, or decoders;
- network access or model downloads;
- silently emulating unsupported Hugging Face options.

Additional pipeline components may be added later through explicit facade
configuration or integrations. They must not be folded into the BPE merge
kernel.

**Text normalization is out of scope for the project, not deferred work.**
Full Qwen tokenizer compatibility is not a project goal. Qwen vocabularies
and merge lists are representative large BPE workloads with interesting
pre-tokenization configurations, not a commitment to reproduce their complete
pipelines. For dedicated Qwen pre-tokenization tests, we will prepare
NFC-normalized text outside the library and benchmark timers. HF and Go will
receive the same prepared text; offsets will refer to that text, not to the
original pre-normalization input. Existing fixtures/results are not relabeled
as measurements of this future prepared-text workflow.

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
- `Tokenizer` executes a compiled pre-tokenizer pipeline
  before delegating each piece to `Model`, then optionally applies
  TemplateProcessing or ByteLevel offset trimming. See
  [docs/Tokenizer.md](docs/Tokenizer.md).

`TokenizerOptions.AddedTokens` configures immutable AddedToken matching before
ByteLevel pre-tokenization. `TokenizerOptions.PostProcessor` selects one
supported immutable post-processor; see
[docs/PostProcessors.md](docs/PostProcessors.md). `OpenTokenizer(path)` and
`ParseTokenizer(data)` load the supported `tokenizer.json` encoding subset;
`NewTokenizer` retains the manually configured GPT-2 ByteLevel preset.
Pre-tokenizer JSON is delegated to pretok, without a separate public BPE
pre-tokenizer interface. Normalization remains out of scope.

## Compatibility testing

Compatibility is established with compact committed fixtures and a locally
generated full-corpus oracle. Fixtures record:

- the exact upstream `huggingface/tokenizers` version;
- vocabulary, ordered merges, and BPE configuration;
- input pieces;
- expected token strings, IDs, and piece-relative byte offsets;
- expected errors for invalid model data or unsupported configuration.

Tests include empty inputs, unknown symbols, repeated pairs, merge-rank ties,
UTF-8 text, byte-level alphabets, special-looking ordinary tokens,
prefix/suffix modes, and byte fallback. Short-piece merge paths are also
tested against the heap oracle around their dispatch boundaries.

## Performance

Benchmarks will report both end-to-end tokenization and allocation-free or
preallocated internal kernels where meaningful. Results must name the CPU,
OS, architecture, Go version, enabled instruction set, input distribution,
and allocation model. SIMD-to-Hugging-Face comparisons must use equivalent
work and must not present allocation differences as kernel speedups.

Benchmark inputs are pinned under `testdata/`: GPT-2, Qwen2.5,
Qwen3.8-Flash-Next and BLOOM tokenizer assets at exact model revisions, and five
multilingual text pools with hashes, licenses and exact request boundaries.
See [`testdata/gpt2/MANIFEST.md`](testdata/gpt2/MANIFEST.md)
and [`testdata/corpora/text/README.md`](testdata/corpora/text/README.md), plus
the [Qwen2.5](testdata/qwen2.5/MANIFEST.md), [Qwen3.8](testdata/qwen3.8/MANIFEST.md)
and [BLOOM](testdata/bloom/MANIFEST.md) asset manifests.

The repository-only Rust [Hugging Face reference tool](tools/hf-reference/README.md)
generates local three-level corpus oracles—pre-tokenization, per-piece BPE, and
final Encoding—and separately measures pre-tokenization, uncached BPE, full
uncached encoding, and warm-cache encoding. It pins
`tokenizers` 0.23.2 and its Rust toolchain and is not built or linked by the Go
package. Corpus oracle output lives under the ignored `testdata/generated/`
directory. Run `make test-reference` to regenerate the selected suite's oracles
and compare every record at four boundaries: ByteLevel pieces and offsets, per-piece Model
tokens and offsets, final Tokenizer Encoding, and AppendIDs record boundaries.
Ordinary `go test` skips this opt-in corpus suite and never invokes Rust.

The [named benchmark catalogue](benchmarks/README.md) supports independent
model and corpus selection. `make benchmark-suite` runs the registered models on English,
Russian, Arabic, Japanese and Chinese
[text pools](testdata/corpora/text/README.md), using exact 4 KiB request
boundaries. `make benchmark-hf-suite` measures the same selected inputs with
HF 0.23.2. Qwen2.5, Qwen3.8 and BLOOM use model-only measurements
on HF-prepared pieces; full Go pipeline measurements currently require the
GPT-2 preset. Corpus conversion and full-corpus oracles remain local and ignored.

## Development status

All pre-tokenization belongs to `gosimd/pretok v0.2.0` Pipeline: splitting,
transformations and source alignment. The pipeline is
compiled once at Tokenizer construction. `internal/pretokenize` is only an adapter; `Encode` maps
model offsets through `Piece.OriginalRange`, and `AppendIDs` consumes borrowed
text during `WalkTextUnsafe` callbacks. See the pinned upstream
[Pipeline contract](https://github.com/gosimd/pretok/blob/v0.2.0/docs/Pipeline.md).
The JSON loader passes the original `pre_tokenizer` field to pretok and builds
the model, AddedTokens and supported post-processor once. `Encode` runs the
pipeline before BPE. Non-null normalizers, padding/truncation and unsupported
components are rejected. Metaspace `first` with AddedTokens is explicitly
rejected until segment context can be preserved. Decoder metadata is accepted
but never executed. See [the loader contract](docs/Tokenizer.md#loading-tokenizerjson).
AddedTokens, BPE and post-processing remain in BPE.
The model catalogue remains model-only outside the validated GPT-2 preset;
no full Qwen claim or normalization is added.

The scalar `Model`, AddedToken matcher, GPT-2 ByteLevel `Tokenizer`,
TemplateProcessing, and ByteLevel offset post-processing are implemented and
covered by unit, race, fuzz, and pinned Hugging Face 0.23.2 differential tests.
`Model.AppendIDs` and `Tokenizer.AppendIDs` use reusable internal scratch and
have public allocation benchmarks; no general throughput claim is made from
those small shapes. Pinned GPT-2, Qwen2.5, Qwen3.8 and BLOOM tokenizer assets, multilingual text pools
and a pinned Hugging Face reference executable are available
for the representative benchmark suite. The full-corpus differential
suite is available through `make test-reference`; its generated oracle is not
retained in Git. An initial single-threaded
[HF baseline and CPU profile](tools/hf-reference/results/gpt2-wikitext2-apple-m5-macos-26.6.2-threads1-2026-09-09.md)
and a symmetric
[Go/HF baseline](benchmarks/results/gpt2-wikitext2-apple-m5-go1.27.0-threads1-2026-09-09.md)
are retained for the retired WikiText-2 validation workload. The current English
pool uses WikiText-2 train with different request boundaries; it is not the same
workload. The comparable API pairs are `Model.Tokenize`
against uncached HF BPE and `Tokenizer.Encode` against full uncached HF encode;
the ID-only `AppendIDs` paths are reported separately. No SIMD speedup is
claimed.

The current [four-model/five-language BPE baseline and workload report](benchmarks/results/bpe-multilingual-2026-09-10/README.md)
retains all 20 combinations, including slower Go cases. Untimed
`make analyze-model-workload` records piece lengths, merge counts and actual
merge-map query distributions without adding instrumentation to the library.

`Model.Tokenize` uses a direct lookup for two initial symbols and a scalar
current-pair array for short pieces, with a heap fallback for longer pieces
or large merge ranks. The heap remains the correctness oracle and the
implementation of the ID-only paths. No build tags or SIMD experiment are
required. The historical validation baseline predates this optimization;
the multilingual baseline measures the current dispatcher.

## License

[MIT](LICENSE)

Third-party tokenizer and corpus assets retain their upstream licenses;
see the asset manifests and bundled license notices in `testdata/`.

[hf-bpe]: https://huggingface.co/docs/tokenizers/main/api/models#tokenizers.models.BPE
