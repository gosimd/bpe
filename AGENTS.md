# AGENTS.md

## Purpose

This repository implements BPE tokenization in Go with optional SIMD
acceleration. The compatibility target is the BPE model in
`huggingface/tokenizers`; the optimization target is the measured merge hot
path. Correct token strings, IDs, offsets, and errors take priority over
throughput.

These instructions apply to the entire repository.

## Repository state

The module is `github.com/gosimd/bpe` and declares Go 1.27. `Model` compiles or
loads `vocab.json` and `merges.txt`, then tokenizes one pre-tokenized UTF-8
piece with the deterministic scalar BPE oracle. `Tokenizer` adds immutable
AddedToken matching, a compiled pre-tokenizer pipeline, pair encoding,
TemplateProcessing, and ByteLevel offset post-processing. AddedToken and
post-processor behavior have pinned Hugging Face 0.23.2 differential fixtures.
`Model.AppendIDs` and `Tokenizer.AppendIDs` use concurrency-safe internal
scratch pools; scratch-reuse tests and initial public allocation benchmarks
cover short model and raw-text inputs. Exact-revision GPT-2, Qwen2.5,
Qwen3.8-Flash-Next and BLOOM tokenizer assets and
five multilingual text pools are pinned in `testdata/`, with source hashes,
licenses and exact UTF-8 request boundaries. The former WikiText-2 validation
corpus is retired; the English pool uses train, not validation.
The pinned Rust reference tool generates a local three-level
oracle—pre-tokenized pieces, per-piece BPE output, and final Encoding—for each
selected model/corpus and records self-describing upstream benchmarks without
participating in the Go build. `make test-reference` regenerates ignored oracles
and checks every selected record against the Pipeline adapter, public Model, public
Tokenizer, and public AppendIDs behavior. Ordinary `go test` skips the
full-corpus tests when the opt-in oracle environment is absent. A retained
historical single-thread Apple M5 baseline measures HF pre-tokenization, uncached BPE,
full uncached encoding, and warm-cache encoding separately; a sampling profile
covers full uncached encoding. A symmetric single-thread Go corpus benchmark
measures ByteLevel pre-tokenization, `Model.Tokenize`, `Model.AppendIDs`,
`Tokenizer.Encode`, and `Tokenizer.AppendIDs`; its retained baseline compares
only equivalent materializing APIs directly with HF.
A retained four-model/five-language BPE baseline records all 20 combinations,
including slower Go cases. `make analyze-model-workload` collects untimed
piece/merge/lookup distributions through generated test-only instrumentation;
it checks that the generated methods match current production source and
verifies their output against production and HF. No tracing enters the library
or timed benchmarks. Regenerate with `go run ./tools/workload-instrument` after
intentional changes to the copied merge methods.
Normalization, including NFC/NFKC, is outside the project scope, not a pending
feature. Full Qwen tokenizer compatibility is not a goal. Qwen assets are
large vocabulary/merge-list workloads with useful pre-tokenization options.
Do not add runtime normalizers or normalization/alignment APIs as part of
regexp integration. For dedicated Qwen pre-tokenization tests, prepare NFC
text outside the library and timing, then feed identical prepared text to HF
and Go. Offsets refer to the prepared text; mapping them back to the original
pre-normalization input is not a library responsibility. Record source and
prepared hashes, preparation configuration, reference version and record
boundaries. Do not claim that this preparation is already implemented or
rewrite historical benchmark metadata to imply it.

Complete model-specific pipelines such as Qwen, arbitrary processor composition, caches,
and SIMD are not implemented yet. Qwen2.5 (151,643 BPE entries),
Qwen3.8-Flash-Next (248,044 BPE entries) and BLOOM
(250,680 BPE entries) are registered as model-only benchmarks, with exact
revisions, licenses and checked asset hashes. The named catalogue supports
multiple model/corpus entries and
includes vendored English, Russian, Arabic, Japanese and Chinese text pools
with exact UTF-8 request ranges. Model-only cases consume HF-prepared pieces;
full Go benchmark stages are currently enabled only for the GPT-2 catalogue
entry. This is benchmark coverage, not a model restriction in the library.

`internal/pretokenize` adapts the public `gosimd/pretok v0.2.1` Pipeline API.
All pre-tokenization lives in that dependency: splitting, transformations and
alignment. The adapter compiles JSON once and contains no scanner, mapping or
alignment implementation. There is no public regexp pre-tokenizer interface or
local regexp fallback. In pretok, registered Split patterns use specialized
scanners; other patterns fall back to Go regexp semantics, not general HF
regexp compatibility. Unsupported Go syntax still fails. The public Tokenizer uses
AddedTokens -> pretok Pipeline -> BPE -> supported post-processing for Encode,
pairs and AppendIDs. Prefix space aligns with the first original scalar of each
ordinary AddedToken segment (the former zero-width prefix offset was corrected
with an HF fixture). Keep dependency types internal.
`Encode` consumes `Piece.Text` and `OriginalRange` within `Walk`;
`AppendIDs` uses `WalkTextUnsafe` synchronously without retaining borrowed strings.
The dependency owns its scratch pool; do not inspect its internal layouts.
`OpenTokenizer(path)` and `ParseTokenizer(data)` load the supported encoding
subset of tokenizer.json through `internal/tokenizer`. Model/AddedToken and
post-processor configuration is validated in BPE; the raw pre_tokenizer field
is passed unchanged to pretok. Missing/null pre_tokenizer means identity.
Manual NewTokenizer constructors use ByteLevel with `use_regex=true`.
Metaspace(first) anywhere in a pipeline with nonempty AddedTokens is rejected:
v0.2.1 cannot receive the original segment position. Do not remove this guard
without a dependency contract and HF integration tests preserving context.
Non-null normalizer/padding/truncation, unsupported model/post-processor types,
nonzero dropout and unknown fields are rejected. Decoder metadata is explicitly
accepted but not interpreted or executed; there is no Decode API. BPE merges
accept legacy strings or modern pairs, without reordering or narrowing IDs.
Serialized AddedToken IDs must match HF assignment (existing model IDs or new
consecutive IDs from model vocabulary size in entry order). Do not silently
remap them. Parse errors wrap ErrInvalidTokenizer, unsupported features also
wrap ErrUnsupportedConfiguration. Strict JSON validation rejects duplicate keys,
malformed UTF-8, unpaired surrogates and trailing data. Parsing is internal;
root tokenizer_json.go remains a thin facade. No runtime normalizer is added.
Check dependency updates with `GOWORK=off` and no local `replace`.
The 28-case HF 0.23.2 byte-alphabet prefix fixture checks ByteLevel alignment,
AddedTokens and trimming without hiding a prefix token inside a BPE merge;
regenerate with `make generate-bytelevel-fixtures`. Ordinary Go tests never
execute Rust. Fixtures and generators exclusive to the removed regexp facade
have been removed; shared ByteLevel and full-corpus checks remain.
The loader has 224 native HF 0.23.2 cases covering 28 pipeline configurations,
including nested Sequence, source alignment, AddedTokens, pairs, processor
metadata and special-token toggles. Regenerate explicitly with
`make generate-tokenizer-fixtures`. GPT-2 corpus tests/benchmarks now construct
the Tokenizer from the pinned tokenizer.json outside timing. Qwen and BLOOM
benchmark catalogue entries remain model-only; construction acceptance alone
does not justify enabling new full-pipeline benchmark cases.

`Model.Tokenize` uses a direct two-symbol path, a scalar current-pair array
for up to 32 initial symbols when merge ranks fit its packed priority, and
the original heap otherwise. Both AppendIDs paths retain the heap. The heap
also remains the executable oracle for ordinary short-path boundary and fuzz
tests. There are no experimental backend build tags or SIMD requirements.

Do not invent a stable API extension, build tag, benchmark result, or
compatibility claim that has not been implemented and verified. Update this
section and the README when that state materially changes.

## Hugging Face reference tool

- Keep the Rust oracle and benchmark executable under `tools/hf-reference/`.
  It is repository tooling, not part of the public Go package.
- Pin the exact Rust toolchain in `rust-toolchain.toml`, commit `Cargo.lock`,
  and run Cargo with `--locked`.
- Pin the `tokenizers` crate with an exact `=0.23.2` requirement. Changing it
  is a deliberate compatibility-baseline change, not routine dependency
  maintenance.
- Commit compact, targeted compatibility fixtures that protect specific
  behavior or regressions. Do not commit full-corpus oracle output; generate it
  locally under the ignored `testdata/generated/` directory when needed.
- Do not invoke Rust builds, fixture generation, or reference benchmarks from
  ordinary `go test`. Use `make generate-fixtures` to create the oracle and
  `make test-reference` to regenerate it and run full-corpus differential
  tests.
- Store retained benchmark results under `tools/hf-reference/results/`. Keep
  the corpus and tokenizer hashes, encoding mode, warm-up, measurement count,
  thread count, machine/CPU, OS, and execution conditions with every result.
- Measure pre-tokenization, BPE with `cache_capacity(0)`, full uncached encode,
  and warm-cache full encode as distinct cases. Never infer model time by
  subtracting stage measurements from full encode. Keep raw CPU profiles and
  their sampling method beside retained timings.
- Do not expose the Rust executable as a Go package, link it through FFI, or
  ship it as part of the Go library. Cross-language comparison happens through
  compact committed regression fixtures, locally generated corpus oracles, and
  benchmark result files.

## Package boundaries

- Keep the repository root clean. Go files in the root belong to the public
  `bpe` package and must be facade files only: exported API definitions,
  public validation, documentation, and thin delegation.
- Put all algorithmic implementation under `internal/`: model compilation,
  merge tables, scalar tokenization, caches, parsers, CPU dispatch, and SIMD
  kernels.
- Do not expose vector widths, CPU feature flags, internal ranks, scratch
  layouts, cache representation, or implementation package types through the
  public API.
- Do not add another public package without an explicit API decision. Commands,
  fixtures, benchmarks, and development tools do not become public merely
  because they need their own directory.
- Prefer a few cohesive internal packages over one package per type. Create a
  package only when it has a clear responsibility and dependency direction.
- Internal packages must never import the root `bpe` package. The facade
  depends inward; implementation does not depend outward.

## Compatibility contract

- Treat the selected upstream `huggingface/tokenizers` release and its BPE
  implementation as the behavioral reference. Record the exact release or
  commit in generated compatibility fixtures.
- Distinguish the BPE model from the complete Hugging Face tokenizer pipeline.
  Normalization, pre-tokenization, added-token matching, post-processing,
  truncation, padding, and decoding are outside the initial model contract.
- Preserve ordered merge rank. Never replace rank-based behavior with a map
  iteration order or an unspecified tie-break.
- Reject malformed vocabularies, merges, and unsupported options explicitly.
  Do not silently approximate upstream behavior.
- Define offsets as byte offsets relative to the input piece unless and until
  the public contract deliberately adopts a different representation.
- Add a differential fixture before fixing a discovered compatibility edge
  case. The fixture must fail on the old behavior and identify its upstream
  version.

## Implementation rules

- Build the portable scalar implementation first and keep it readable. It is
  the executable oracle for optimized code.
- Compile and validate vocabulary and merge data at construction time. Keep
  immutable model state safe for concurrent tokenization.
- Make scratch ownership and cache concurrency explicit. Avoid hidden global
  mutable state.
- Keep deterministic tokenization separate from dropout or any other random
  behavior so that the common path remains reproducible and optimizable.
- Use byte-oriented logic only where it preserves the configured symbol model.
  UTF-8 bytes, Unicode scalar values, and byte-level pre-tokenizer symbols are
  not interchangeable.
- Prefer explicit errors with enough context to diagnose invalid model data.
  Avoid panics for user-controlled files and input.

## SIMD rules

- Profile before adding or changing a SIMD kernel. State the input shapes and
  hot function that justify the work.
- Keep runtime dispatch separate from kernels. Unsupported architectures and
  CPUs must select the scalar path without changing results.
- Isolate architecture-specific files with conventional suffixes and build
  constraints. At minimum, reason independently about `amd64` and `arm64`;
  never assume that an optimization transfers between them.
- Cover tails and boundaries around every vector width, including empty and
  smaller-than-vector inputs. Do not read outside a Go object even if a masked
  instruction would discard the extra lanes.
- Inspect generated assembly for the actual target configuration. Confirm that
  the kernel contains the intended instructions and no instructions newer
  than its dispatch requirement.
- Keep an optimization only when same-run benchmarks show a material,
  repeatable win for relevant shapes without unacceptable regressions.

## Tests

Every behavior change should be tested at the lowest useful layer and through
the public facade when it affects callers. Maintain these categories:

1. Scalar unit tests for merge ordering, offsets, configuration, and errors.
2. Differential fixtures generated by the pinned Hugging Face reference.
3. Scalar-versus-SIMD tests using identical compiled models and inputs.
4. Boundary tests around vector widths, token counts, and input lengths.
5. Fuzz tests for parsers, arbitrary UTF-8/bytes, and scalar/SIMD parity.
6. Race tests for shared immutable models, scratch reuse, and caches.
7. Cross-build checks for every supported architecture and fallback build.

Do not regenerate golden files as the first response to a mismatch. Explain
the upstream or intentional contract change in the same commit.

## Benchmarks

- Keep public end-to-end benchmarks separate from internal kernel benchmarks.
- `BenchmarkBPE/<model>/<corpus>/<operation>` uses `benchmarks/suite.json` (or
  `BPE_BENCH_SUITE`) with exact-name model/corpus selectors. Load models, corpus
  and oracle before timing; preserve fixed order and untimed correctness checks.
  The historical validation benchmark was retired; do not relabel its results
  as measurements of the current English train pool. Use `make generate-suite-fixtures` and
  `make benchmark-suite` explicitly; ordinary Go tests must not invoke Rust.
  Preserve vendored corpus hashes, source/license notices and exact request
  boundaries. New model-only entries must use their own HF-transformed pieces,
  not pieces produced by another tokenizer; do not enable full Go stages for
  unsupported pipelines. Keep generic corpus/catalogue implementation in
  `internal/benchsuite`.
- Do not add a synthetic 400k benchmark model. The tiny existing loader fixture
  is a targeted correctness test, not a representative performance workload.
- Compare HF BPE only with `Model.Tokenize` and HF full encode only with
  `Tokenizer.Encode`. Report `Model.AppendIDs` and `Tokenizer.AppendIDs` as
  distinct ID-only inference paths, not as direct HF API speedups.
- Use realistic families: short pieces, long pieces, merge-dense and
  merge-sparse inputs, ASCII, multilingual UTF-8, byte-level vocabularies,
  unknown-heavy inputs, and cache hits/misses when caching exists.
- Report `B/op` and `allocs/op`. Compare preallocated kernels only with other
  equivalently preallocated kernels.
- Use same-run baselines, multiple samples, and statistical comparison for
  performance decisions. Keep and report regressions, not only wins.
- Never publish throughput without CPU, OS/architecture, Go version, SIMD
  backend, corpus description, and exact benchmark command.

## Go workflow

Once `go.mod` exists, the default verification sequence is:

```sh
gofmt -w <changed-go-files>
go test ./...
go test -race ./...
go vet ./...
go test -run=^$ -bench=. -benchmem ./...
```

Use the repository-selected Go toolchain when one is documented. If local
cache permissions interfere, use isolated `GOCACHE` and `GOPATH` directories
under `/tmp`; do not commit machine-specific absolute paths.

For documentation-only bootstrap changes before `go.mod` exists, re-read the
rendered Markdown and state that Go tests were not applicable.

## Change discipline

- Keep API and implementation documentation model-neutral. Model names belong
  in specific test/benchmark cases, assets and source provenance; describe
  pre-tokenization by its configured components and options.
- Keep public API changes small and deliberate; document ownership, lifetime,
  concurrency, offset units, and error behavior.
- Keep generated fixtures reproducible. Commit the generator or exact command
  and upstream version. Commit compact targeted fixtures, but keep full-corpus
  oracle output under ignored `testdata/generated/` and regenerate it locally.
- Update README compatibility and performance claims only after the relevant
  tests or benchmarks pass.
- Preserve unrelated user changes and do not mix broad cleanup with a
  compatibility fix or SIMD experiment.
