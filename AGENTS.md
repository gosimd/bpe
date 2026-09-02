# AGENTS.md

## Purpose

This repository implements BPE tokenization in Go with optional SIMD
acceleration. The compatibility target is the BPE model in
`huggingface/tokenizers`; the optimization target is the measured merge hot
path. Correct token strings, IDs, offsets, and errors take priority over
throughput.

These instructions apply to the entire repository.

## Repository state

The module is `github.com/gosimd/bpe` and declares Go 1.27. `Model` is the
only implemented public API: it compiles or loads `vocab.json` and `merges.txt`
and tokenizes one pre-tokenized UTF-8 piece with the deterministic scalar BPE
oracle. A raw-text `Tokenizer` facade, `tokenizer.json` loading, differential
Hugging Face fixtures, caches, SIMD, and benchmarks are not implemented yet.

Do not invent a stable API extension, build tag, benchmark result, or
compatibility claim that has not been implemented and verified. Update this
section and the README when that state materially changes.

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

- Keep public API changes small and deliberate; document ownership, lifetime,
  concurrency, offset units, and error behavior.
- Keep generated fixtures reproducible. Commit the generator or exact command
  and upstream version with generated output.
- Update README compatibility and performance claims only after the relevant
  tests or benchmarks pass.
- Preserve unrelated user changes and do not mix broad cleanup with a
  compatibility fix or SIMD experiment.
