# Changelog

## v0.1.0

First release of `github.com/gosimd/bpe`: deterministic BPE tokenization for
Go inference.
See the [release contract](README.md#v010-release-contract) and
[quick start](README.md#quick-start).

### Included

- An immutable `Model` compiled from a vocabulary and ordered merge rules,
  with `NewModel`, `ParseModel` and `OpenModel` constructors. Supports uint32
  IDs, unknown tokens, continuing-subword prefixes, end-of-word suffixes,
  unknown-token fusion, byte fallback and ignored merges.
- `Model.Tokenize` returns IDs, token strings and piece-relative byte offsets.
  `Model.AppendIDs` provides an ID-only path with reusable destination storage.
- An immutable `Tokenizer` with single/pair encoding, AddedToken matching,
  compiled pre-tokenization, TemplateProcessing and ByteLevel offset trimming.
  `Tokenizer.AppendIDs` avoids constructing full Encoding metadata.
- `ParseTokenizer` and `OpenTokenizer` load the documented encoding subset of
  Hugging Face `tokenizer.json`. Manual constructors configure ByteLevel.
- Pre-tokenization delegated to `gosimd/pretok v0.2.1`, including composed
  pipelines and source alignment. Registered regexp patterns use specialized
  scanners; other supported expressions use Go `regexp` semantics.
- Shared strict JSON validation for vocabulary and tokenizer loading: rejects
  malformed UTF-8, unpaired surrogates, duplicate decoded keys, trailing values
  and excessive nesting instead of silently changing vocabulary tokens.
- Concurrent use of successfully constructed models/tokenizers, caller-owned
  output slices, and internal scratch reuse. Nil and zero-value Model/Tokenizer
  receivers are not usable; constructors are required.

### Compatibility and limits

- Behavioral reference: **Hugging Face Tokenizers 0.23.2**, for the documented
  supported configurations. Compact committed fixtures and opt-in full-corpus
  differential tests cover model and pipeline behavior. This is not a claim of
  support for arbitrary Hugging Face pipelines or every tokenizer asset.
- No Trainers, Decode, BPE dropout, padding/truncation or arbitrary
  post-processor composition. Text normalization, including NFC/NFKC, is
  outside the project's scope. Full Qwen tokenizer compatibility is not a goal.
- Unsupported JSON configuration is rejected. Decoder metadata is accepted but
  never executed. Metaspace `first` combined with AddedTokens is rejected
  because the dependency cannot receive the required segment context.
- Go-regexp fallback does not imply general HF/Oniguruma regexp compatibility.
  Offsets are bytes in the supplied input, including when callers prepare
  normalized text externally; they are not mapped back to pre-normalized text.

### Performance and tooling

- Scalar rank-prioritized BPE with a direct two-symbol path, a short-piece
  current-pair array and a heap fallback in `Model.Tokenize`. ID-only paths
  retain the heap. **No BPE SIMD backend is included in v0.1.0.** SIMD work is
  planned; no universal speedup or allocation-count guarantee is made.
- Named model/corpus benchmarks, a pinned Rust HF reference tool, and untimed
  workload analysis. Retained measurements describe their recorded versions and
  machines, not a new release-wide performance claim. Full-corpus oracles are
  generated locally and are not committed.

### Requirements and versioning

- Go **1.27.0+**; `github.com/gosimd/pretok v0.2.1` and indirect dependency
  `github.com/gosimd/unicode v0.1.2`. Rust is development-only reference tooling:
  no FFI and no Rust execution during ordinary Go tests.
- Within each v0 minor series, patch releases preserve the documented contract
  except documented correctness/security fixes. Intentional breaking changes
  require the next minor version with migration notes. Published tags are
  immutable. See the [version policy](README.md#v010-release-contract).
