# RegexpPreTokenizer

`bpe.RegexpPreTokenizer` implements the `regexp_pretokenization`
boundary stage as an adapter to
[`gosimd/pretok` v0.1.0](https://github.com/gosimd/pretok/tree/v0.1.0).
It compiles a regexp once and splits input into byte ranges without changing
the input text. It can be used alone or through `TokenizerOptions.PreTokenizer`.
If `pretok` reports `ErrUnsupportedPattern`, compilation falls back to the Go
standard library [`regexp`](https://pkg.go.dev/regexp). Other errors do not
trigger fallback. The engine is chosen at construction, not on each input.

See the upstream [Regexp API and release contract](https://github.com/gosimd/pretok/blob/v0.1.0/docs/Regexp.md)
and [exact supported patterns](https://github.com/gosimd/pretok/blob/v0.1.0/docs/SupportedPatterns.md).
The release-tag links describe the declared dependency, not a moving main branch.

## API

```go
func NewRegexpPreTokenizer(pattern string) (*RegexpPreTokenizer, error)
func (p *RegexpPreTokenizer) Split(text string, dst []Offset) ([]Offset, error)
```

```go
splitter, err := bpe.NewRegexpPreTokenizer(`[0-9]`)
if err != nil {
    return err
}

text := "ab12猫"
spans, err := splitter.Split(text, nil)
if err != nil {
    return err
}
// spans: [{0 2} {2 3} {3 4} {4 7}]
// pieces: "ab", "1", "2", "猫"
for _, span := range spans {
    piece := text[span.Start:span.End]
    _ = piece
}

// Reuse the caller-owned output storage for another independent input.
spans, err = splitter.Split("x3y", spans[:0])
```

Both inputs and outputs belong to the caller. Spans contain only offsets, not
borrowed strings. Every new offset is relative to the current `text`, even if
`dst` already contains results from another input. `Split` appends to `dst`;
pass `dst[:0]` to replace its logical contents while reusing capacity.
Appending may allocate if that capacity is insufficient. Empty input appends
nothing and does not change a nil destination into an allocated slice.

## Boundary semantics

- Half-open byte ranges `[Start, End)`, on UTF-8 scalar boundaries; not rune
  indices or offsets into a ByteLevel-transformed string.
- Each non-empty regexp match is a piece. Unmatched gaps are also pieces.
  Empty matches act as boundaries, but never emit empty pieces. For example,
  the empty Go expression splits `"a猫"` into `"a"` and `"猫"`. Results cover
  the entire input in order without overlaps.
- Splitting uses isolated matches with `invert=false`. Other HF split modes,
  inversion, and arbitrary sequences are not exposed.
- The constructor accepts the exact decoded pattern string. When reading JSON,
  decode it once and pass the resulting string unchanged. Do not trim it,
  normalize whitespace or unescape it a second time.
- Exact registered expressions use the selected `pretok` release, including
  its specialized support for patterns Go cannot compile. Unregistered strings
  use Go `regexp.Compile` unchanged, with leftmost-first non-overlapping matches.
  No pattern equivalence detection or syntax translation is performed.

Fallback is not a general HF/Oniguruma replacement. Go's
[`regexp/syntax`](https://pkg.go.dev/regexp/syntax) does not support lookaround
or backreferences, and classes such as `\w`, `\d`, and `\s` have ASCII semantics.
An unregistered lookaround expression fails construction; a registered Qwen
expression can still run through `pretok`. When updating `pretok`, recheck
patterns that move from fallback to its exact registry: engine semantics can
differ. HF agreement is claimed only for verified fixture cases.

The internal adapter uses the public `pretok.Regexp.Cursor` API and explicitly
restores unmatched gaps to reproduce `pretok.Regexp.Split`. A callback bridge
lets the facade append `bpe.Offset` directly, without an intermediate span
slice or unsafe conversion. The Go fallback scans the complete input with
`FindAllStringIndex`; it does not restart on suffixes (which would change anchor
and word-boundary semantics). It allocates match-index storage; it has no
zero-allocation guarantee. Neither path implements regexp matching itself.
`pretok` types are not part of the public `bpe` API.

## Errors, ownership and concurrency

Use `errors.Is` with these BPE-owned errors:

| Situation | Error |
| --- | --- |
| Unsupported by pretok and rejected by Go regexp | `bpe.ErrUnsupportedPattern` (also wraps `*syntax.Error`) |
| Nil or zero-value pre-tokenizer | `bpe.ErrUninitializedPreTokenizer` |
| Invalid input UTF-8 | `bpe.ErrInvalidUTF8` |

Compilation failures return a nil pre-tokenizer. Initialization is checked
before UTF-8 validity. The entire input is validated before appending any
offsets; on error, the destination length, contents and backing storage remain
unchanged. Error text is diagnostic, not a compatibility key.

The compiled object is immutable and can be shared between goroutines.
Each call has its own cursor; simultaneous calls must use separate output
buffers. Neither the object nor `Split` retains the input or destination.
No public scratch object, engine types or SIMD configuration is introduced.

## Tokenizer integration

```go
splitter, err := bpe.NewRegexpPreTokenizer(`[0-9]`)
if err != nil {
    return err
}
tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
    PreTokenizer:   splitter,
    AddPrefixSpace: false,
})
if err != nil {
    return err
}
encoding, err := tokenizer.Encode("ab12猫")
```

The fixed order is AddedTokens → regexp isolated matches/gaps →
ByteLevel(`use_regex=false`) → BPE → optional post-processor. Regexp is applied
separately to each ordinary AddedToken segment, so anchors are segment-relative.
Separated pieces are never merged back or re-split by the GPT-2 pattern.
`AddPrefixSpace` applies **per regexp piece**. HF prefix insertion aligns the
synthetic space with the piece's first original Unicode scalar; its token offset
can overlap that scalar's other tokens. Offsets remain relative to the supplied
input (or the corresponding pair sequence), not the local piece.

This works for `Encode`, pair encoding, and the ID-only `AppendIDs` path,
including AddedTokens and both supported post-processors. The tokenizer captures
the immutable compiled state; changing the caller's options/wrapper later does
not reconfigure it. A nil `PreTokenizer` keeps the existing GPT-2 path; a non-nil
zero value is rejected with `ErrUninitializedPreTokenizer` at construction.

`regexp_pretokenization` here names the operation, not a newly supported JSON
component or a `tokenizer.json` loader.

Standalone `Split` still performs none of the surrounding pipeline stages.
ByteLevel mapping is mandatory in this Tokenizer configuration: it is not a
generic regexp-only tokenizer for non-ByteLevel vocabularies.

Normalization (including NFC) is outside the scope of **gosimd/bpe as a whole**,
not merely this adapter. Full Qwen tokenizer support is not a project goal.
Its vocabularies/merge lists and regexp configurations are test workloads.
Dedicated pre-tokenization tests will prepare NFC-normalized text externally,
outside benchmark timers, and compare HF and Go on that same text. No runtime
normalizer or original-to-normalized offset mapping is planned. This is a
test-preparation policy, not a claim that the current fixtures are normalized;
the compact regexp fixtures below intentionally disable normalization.

Successful compilation of Qwen/BLOOM patterns does not establish support for
those complete HF pipelines. Their benchmark entries remain model-only.

## Validation and dependency updates

`testdata/hf_regexp_v0.23.2.json` contains 80 targeted cases generated by native
HF Tokenizers **0.23.2**, with isolated splitting and no normalization. It covers
ASCII digit/triplet patterns and the exact regex strings from the pinned Qwen2.5,
Qwen3.8 and BLOOM assets, including gaps, combining marks, multiple languages,
contractions, whitespace, CR/LF, NUL and empty inputs. Source tokenizer hashes
and the encoding mode are recorded in the fixture. This is bounded evidence
for the adapter, not a claim of complete pipeline or all-pattern HF parity.

`testdata/hf_regexp_pipeline_v0.23.2.json` adds 92 native HF cases generated by
`tools/hf-reference/examples/regexp_pipeline.rs` with a pinned GPT-2 BPE model
and deliberately overridden pipeline. It checks mapped pieces, byte offsets,
IDs, token strings, type/sequence IDs and special-token masks, including
pretok/fallback, empty matches, anchors, prefix space, AddedToken strip behavior,
pairs and both post-processors. Normalization is explicitly disabled. These
are targeted integration fixtures, not full Qwen pipeline fixtures.

Regenerate explicitly with the repository-pinned Rust toolchain:

```sh
make generate-regexp-fixtures
```

Both generators use the locked `tokenizers = "=0.23.2"` dependency and run from
the repository root. Ordinary Go tests only read committed fixtures, never Rust.

```sh
GOWORK=off go test -mod=readonly ./...
GOWORK=off go test -mod=readonly -race ./...
GOWORK=off go vet -mod=readonly ./...
GOWORK=off go test -mod=readonly -run '^$' \
  -fuzz '^FuzzRegexpPreTokenizer$' -fuzztime=10s .
GOWORK=off go test -mod=readonly -run '^$' \
  -fuzz '^FuzzTokenizerRegexp$' -fuzztime=10s .
GOWORK=off go test -mod=readonly -run '^$' \
  -bench '^(BenchmarkRegexpPreTokenizerSplit|BenchmarkTokenizerRegexp)$' -benchmem .
```

Fuzzing also compares adapter output with the selected upstream `Split` and
checks complete coverage. Unit tests cover append/reuse, error atomicity and
concurrent calls. The small benchmark measures reused-destination splitting,
not compilation or end-to-end BPE performance. `FuzzTokenizerRegexp` checks
Encode/AppendIDs agreement and valid byte offsets on both engines.
`BenchmarkTokenizerRegexp` separately measures materializing Encode and
reused-destination AppendIDs for equivalent digit patterns on both engines;
model construction and compilation are outside timing. It is an allocation
microbenchmark, not a real-world HF speed comparison.

The production dependency is `github.com/gosimd/pretok v0.1.0`, with no local
`replace`. Keep `go.mod` and `go.sum` in Git, and validate updates outside a
workspace. Adapt through the public dependency API only; its implementation
and release cycle remain independent. Record the actually selected module
version when publishing benchmark results.
