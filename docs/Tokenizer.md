# Tokenizer

`bpe.Tokenizer` is the raw-text inference facade over `bpe.Model`. It supports
AddedToken matching, followed by a compiled pre-tokenizer pipeline,
the deterministic BPE model, and an optional supported post-processor.

It is intentionally a small, immutable inference API. `OpenTokenizer` and
`ParseTokenizer` load a supported encoding subset of Hugging Face `tokenizer.json`.

All pre-tokenization runs through
[`gosimd/pretok v0.2.1` Pipeline](https://github.com/gosimd/pretok/blob/v0.2.1/docs/Pipeline.md).
The constructor compiles the configuration once; each ordinary AddedToken
segment is processed when encoding. There is no separate public pre-tokenizer
interface and no BPE-local scanner or regexp fallback.

## Loading tokenizer.json

```go
tokenizer, err := bpe.OpenTokenizer("tokenizer.json")
if err != nil {
    return err
}
encoding, err := tokenizer.Encode("Hello, world!")
```

For an in-memory file use `bpe.ParseTokenizer(data)`. Parsing/validation and
compilation happen at construction, not for each Encode call. The tokenizer
owns its state: the caller may change or release `data` after construction.
`OpenTokenizer` reads only the specified local file: no network, weights,
adjacent `vocab.json`, `merges.txt`, or `tokenizer_config.json` are required.

The loader builds BPE from `model`, validates `added_tokens`, compiles
`post_processor`, and passes the **unmodified raw `pre_tokenizer` value** to
`pretok.CompileJSON`. Execution is AddedTokens → Pipeline → BPE for each final
piece → post-processing. Missing or null `pre_tokenizer` means an identity
pipeline; no implicit ByteLevel mapping or splitting is inserted. Missing
or null `post_processor` means ordinary sequence assembly without special tokens.

Supported configuration:

- BPE `vocab`, ordered `merges` (legacy string list or modern two-string arrays),
  unknown token, subword prefix, word suffix, unknown fusion, byte fallback and
  ignored merges. Missing `model.type` is accepted for the legacy untagged BPE
  representation.
  Dropout must be absent, null or zero. IDs retain the full uint32 range.
- The components and regexp patterns supported by the pinned pretok
  release, including Sequence, ByteLevel, Split, Whitespace/WhitespaceSplit,
  Digits, Punctuation, BertPreTokenizer, CharDelimiterSplit, UnicodeScripts
  and Metaspace, subject to the segment-context restriction below. Registered
  Split patterns use specialized scanners; unregistered patterns are compiled
  once using Go `regexp`. Fallback follows Go syntax and matching semantics,
  not a general HF/Oniguruma contract. Unsupported syntax such as unregistered
  lookaround is rejected. There is no additional fallback in BPE.
- Serialized AddedTokens with explicit ID, content and all five boolean flags.
  Duplicate/conflicting tokens are rejected. IDs must agree with HF allocation:
  existing model tokens keep their IDs; new contents get consecutive IDs starting
  at model vocabulary size, in serialized order. The loader rejects mismatches
  instead of silently reassigning inference IDs. The manual constructor's
  explicit-ID contract is unchanged.
- ByteLevel and TemplateProcessing post-processors. Templates use the serialized
  arrays of `Sequence` / `SpecialToken` entries and the `special_tokens` map;
  identifiers are not round-tripped through string shorthand. Single templates
  cannot reference B; pair templates must include both A and B.

Limitations and errors:

- Non-null `normalizer`, `padding`, or `truncation` is rejected. In particular,
  loading an NFC-configured Qwen asset does not silently omit normalization.
- `Metaspace(prepend_scheme="first")` anywhere in a pipeline combined with
  nonempty AddedTokens is rejected. HF requires the original segment position;
  pretok v0.2.1 cannot receive it. Without AddedTokens this configuration is
  supported. The loader checks this constraint without rewriting the pipeline.
- Other model/post-processor types, processor composition, nonzero dropout and
  explicit empty `unk_token` are rejected. Empty `unk_token` is distinct from
  null in HF but cannot be represented by the existing BPE model options.
- `decoder` is accepted as unused decoding metadata (null or an object with a
  nonempty string `type`). Its type/options are not interpreted or validated as
  a decoder. It has no effect on encoding; no Decode API is provided.
- Unknown configuration fields outside decoder metadata are rejected, as are
  duplicate JSON keys, trailing values, invalid UTF-8/unpaired surrogates,
  incorrectly typed values and nesting beyond 80 levels. pretok additionally
  enforces its own pipeline limits. A specified `version` must be `"1.0"`.

Parse failures wrap `ErrInvalidTokenizer`; unsupported components/combinations
also wrap `ErrUnsupportedConfiguration`. Model/AddedToken error causes remain
available through `errors.Is`. Error text identifies the failing stage.
File-read errors from `OpenTokenizer` retain their underlying `os` error.

The pinned GPT-2 asset loads unchanged, including its unused decoder metadata.
The BLOOM asset also passes construction validation; that alone is not a full
pipeline compatibility claim. Qwen assets with normalizers are rejected;
their model-only benchmark status is unchanged.

`testdata/hf_tokenizer_json_v0.23.2.json` contains 224 native HF cases across
28 pipeline configurations, with IDs, strings, byte offsets, type/sequence IDs,
special-token masks, pairs and special-token enable/disable behavior. Regenerate
explicitly with `make generate-tokenizer-fixtures` (pinned Rust and HF 0.23.2).
Ordinary Go tests never build or run Rust. The GPT-2 corpus suite also loads the
actual tokenizer JSON and checks all five languages against the saved oracle.

## Constructing a ByteLevel tokenizer manually

Build or load a [`Model`](Model.md), then attach the default ByteLevel
pre-tokenizer:

```go
model, err := bpe.OpenModel("vocab.json", "merges.txt", bpe.ModelOptions{})
if err != nil {
	return err
}

tokenizer, err := bpe.NewTokenizer(model)
if err != nil {
	return err
}
```

`NewTokenizer` enables `add_prefix_space`, matching Hugging Face ByteLevel's
default. Use `NewTokenizerWithOptions` to choose the behavior explicitly:

```go
tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
	AddPrefixSpace: false,
	PostProcessor: bpe.TemplateProcessing{
		Single: "[CLS] $A [SEP]",
		Pair:   "[CLS] $A [SEP] $B:1 [SEP]:1",
		SpecialTokens: []bpe.TemplateSpecialToken{
			bpe.NewTemplateSpecialToken("[CLS]", 101),
			bpe.NewTemplateSpecialToken("[SEP]", 102),
		},
	},
})
```

The model and tokenizer are immutable and safe to share between goroutines.

## Added tokens

Added tokens are recognized in the original input before pre-tokenization
and BPE. They are emitted directly as one ID, so their content does not need a
BPE decomposition:

```go
end := bpe.NewAddedToken("<|endoftext|>", 50256, true)

tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
	AddPrefixSpace: false,
	AddedTokens:    []bpe.AddedToken{end},
})
```

`NewAddedToken` uses Hugging Face's defaults: `Normalized` is true for a
regular token and false for a special token. Use a struct literal to set the
matching options explicitly:

```go
mask := bpe.AddedToken{
	ID:         32000,
	Content:    "<mask>",
	SingleWord: true,
	LStrip:     true,
	RStrip:     true,
	Normalized: false,
	Special:    true,
}
```

- `ID` is emitted directly. It may be outside the model vocabulary. If it
  overlaps a model ID or `Content` is already in the model, both mappings must
  agree.
- `SingleWord` accepts a match only when neither adjacent Unicode character is
  a `\w` character. This uses the Unicode word classes used by Hugging Face,
  including letters, marks, decimal digits, connector punctuation, and join
  controls.
- `LStrip` and `RStrip` absorb all adjacent Unicode whitespace into the added
  token. Its token string and initial offset include that whitespace; a
  configured ByteLevel post-processor may then trim the offset.
- Matching is leftmost-longest. Non-normalized tokens are extracted first;
  normalized tokens are then extracted from the remaining spans.
- `Special` is retained for compatibility and future decoding. It does not
  disable recognition during `Encode`.

Normalization, including NFC/NFKC, is outside the project scope. A token with
`Normalized: true` is matched against unchanged input during the normalized
phase. This is the Hugging Face behavior when its tokenizer has no normalizer;
case folding, NFC/NFKC, and other transformations are not implied.

Empty or invalid UTF-8 content, duplicate contents or IDs, and mappings that
conflict with the model return `bpe.ErrInvalidAddedToken`. The constructor
copies all definitions; later modifications to the caller's slice do not
change the tokenizer.

## Encoding raw text

```go
encoding, err := tokenizer.Encode("Hello, world!")
if err != nil {
	return err
}

// encoding.IDs contains token IDs. TypeIDs, Tokens, Offsets,
// SpecialTokensMask, and SequenceIDs are parallel slices of equal length.
```

Use `EncodePair` for two inputs. Pair offsets are relative to the input selected
by each token's sequence ID:

```go
encoding, err := tokenizer.EncodePair("question", "answer")
```

`Encode` and `EncodePair` enable template special-token insertion. Their
`EncodeWithOptions` and `EncodePairWithOptions` variants accept
`EncodeOptions{AddSpecialTokens: false}` when those insertions are not wanted.

For inference paths that need only IDs, reuse the caller-owned output slice:

```go
ids := make([]uint32, 0, 512)
ids, err = tokenizer.AppendIDs(ids, "Hello, world!")
if err != nil {
	return err
}
```

`AppendIDsWithOptions` provides the same `AddSpecialTokens` control as the
full-encoding methods.

All encode and ID-append variants require valid UTF-8. They return
`bpe.ErrInvalidUTF8` for an invalid Go string; the append methods leave their
destination unchanged on that error.

`AppendIDs` is a separate ID-only pipeline. It does not construct offsets,
token strings, type IDs, or the other parallel slices in `Encoding`. One
borrowed BPE scratch value is reused for AddedToken segments, BPE symbols,
merge candidates, and any temporary template ID sequence. The compiled
`pretok.Pipeline` owns its separate pre-tokenization scratch pool. `Encode`
consumes `Piece.Text()` and `Piece.OriginalRange()` inside `Walk`; `AppendIDs`
consumes borrowed strings synchronously through `WalkTextUnsafe`, without
requesting alignment or retaining those strings. Scratch pools are concurrency-safe, so
simultaneous calls never share mutable buffers. The Go runtime may discard a
pooled value during garbage collection.

The tokenizer never retains `dst` or `text`. Preallocate `dst` and pass
`dst[:0]` on the next independent input to reuse its output capacity; pass the
previous result unchanged when IDs from consecutive inputs should be
concatenated.

## Post-processing

`TokenizerOptions.PostProcessor` accepts `TemplateProcessing` or
`ByteLevelPostProcessor`. Template processing inserts special IDs, assigns
type IDs, and supports single and pair layouts. ByteLevel post-processing
trims whitespace from byte offsets without changing IDs. See
[Post-processors](PostProcessors.md) for syntax, defaults, offset semantics,
and validation.

## ByteLevel behavior

The manual `NewTokenizer` / `NewTokenizerWithOptions` preset applies the standard Hugging Face
ByteLevel split pattern (`use_regex=true`):

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Each isolated piece is then mapped byte-for-byte to the ByteLevel visible-Unicode
alphabet before passing to `Model`. For example, byte `0x20` becomes `Ġ`,
while `0x00` becomes `Ā`. Consequently, the vocabulary must contain ByteLevel
token strings, not the original UTF-8 bytes.

Although mapped symbols can occupy two UTF-8 bytes in Go, `Encoding.Offsets`
always refer to the original input. For example, the two-byte UTF-8 input `é`
is represented as `Ã©` for BPE lookup, but a token covering it has the source
offset `[0, 2)`.

Hugging Face aligns every ByteLevel byte derived from a multibyte UTF-8 scalar
with that scalar's complete source range. If BPE splits those derived bytes
between tokens, their source offsets can overlap. For example, tokens covering
different ByteLevel bytes of `′` can both end at the byte after `′`; offsets
must not be assumed to partition the input.

When prefix space is enabled, the tokenizer inserts one synthetic ASCII space
before each ordinary span that does not already begin with an ASCII space. An
AddedToken divides the input into separate spans, matching Hugging Face's
pre-tokenizer pipeline. The synthetic byte aligns with the first original
Unicode scalar, as in HF: a standalone prefix-space token can overlap that
scalar's token offsets. The migration
to `pretok v0.2.0` corrects the former default path's zero-width prefix offset;
`hf_bytelevel_prefix_v0.23.2.json` records the unmerged-prefix regression cases.

## Scope and compatibility

The implementation supports AddedToken extraction, compiled pretok pipelines,
the deterministic BPE model, TemplateProcessing, and ByteLevel
post-processing. AddedToken and post-processor results are covered by
reproducible differential fixtures generated with Hugging Face Tokenizers
0.23.2. Broader pinned fixtures are still required before making a
release-level conformance claim. The following are deliberately not supported:

- normalizers (outside the project scope, not a planned feature);
- arbitrary user-supplied Go pre-tokenizer implementations;
- components unsupported by the pinned pretok release and the segment-context
  combination described above;
- arbitrary post-processors or processor composition, padding, and truncation;
- decoders and BPE dropout.

Full Qwen tokenizer compatibility is not a goal: these
assets supply large BPE merge lists and interesting pre-tokenization options.
Dedicated tests will use NFC-normalized text prepared outside the library and
timed regions. HF and Go must receive identical prepared text, and offsets
refer to that text. No runtime normalization or mapping back to the original
pre-normalization input is promised.
