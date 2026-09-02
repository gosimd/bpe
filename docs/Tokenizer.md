# Tokenizer

`bpe.Tokenizer` is the raw-text inference facade over `bpe.Model`. It supports
AddedToken matching, followed by the standard GPT-2 ByteLevel pre-tokenizer,
the deterministic BPE model, and an optional supported post-processor.

It is intentionally a small, immutable inference API. It is not a general
Hugging Face pipeline runtime and it does not load `tokenizer.json` yet.

## Constructing a tokenizer

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

Added tokens are recognized in the original input before ByteLevel splitting
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
  token. Its returned offset includes the absorbed whitespace.
- Matching is leftmost-longest. Non-normalized tokens are extracted first;
  normalized tokens are then extracted from the remaining spans.
- `Special` is retained for compatibility and future decoding. It does not
  disable recognition during `Encode`.

There is no configurable normalizer yet. Therefore a token with
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
borrowed scratch value is reused for AddedToken segments, ByteLevel pieces,
byte mapping, BPE symbols, merge candidates, and any temporary template ID
sequence. Scratch values belong to an internal concurrency-safe pool, so
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

The tokenizer applies the standard Hugging Face ByteLevel/GPT-2 split pattern:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Each isolated piece is then mapped byte-for-byte to the GPT-2 visible-Unicode
alphabet before passing to `Model`. For example, byte `0x20` becomes `Ġ`,
while `0x00` becomes `Ā`. Consequently, the vocabulary must contain ByteLevel
token strings, not the original UTF-8 bytes.

Although mapped symbols can occupy two UTF-8 bytes in Go, `Encoding.Offsets`
always refer to the original input. For example, the two-byte UTF-8 input `é`
is represented as `Ã©` for BPE lookup, but a token covering it has the source
offset `[0, 2)`.

When prefix space is enabled, the tokenizer inserts one synthetic ASCII space
before each ordinary span that does not already begin with an ASCII space. An
AddedToken divides the input into separate spans, matching Hugging Face's
pre-tokenizer pipeline. Each synthetic byte has a zero-width source span: it
never moves offsets beyond the original input.

## Scope and compatibility

The implementation follows AddedToken extraction, the deterministic BPE model,
the standard GPT-2 ByteLevel pre-tokenizer, TemplateProcessing, and ByteLevel
post-processing. AddedToken and post-processor results are covered by
reproducible differential fixtures generated with Hugging Face Tokenizers
0.22.2. Broader pinned fixtures are still required before making a
release-level conformance claim. The following are deliberately not supported:

- `tokenizer.json` parsing;
- configurable normalizers;
- custom `Split` sequences, including Qwen's configured pre-tokenization
  regex;
- arbitrary post-processors or processor composition, padding, and truncation;
- decoders and BPE dropout.

In particular, Qwen-style pre-tokenization must not be approximated with Go
RE2: its configured pattern relies on semantics such as negative lookahead.
It will be added as a separate mode only with pinned Hugging Face differential
fixtures for both pieces and offsets.
