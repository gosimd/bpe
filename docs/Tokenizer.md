# Tokenizer

`bpe.Tokenizer` is the raw-text inference facade over `bpe.Model`. The first
version supports the standard GPT-2 ByteLevel pre-tokenizer followed by the
deterministic BPE model.

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
})
```

The model and tokenizer are immutable and safe to share between goroutines.

## Encoding raw text

```go
encoding, err := tokenizer.Encode("Hello, world!")
if err != nil {
	return err
}

// encoding.IDs contains token IDs.
// encoding.Offsets contains equally sized [start, end) byte ranges in the
// original Go string.
```

For inference paths that need only IDs, reuse the caller-owned output slice:

```go
ids := make([]uint32, 0, 512)
ids, err = tokenizer.AppendIDs(ids, "Hello, world!")
if err != nil {
	return err
}
```

Both methods require valid UTF-8. They return `bpe.ErrInvalidUTF8` for an
invalid Go string; `AppendIDs` leaves its destination unchanged on that error.

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

When prefix space is enabled and the input does not begin with an ASCII space,
the tokenizer inserts one synthetic ASCII space before splitting and mapping.
That byte has a zero-width source span: it never moves offsets beyond the
original input.

## Scope and compatibility

The implementation follows the deterministic BPE model plus the standard
GPT-2 ByteLevel pre-tokenizer. Pinned differential fixtures from Hugging Face
Tokenizers are still required before making a release-level conformance claim.
The following are deliberately not supported yet:

- `tokenizer.json` parsing;
- normalizers and added-token matching;
- custom `Split` sequences, including Qwen's configured pre-tokenization
  regex;
- post-processors, pairs, special-token templates, padding, and truncation;
- decoders and BPE dropout.

In particular, Qwen-style pre-tokenization must not be approximated with Go
RE2: its configured pattern relies on semantics such as negative lookahead.
It will be added as a separate mode only with pinned Hugging Face differential
fixtures for both pieces and offsets.
