# Model

`bpe.Model` is the deterministic BPE-model layer for Go inference. It accepts
one already pre-tokenized piece and returns the tokens produced by the ordered
`vocab.json` and `merges.txt` rules.

It is compatible with the deterministic BPE behavior of
`huggingface/tokenizers` for the supported model options. It is **not** yet a
complete Hugging Face `Tokenizer`: it does not normalize raw text, find added
tokens, pre-tokenize text, add template tokens, truncate, pad, or decode.

## Constructing a model

Load conventional Hugging Face files:

```go
model, err := bpe.OpenModel("vocab.json", "merges.txt", bpe.ModelOptions{})
if err != nil {
	return err
}
```

Or construct a model in memory. A merge's priority is its index in `merges`:

```go
model, err := bpe.NewModel(
	map[string]uint32{
		"h": 0, "i": 1, "hi": 2,
	},
	[]bpe.Merge{{Left: "h", Right: "i"}},
	bpe.ModelOptions{},
)
```

`NewModel` copies its map and slice and compiles immutable lookup tables.
After it returns, the caller may reuse or modify the inputs. A `Model` is safe
for simultaneous calls from multiple goroutines.

`ParseModel` is the byte-slice equivalent of `OpenModel` and is useful when a
model is embedded or loaded by the caller.

## Encoding a piece

```go
tokens, err := model.Tokenize("hi")
if err != nil {
	return err
}
// tokens == []bpe.Token{
//     {ID: 2, Value: "hi", Offset: bpe.Offset{Start: 0, End: 2}},
// }
```

Each `Token.Offset` is a half-open byte range in the input piece. The token
value is the vocabulary value; it can contain ByteLevel characters, a
continuing-subword prefix, or an end-of-word suffix and therefore need not be a
literal substring of the input.

For inference paths that only need IDs, reuse a caller-owned destination:

```go
ids := make([]uint32, 0, 512)
ids, err = model.AppendIDs(ids, "hi")
if err != nil {
	return err
}
```

`Tokenize` and `AppendIDs` accept valid UTF-8 only. An invalid Go string
returns `bpe.ErrInvalidUTF8`; `AppendIDs` leaves its destination unchanged in
that case.

The methods process exactly one piece. Calling `Tokenize("a")` and then
`Tokenize("b")` can never apply an `a b` merge. A future `Tokenizer` facade
will be responsible for producing the pieces from raw text.

## Merge behavior

For every initial adjacent pair, the model selects the merge with the smallest
rank. After a merge, it considers the newly adjacent pair on either side. Ties
are resolved by the earlier position in the piece. This global priority is
essential: applying `merges.txt` once in file order is not equivalent.

The implementation is a scalar heap-based oracle. It deliberately favors
exactness over micro-optimizations and will be the differential reference for
future incremental and SIMD implementations.

## Supported options

`ModelOptions` represents deterministic Hugging Face BPE model options:

| Field | Behavior |
| --- | --- |
| `UnknownToken` | Vocab value emitted for an unknown symbol. Empty disables it. |
| `ContinuingSubwordPrefix` | Prepended to every initial symbol after the first. |
| `EndOfWordSuffix` | Appended to the last initial symbol. |
| `FuseUnknown` | Coalesces consecutive unknown symbols into one unknown token. |
| `ByteFallback` | Uses vocab entries such as `<0xC3>` and `<0xA9>` before `UnknownToken`. |
| `IgnoreMerges` | Returns an exact whole-piece vocabulary hit before normal BPE merging. |

`dropout` is intentionally not exposed. This package is for deterministic
inference; a serialized model that requires BPE dropout must be rejected by a
higher-level tokenizer loader.

Byte fallback records one byte of offset for every fallback token, consistent
with Hugging Face BPE behavior.

## Model-data validation

The parser intentionally accepts the conventional subset required for a sound,
deterministic model:

- `vocab.json` must be a JSON object mapping strings to unsigned decimal
  integers in the `uint32` range;
- each ID must be unique;
- `merges.txt` records must contain exactly two non-empty tokens separated by
  one ASCII space; `#version` lines are ignored;
- each merge component and the token formed by that merge must exist in the
  vocabulary;
- merge pairs must not be duplicated;
- a configured `UnknownToken` must be in the vocabulary;
- when `ContinuingSubwordPrefix` is configured, each merge's right token must
  begin with it.

Violations wrap one of `ErrInvalidVocabulary`, `ErrInvalidMerge`,
`ErrDuplicateTokenID`, or `ErrUnknownTokenNotInVocab`, so callers can use
`errors.Is`.

## Lookup

```go
id, ok := model.TokenToID("hi")
token, ok := model.IDToToken(id)
size := model.VocabSize()
```

The model never exposes its mutable internal maps. These methods are the
stable lookup surface for production callers.
