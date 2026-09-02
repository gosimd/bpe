# Post-processors

Post-processing runs after AddedToken matching, ByteLevel pre-tokenization, and
BPE. A `Tokenizer` can use one supported post-processor:

- `TemplateProcessing` inserts configured special tokens and assigns type IDs;
- `ByteLevelPostProcessor` trims whitespace represented in token offsets.

The processor configuration is validated and copied by
`NewTokenizerWithOptions`. The resulting tokenizer remains immutable and safe
for concurrent use.

## TemplateProcessing

The following configuration implements the usual BERT single and pair layout:

```go
processor := bpe.TemplateProcessing{
	Single: "[CLS] $A [SEP]",
	Pair:   "[CLS] $A [SEP] $B:1 [SEP]:1",
	SpecialTokens: []bpe.TemplateSpecialToken{
		bpe.NewTemplateSpecialToken("[CLS]", 101),
		bpe.NewTemplateSpecialToken("[SEP]", 102),
	},
}

tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
	AddPrefixSpace: false,
	PostProcessor:  processor,
})
```

`Single` is used by `Encode`; `Pair` is used by `EncodePair`:

```go
single, err := tokenizer.Encode("hello")
pair, err := tokenizer.EncodePair("question", "answer")
```

`Encode` and `EncodePair` insert template special tokens by default. To process
the selected template while omitting its special-token pieces, call the
options variants:

```go
pair, err := tokenizer.EncodePairWithOptions(
	"question",
	"answer",
	bpe.EncodeOptions{AddSpecialTokens: false},
)
```

This flag affects only tokens inserted by `TemplateProcessing`. It does not
disable matching of `TokenizerOptions.AddedTokens` already present in input.
`AppendIDs` follows `Encode` and therefore inserts configured template special
tokens. `AppendIDsWithOptions` can disable those insertions.

### Template syntax

Templates use the Hugging Face string syntax and are split on literal spaces:

- `$A`, `$a`, or `$` inserts the first sequence with type ID 0;
- `$B` or `$b` inserts the pair sequence with type ID 0;
- `$0`, `$1`, and other unsigned integers insert sequence A with that type ID;
- `$A:2` or `$B:3` selects both the sequence and type ID;
- `[CLS]` or `[SEP]:1` references a configured special-token identifier and
  optionally assigns its type ID.

A pair template must reference both A and B. A single template cannot reference
B. Every referenced special-token identifier must have exactly one definition.
Malformed pieces, missing definitions, duplicate identifiers, and mismatched
`IDs`/`Tokens` lengths return `ErrInvalidTemplate` while constructing the
tokenizer.

Empty `Single` and `Pair` fields select the pass-through defaults `$A` and
`$A:0 $B:1`. Templates may reorder or repeat an input sequence.

### Special-token definitions

The common definition emits one ID and uses the identifier as its token string:

```go
sep := bpe.NewTemplateSpecialToken("[SEP]", 102)
```

The complete form can emit multiple IDs, matching Hugging Face
`TemplateProcessing`:

```go
special := bpe.TemplateSpecialToken{
	Identifier: "<SPECIAL>",
	IDs:        []uint32{10, 11},
	Tokens:     []string{"<", "SPECIAL>"},
}
```

Template IDs and token strings are emitted verbatim. They do not have to be
entries in the BPE vocabulary or `AddedTokens`; the caller is responsible for
supplying the mapping intended by the model.

Inserted special tokens have offset `[0, 0)`, a special-token mask value of 1,
and `SequenceIDs` value `bpe.NoSequence`. Tokens copied from A or B retain
offsets relative to their own input and receive sequence IDs 0 or 1.

## ByteLevel post-processing

ByteLevel BPE token strings can contain `Ġ`, the visible representation of an
ASCII space. The ByteLevel post-processor can remove leading and trailing
whitespace from the reported offsets without changing IDs or token strings:

```go
tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
	AddPrefixSpace: true,
	PostProcessor:  bpe.NewByteLevelPostProcessor(),
})
```

`NewByteLevelPostProcessor` enables both `AddPrefixSpace` and `TrimOffsets`,
matching Hugging Face defaults. Use a struct literal for an explicit
configuration:

```go
bpe.ByteLevelPostProcessor{
	AddPrefixSpace: false,
	TrimOffsets:    true,
}
```

The post-processor's `AddPrefixSpace` setting controls the first-token trimming
exception. It is independent of `TokenizerOptions.AddPrefixSpace`, which
controls actual pre-tokenization. Normally the two values should match.

For pair input, each sequence is trimmed independently. Offsets remain relative
to the sequence selected by the corresponding `SequenceIDs` entry.

## Encoding fields

Every slice in `Encoding` has the same length:

- `IDs` contains emitted token IDs;
- `TypeIDs` contains type IDs assigned by the processor;
- `Tokens` contains vocabulary or configured template token strings;
- `Offsets` contains half-open byte ranges in the corresponding input;
- `SpecialTokensMask` is 1 only for template-inserted special tokens;
- `SequenceIDs` is 0 for A, 1 for B, and `NoSequence` for inserted tokens.

An AddedToken matched in the original input is not template-inserted: its
special-token mask remains 0 even when its `AddedToken.Special` metadata is
true.

## Current boundary

Only one post-processor can be configured. Hugging Face processor composition
(`Sequence`) and dedicated `BertProcessing`, `RobertaProcessing`, and other
processor types are not implemented. Common BERT layouts can already be
expressed with `TemplateProcessing`; unsupported serialized processor types
must not be approximated silently.

The behavior is covered by a reproducible differential fixture generated with
Hugging Face Tokenizers 0.22.2.
