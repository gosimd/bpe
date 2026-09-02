package bpe

import (
	"unicode/utf8"

	internaltokenizer "github.com/gosimd/bpe/internal/tokenizer"
)

// TokenizerOptions configures the supported GPT-2 ByteLevel pre-tokenizer.
//
// AddPrefixSpace has the same meaning as Hugging Face ByteLevel's
// add_prefix_space option. NewTokenizer uses true, matching the Hugging Face
// ByteLevel default. NewTokenizerWithOptions permits an explicit false value.
type TokenizerOptions struct {
	AddPrefixSpace bool
}

// Tokenizer tokenizes raw UTF-8 input with the GPT-2 ByteLevel pre-tokenizer
// and an immutable BPE Model.
//
// It is safe for concurrent use. This initial API intentionally does not load
// tokenizer.json or implement normalizers, added tokens, post-processing,
// padding, truncation, or decoding.
type Tokenizer struct {
	impl *internaltokenizer.Tokenizer
}

// NewTokenizer constructs a GPT-2 ByteLevel tokenizer with Hugging Face's
// default AddPrefixSpace behavior enabled.
func NewTokenizer(model *Model) (*Tokenizer, error) {
	return NewTokenizerWithOptions(model, TokenizerOptions{AddPrefixSpace: true})
}

// NewTokenizerWithOptions constructs a GPT-2 ByteLevel tokenizer with explicit
// pre-tokenizer options.
func NewTokenizerWithOptions(model *Model, options TokenizerOptions) (*Tokenizer, error) {
	if model == nil || model.impl == nil {
		return nil, ErrNilModel
	}
	return &Tokenizer{impl: internaltokenizer.New(model.impl, options.AddPrefixSpace)}, nil
}

// Encode tokenizes raw UTF-8 text and returns IDs with offsets in the original
// input string.
func (t *Tokenizer) Encode(text string) (Encoding, error) {
	if !utf8.ValidString(text) {
		return Encoding{}, ErrInvalidUTF8
	}
	result := t.impl.Encode(text)
	encoding := Encoding{
		IDs:     make([]uint32, len(result)),
		Offsets: make([]Offset, len(result)),
	}
	for i, token := range result {
		encoding.IDs[i] = token.ID
		encoding.Offsets[i] = Offset{Start: token.Start, End: token.End}
	}
	return encoding, nil
}

// AppendIDs tokenizes raw UTF-8 text and appends its IDs to dst.
//
// It avoids constructing offsets. On invalid UTF-8 input, it returns dst
// unchanged together with ErrInvalidUTF8.
func (t *Tokenizer) AppendIDs(dst []uint32, text string) ([]uint32, error) {
	if !utf8.ValidString(text) {
		return dst, ErrInvalidUTF8
	}
	return t.impl.AppendIDs(dst, text), nil
}
