package bpe

import (
	"fmt"
	"unicode/utf8"

	"github.com/gosimd/bpe/internal/addedtoken"
	internaltokenizer "github.com/gosimd/bpe/internal/tokenizer"
)

// TokenizerOptions configures the supported GPT-2 ByteLevel pre-tokenizer.
type TokenizerOptions struct {
	// AddPrefixSpace has the same meaning as Hugging Face ByteLevel's
	// add_prefix_space option. NewTokenizer uses true, matching the Hugging
	// Face ByteLevel default. NewTokenizerWithOptions permits false.
	AddPrefixSpace bool

	// AddedTokens are immutable definitions matched before ByteLevel and BPE.
	// The constructor validates and copies the slice.
	AddedTokens []AddedToken
}

// Tokenizer extracts configured AddedTokens, then tokenizes remaining raw
// UTF-8 input with the GPT-2 ByteLevel pre-tokenizer and an immutable BPE Model.
//
// It is safe for concurrent use. This initial API intentionally does not load
// tokenizer.json or implement normalizers, post-processing, padding,
// truncation, or decoding.
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
	definitions, err := validateAddedTokens(model, options.AddedTokens)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{impl: internaltokenizer.New(model.impl, options.AddPrefixSpace, definitions)}, nil
}

func validateAddedTokens(model *Model, tokens []AddedToken) ([]addedtoken.Definition, error) {
	definitions := make([]addedtoken.Definition, len(tokens))
	contents := make(map[string]int, len(tokens))
	ids := make(map[uint32]int, len(tokens))
	for index, token := range tokens {
		if token.Content == "" {
			return nil, fmt.Errorf("%w: token %d has empty content", ErrInvalidAddedToken, index)
		}
		if !utf8.ValidString(token.Content) {
			return nil, fmt.Errorf("%w: token %d content is not valid UTF-8", ErrInvalidAddedToken, index)
		}
		if previous, ok := contents[token.Content]; ok {
			return nil, fmt.Errorf("%w: tokens %d and %d have duplicate content %q", ErrInvalidAddedToken, previous, index, token.Content)
		}
		if previous, ok := ids[token.ID]; ok {
			return nil, fmt.Errorf("%w: tokens %d and %d have duplicate ID %d", ErrInvalidAddedToken, previous, index, token.ID)
		}
		if id, ok := model.TokenToID(token.Content); ok && id != token.ID {
			return nil, fmt.Errorf("%w: content %q has model ID %d, not %d", ErrInvalidAddedToken, token.Content, id, token.ID)
		}
		if content, ok := model.IDToToken(token.ID); ok && content != token.Content {
			return nil, fmt.Errorf("%w: ID %d belongs to model token %q, not %q", ErrInvalidAddedToken, token.ID, content, token.Content)
		}
		contents[token.Content] = index
		ids[token.ID] = index
		definitions[index] = addedtoken.Definition{
			ID:         token.ID,
			Content:    token.Content,
			SingleWord: token.SingleWord,
			LStrip:     token.LStrip,
			RStrip:     token.RStrip,
			Normalized: token.Normalized,
		}
	}
	return definitions, nil
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
