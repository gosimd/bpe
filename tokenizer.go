package bpe

import (
	"fmt"
	"unicode/utf8"

	"github.com/gosimd/bpe/internal/addedtoken"
	"github.com/gosimd/bpe/internal/pretokenize"
	internaltokenizer "github.com/gosimd/bpe/internal/tokenizer"
)

// TokenizerOptions configures the supported tokenizer pipeline.
type TokenizerOptions struct {
	// AddPrefixSpace has the same meaning as Hugging Face ByteLevel's
	// add_prefix_space option. NewTokenizer uses true, matching the Hugging
	// Face ByteLevel default. NewTokenizerWithOptions permits false.
	AddPrefixSpace bool

	// PreTokenizer optionally replaces GPT-2 regex splitting. Its isolated
	// matches and unmatched gaps pass through ByteLevel with use_regex=false;
	// AddPrefixSpace applies separately to each piece. Nil keeps GPT-2 behavior.
	// Construct it with NewRegexpPreTokenizer. The compiled state is immutable
	// and shared, not copied; dependency types remain internal.
	PreTokenizer *RegexpPreTokenizer

	// AddedTokens are immutable definitions matched before ByteLevel and BPE.
	// The constructor validates and copies the slice.
	AddedTokens []AddedToken

	// PostProcessor is an optional TemplateProcessing or ByteLevelPostProcessor
	// configuration. The constructor validates and copies it.
	PostProcessor PostProcessor
}

// Tokenizer extracts configured AddedTokens, then tokenizes remaining raw
// UTF-8 input with ByteLevel and an immutable BPE Model. The default splitter
// is GPT-2; TokenizerOptions.PreTokenizer selects regexp splitting instead.
//
// It is safe for concurrent use. This API intentionally does not load
// tokenizer.json or implement normalizers, padding, truncation, or decoding.
type Tokenizer struct {
	impl *internaltokenizer.Tokenizer
}

// NewTokenizer constructs a GPT-2 ByteLevel tokenizer with Hugging Face's
// default AddPrefixSpace behavior enabled.
func NewTokenizer(model *Model) (*Tokenizer, error) {
	return NewTokenizerWithOptions(model, TokenizerOptions{AddPrefixSpace: true})
}

// NewTokenizerWithOptions constructs a ByteLevel tokenizer with explicit
// pre-tokenizer options.
func NewTokenizerWithOptions(model *Model, options TokenizerOptions) (*Tokenizer, error) {
	if model == nil || model.impl == nil {
		return nil, ErrNilModel
	}
	var splitter *pretokenize.Regexp
	if options.PreTokenizer != nil {
		if options.PreTokenizer.impl == nil {
			return nil, ErrUninitializedPreTokenizer
		}
		splitter = options.PreTokenizer.impl
	}
	definitions, err := validateAddedTokens(model, options.AddedTokens)
	if err != nil {
		return nil, err
	}
	processor, err := compilePostProcessor(options.PostProcessor)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{impl: internaltokenizer.New(model.impl, options.AddPrefixSpace, definitions, processor, splitter)}, nil
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

// Encode tokenizes raw UTF-8 text and enables template special tokens.
func (t *Tokenizer) Encode(text string) (Encoding, error) {
	return t.encode(text, "", false, true)
}

// EncodePair tokenizes two raw UTF-8 inputs and enables template special
// tokens. Offsets for each token are relative to the input identified by its
// SequenceIDs entry.
func (t *Tokenizer) EncodePair(first, second string) (Encoding, error) {
	return t.encode(first, second, true, true)
}

// EncodeOptions configures one EncodeWithOptions or EncodePairWithOptions
// call. Its zero value disables template special-token insertion.
type EncodeOptions struct {
	AddSpecialTokens bool
}

// EncodeWithOptions tokenizes one raw UTF-8 input with explicit call options.
func (t *Tokenizer) EncodeWithOptions(text string, options EncodeOptions) (Encoding, error) {
	return t.encode(text, "", false, options.AddSpecialTokens)
}

// EncodePairWithOptions tokenizes two raw UTF-8 inputs with explicit call
// options.
func (t *Tokenizer) EncodePairWithOptions(first, second string, options EncodeOptions) (Encoding, error) {
	return t.encode(first, second, true, options.AddSpecialTokens)
}

func (t *Tokenizer) encode(first, second string, pair, addSpecialTokens bool) (Encoding, error) {
	if !utf8.ValidString(first) || pair && !utf8.ValidString(second) {
		return Encoding{}, ErrInvalidUTF8
	}
	result, err := t.impl.Encode(first, second, pair, addSpecialTokens)
	if err != nil {
		return Encoding{}, err
	}
	encoding := Encoding{
		IDs:               make([]uint32, len(result)),
		TypeIDs:           make([]uint32, len(result)),
		Tokens:            make([]string, len(result)),
		Offsets:           make([]Offset, len(result)),
		SpecialTokensMask: make([]uint32, len(result)),
		SequenceIDs:       make([]int, len(result)),
	}
	for i, token := range result {
		encoding.IDs[i] = token.ID
		encoding.TypeIDs[i] = token.TypeID
		encoding.Tokens[i] = token.Value
		encoding.Offsets[i] = Offset{Start: token.Start, End: token.End}
		if token.Special {
			encoding.SpecialTokensMask[i] = 1
		}
		encoding.SequenceIDs[i] = token.SequenceID
	}
	return encoding, nil
}

// AppendIDs tokenizes raw UTF-8 text and appends its IDs to dst.
//
// It executes an ID-only pipeline and does not construct Encoding's parallel
// metadata slices, including when TemplateProcessing is configured. On invalid
// UTF-8 input, it returns dst unchanged together with ErrInvalidUTF8.
func (t *Tokenizer) AppendIDs(dst []uint32, text string) ([]uint32, error) {
	return t.AppendIDsWithOptions(dst, text, EncodeOptions{AddSpecialTokens: true})
}

// AppendIDsWithOptions is AppendIDs with explicit control over template
// special-token insertion.
func (t *Tokenizer) AppendIDsWithOptions(dst []uint32, text string, options EncodeOptions) ([]uint32, error) {
	if !utf8.ValidString(text) {
		return dst, ErrInvalidUTF8
	}
	return t.impl.AppendIDs(dst, text, options.AddSpecialTokens)
}
