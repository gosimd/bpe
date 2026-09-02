package bpe

import (
	"fmt"
	"os"
	"unicode/utf8"

	internalmodel "github.com/gosimd/bpe/internal/model"
)

// Merge is one ordered BPE merge rule. Its rank is its index in the merges
// slice: lower indexes have higher priority.
type Merge struct {
	Left  string
	Right string
}

// ModelOptions configures supported Hugging Face BPE model behavior.
//
// An empty UnknownToken disables unknown-token output, matching a BPE model
// without unk_token. Dropout is deliberately absent: Model is deterministic
// and intended for inference.
type ModelOptions struct {
	UnknownToken            string
	ContinuingSubwordPrefix string
	EndOfWordSuffix         string
	FuseUnknown             bool
	ByteFallback            bool
	IgnoreMerges            bool
}

// Model is an immutable, concurrently safe compiled BPE model.
//
// Construct it with NewModel, ParseModel, or OpenModel. Its methods accept one
// already pre-tokenized piece; merges never cross a call boundary.
type Model struct {
	impl *internalmodel.Model
}

// NewModel compiles vocab and ordered merges into an immutable BPE model.
//
// The input map and slice are copied. Every merge component and its resulting
// token must be present in vocab. Duplicate token IDs and duplicate merge pairs
// are rejected because they have ambiguous reverse-lookup or rank semantics.
func NewModel(vocab map[string]uint32, merges []Merge, options ModelOptions) (*Model, error) {
	internalMerges := make([]internalmodel.Merge, len(merges))
	for i, merge := range merges {
		internalMerges[i] = internalmodel.Merge{
			Left:  merge.Left,
			Right: merge.Right,
		}
	}

	impl, err := internalmodel.New(vocab, internalMerges, internalmodel.Options{
		UnknownToken:            options.UnknownToken,
		ContinuingSubwordPrefix: options.ContinuingSubwordPrefix,
		EndOfWordSuffix:         options.EndOfWordSuffix,
		FuseUnknown:             options.FuseUnknown,
		ByteFallback:            options.ByteFallback,
		IgnoreMerges:            options.IgnoreMerges,
	})
	if err != nil {
		return nil, err
	}
	return &Model{impl: impl}, nil
}

// ParseModel parses a Hugging Face-style vocab.json and merges.txt, then
// compiles them as if passed to NewModel.
func ParseModel(vocabJSON, mergesText []byte, options ModelOptions) (*Model, error) {
	vocab, err := internalmodel.ParseVocabulary(vocabJSON)
	if err != nil {
		return nil, err
	}
	merges, err := internalmodel.ParseMerges(mergesText)
	if err != nil {
		return nil, err
	}

	publicMerges := make([]Merge, len(merges))
	for i, merge := range merges {
		publicMerges[i] = Merge{Left: merge.Left, Right: merge.Right}
	}
	return NewModel(vocab, publicMerges, options)
}

// OpenModel reads a Hugging Face-style vocab.json and merges.txt from disk and
// compiles them as if passed to ParseModel.
func OpenModel(vocabFilename, mergesFilename string, options ModelOptions) (*Model, error) {
	vocabJSON, err := os.ReadFile(vocabFilename)
	if err != nil {
		return nil, fmt.Errorf("read vocabulary %q: %w", vocabFilename, err)
	}
	mergesText, err := os.ReadFile(mergesFilename)
	if err != nil {
		return nil, fmt.Errorf("read merges %q: %w", mergesFilename, err)
	}
	return ParseModel(vocabJSON, mergesText, options)
}

// Tokenize tokenizes one UTF-8 pre-tokenized piece.
//
// Offsets are half-open byte ranges relative to piece. Unknown symbols follow
// the model options: they use byte fallback when possible, otherwise
// UnknownToken when configured, otherwise they produce no token, as in Hugging
// Face BPE without an unk_token.
func (m *Model) Tokenize(piece string) ([]Token, error) {
	if !utf8.ValidString(piece) {
		return nil, ErrInvalidUTF8
	}

	result := m.impl.Tokenize(piece)
	tokens := make([]Token, len(result))
	for i, token := range result {
		value, _ := m.impl.IDToToken(token.ID)
		tokens[i] = Token{
			ID:     token.ID,
			Value:  value,
			Offset: Offset{Start: token.Start, End: token.End},
		}
	}
	return tokens, nil
}

// AppendIDs appends the IDs for one UTF-8 pre-tokenized piece to dst.
//
// It is the low-allocation counterpart to Tokenize. On an invalid UTF-8 input,
// it returns dst unchanged together with ErrInvalidUTF8.
func (m *Model) AppendIDs(dst []uint32, piece string) ([]uint32, error) {
	if !utf8.ValidString(piece) {
		return dst, ErrInvalidUTF8
	}
	return m.impl.AppendIDs(dst, piece), nil
}

// TokenToID returns the ID assigned to token.
func (m *Model) TokenToID(token string) (uint32, bool) {
	return m.impl.TokenToID(token)
}

// IDToToken returns the vocabulary token assigned to id.
func (m *Model) IDToToken(id uint32) (string, bool) {
	return m.impl.IDToToken(id)
}

// VocabSize returns the number of tokens in the model vocabulary.
func (m *Model) VocabSize() int {
	return m.impl.VocabSize()
}
