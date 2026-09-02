// Package model contains the compiled, scalar BPE implementation. It is not a
// public API; package bpe owns public validation and result types.
package model

import (
	"errors"
	"fmt"
	"strings"
)

var (
	ErrInvalidVocabulary      = errors.New("bpe: invalid vocabulary")
	ErrInvalidMerge           = errors.New("bpe: invalid merge")
	ErrDuplicateTokenID       = errors.New("bpe: duplicate token ID")
	ErrUnknownTokenNotInVocab = errors.New("bpe: unknown token is not in vocabulary")
	ErrInvalidUTF8            = errors.New("bpe: input is not valid UTF-8")
)

// Merge is one ordered BPE merge rule. Its index determines its rank.
type Merge struct {
	Left  string
	Right string
}

// Options configures supported deterministic Hugging Face BPE options.
type Options struct {
	UnknownToken            string
	ContinuingSubwordPrefix string
	EndOfWordSuffix         string
	FuseUnknown             bool
	ByteFallback            bool
	IgnoreMerges            bool
}

type pair struct {
	left  uint32
	right uint32
}

type merge struct {
	rank  uint32
	newID uint32
}

// Model owns immutable copies of its input vocabulary and merge rules.
type Model struct {
	vocab   map[string]uint32
	reverse map[uint32]string
	merges  map[pair]merge
	options Options
}

// New validates and compiles a model. It copies vocab and merges, so callers
// may safely reuse or change their inputs after New returns.
func New(vocab map[string]uint32, merges []Merge, options Options) (*Model, error) {
	compiledVocab := make(map[string]uint32, len(vocab))
	reverse := make(map[uint32]string, len(vocab))
	for token, id := range vocab {
		if previous, exists := reverse[id]; exists {
			return nil, fmt.Errorf("%w: ID %d is assigned to both %q and %q", ErrDuplicateTokenID, id, previous, token)
		}
		compiledVocab[token] = id
		reverse[id] = token
	}

	if options.UnknownToken != "" {
		if _, ok := compiledVocab[options.UnknownToken]; !ok {
			return nil, fmt.Errorf("%w: %q", ErrUnknownTokenNotInVocab, options.UnknownToken)
		}
	}

	compiledMerges := make(map[pair]merge, len(merges))
	for rank, rule := range merges {
		leftID, ok := compiledVocab[rule.Left]
		if !ok {
			return nil, fmt.Errorf("%w: rule %d left token %q is not in vocabulary", ErrInvalidMerge, rank+1, rule.Left)
		}
		rightID, ok := compiledVocab[rule.Right]
		if !ok {
			return nil, fmt.Errorf("%w: rule %d right token %q is not in vocabulary", ErrInvalidMerge, rank+1, rule.Right)
		}

		merged, err := mergedToken(rule.Left, rule.Right, options.ContinuingSubwordPrefix)
		if err != nil {
			return nil, fmt.Errorf("%w: rule %d: %v", ErrInvalidMerge, rank+1, err)
		}
		newID, ok := compiledVocab[merged]
		if !ok {
			return nil, fmt.Errorf("%w: rule %d result %q is not in vocabulary", ErrInvalidMerge, rank+1, merged)
		}

		key := pair{left: leftID, right: rightID}
		if _, exists := compiledMerges[key]; exists {
			return nil, fmt.Errorf("%w: rule %d duplicates pair %q %q", ErrInvalidMerge, rank+1, rule.Left, rule.Right)
		}
		compiledMerges[key] = merge{rank: uint32(rank), newID: newID}
	}

	return &Model{
		vocab:   compiledVocab,
		reverse: reverse,
		merges:  compiledMerges,
		options: options,
	}, nil
}

func mergedToken(left, right, prefix string) (string, error) {
	if prefix == "" {
		return left + right, nil
	}
	if !strings.HasPrefix(right, prefix) {
		return "", fmt.Errorf("right token %q does not start with continuing subword prefix %q", right, prefix)
	}
	return left + strings.TrimPrefix(right, prefix), nil
}

// TokenToID returns the ID assigned to token.
func (m *Model) TokenToID(token string) (uint32, bool) {
	id, ok := m.vocab[token]
	return id, ok
}

// IDToToken returns the vocabulary token assigned to id.
func (m *Model) IDToToken(id uint32) (string, bool) {
	token, ok := m.reverse[id]
	return token, ok
}

// VocabSize returns the number of entries in vocab.
func (m *Model) VocabSize() int {
	return len(m.vocab)
}
