package bpe

import "github.com/gosimd/bpe/internal/model"

// Errors returned while parsing or compiling a BPE model.
var (
	ErrInvalidVocabulary      = model.ErrInvalidVocabulary
	ErrInvalidMerge           = model.ErrInvalidMerge
	ErrDuplicateTokenID       = model.ErrDuplicateTokenID
	ErrUnknownTokenNotInVocab = model.ErrUnknownTokenNotInVocab
	ErrInvalidUTF8            = model.ErrInvalidUTF8
)
