package bpe

import (
	"errors"

	"github.com/gosimd/bpe/internal/model"
)

// Errors returned while parsing or compiling a BPE model.
var (
	ErrInvalidVocabulary      = model.ErrInvalidVocabulary
	ErrInvalidMerge           = model.ErrInvalidMerge
	ErrDuplicateTokenID       = model.ErrDuplicateTokenID
	ErrUnknownTokenNotInVocab = model.ErrUnknownTokenNotInVocab
	ErrInvalidUTF8            = model.ErrInvalidUTF8
	ErrNilModel               = errors.New("bpe: nil model")
)
