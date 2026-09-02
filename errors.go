package bpe

import (
	"errors"

	"github.com/gosimd/bpe/internal/model"
)

// Errors returned while constructing or using a model or tokenizer.
var (
	ErrInvalidVocabulary      = model.ErrInvalidVocabulary
	ErrInvalidMerge           = model.ErrInvalidMerge
	ErrDuplicateTokenID       = model.ErrDuplicateTokenID
	ErrUnknownTokenNotInVocab = model.ErrUnknownTokenNotInVocab
	ErrInvalidUTF8            = model.ErrInvalidUTF8
	ErrNilModel               = errors.New("bpe: nil model")
	ErrInvalidAddedToken      = errors.New("bpe: invalid added token")
)
