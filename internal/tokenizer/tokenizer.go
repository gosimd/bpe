// Package tokenizer joins a BPE model with a fixed GPT-2 ByteLevel
// pre-tokenizer. Public API and validation belong to package bpe.
package tokenizer

import (
	"strings"

	"github.com/gosimd/bpe/internal/bytelevel"
	"github.com/gosimd/bpe/internal/model"
)

// Token is a token with byte offsets in the original raw input.
type Token struct {
	ID    uint32
	Start int
	End   int
}

// Tokenizer is immutable and safe for concurrent use.
type Tokenizer struct {
	model          *model.Model
	addPrefixSpace bool
}

// New constructs a ByteLevel tokenizer.
func New(model *model.Model, addPrefixSpace bool) *Tokenizer {
	return &Tokenizer{model: model, addPrefixSpace: addPrefixSpace}
}

// Encode tokenizes raw UTF-8 input.
func (t *Tokenizer) Encode(text string) []Token {
	if text == "" {
		return nil
	}
	working, rawAt := t.workingInput(text)
	pieces := bytelevel.Split(working)
	result := make([]Token, 0, len(pieces))
	for _, piece := range pieces {
		mapped := bytelevel.Map(working[piece.Start:piece.End], piece.Start, rawAt)
		for _, token := range t.model.Tokenize(mapped.Text) {
			result = append(result, Token{
				ID:    token.ID,
				Start: mapped.RawOffset(token.Start),
				End:   mapped.RawOffset(token.End),
			})
		}
	}
	return result
}

// AppendIDs appends IDs for raw UTF-8 input to dst.
func (t *Tokenizer) AppendIDs(dst []uint32, text string) []uint32 {
	if text == "" {
		return dst
	}
	working, rawAt := t.workingInput(text)
	for _, piece := range bytelevel.Split(working) {
		mapped := bytelevel.Map(working[piece.Start:piece.End], piece.Start, rawAt)
		dst = t.model.AppendIDs(dst, mapped.Text)
	}
	return dst
}

func (t *Tokenizer) workingInput(text string) (string, func(int) int) {
	if !t.addPrefixSpace || strings.HasPrefix(text, " ") {
		return text, func(offset int) int { return offset }
	}
	return " " + text, func(offset int) int {
		if offset <= 1 {
			return 0
		}
		return offset - 1
	}
}
