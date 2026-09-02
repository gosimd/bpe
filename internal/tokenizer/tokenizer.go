// Package tokenizer joins a BPE model with a fixed GPT-2 ByteLevel
// pre-tokenizer. Public API and validation belong to package bpe.
package tokenizer

import (
	"strings"

	"github.com/gosimd/bpe/internal/addedtoken"
	"github.com/gosimd/bpe/internal/bytelevel"
	"github.com/gosimd/bpe/internal/model"
	"github.com/gosimd/bpe/internal/postprocess"
)

// Token is a post-processable token with byte offsets in the original input.
type Token = postprocess.Token

// Tokenizer is immutable and safe for concurrent use.
type Tokenizer struct {
	model          *model.Model
	addPrefixSpace bool
	addedTokens    *addedtoken.Matcher
	addedValues    map[uint32]string
	postProcessor  postprocess.Processor
}

// New constructs a ByteLevel tokenizer.
func New(model *model.Model, addPrefixSpace bool, definitions []addedtoken.Definition, processor postprocess.Processor) *Tokenizer {
	addedValues := make(map[uint32]string, len(definitions))
	for _, definition := range definitions {
		addedValues[definition.ID] = definition.Content
	}
	return &Tokenizer{
		model:          model,
		addPrefixSpace: addPrefixSpace,
		addedTokens:    addedtoken.New(definitions),
		addedValues:    addedValues,
		postProcessor:  processor,
	}
}

// Encode tokenizes and post-processes one or two raw UTF-8 inputs.
func (t *Tokenizer) Encode(first, second string, pair, addSpecialTokens bool) []Token {
	firstTokens := t.encodeSequence(first)
	var secondTokens []Token
	if pair {
		secondTokens = t.encodeSequence(second)
	}
	if t.postProcessor == nil {
		return postprocess.Merge(firstTokens, secondTokens, pair)
	}
	return t.postProcessor.Process(firstTokens, secondTokens, pair, addSpecialTokens)
}

func (t *Tokenizer) encodeSequence(text string) []Token {
	if text == "" {
		return nil
	}
	if t.addedTokens.Empty() {
		return t.encodeOrdinary(nil, text, 0)
	}
	segments := t.addedTokens.Split(text)
	result := make([]Token, 0, len(segments))
	for _, segment := range segments {
		if segment.Added {
			result = append(result, Token{ID: segment.ID, Value: t.addedValues[segment.ID], Start: segment.Start, End: segment.End})
			continue
		}
		result = t.encodeOrdinary(result, text[segment.Start:segment.End], segment.Start)
	}
	return result
}

// AppendIDs appends IDs for raw UTF-8 input to dst.
func (t *Tokenizer) AppendIDs(dst []uint32, text string, addSpecialTokens bool) []uint32 {
	if t.postProcessor != nil && t.postProcessor.AffectsIDs() {
		for _, token := range t.Encode(text, "", false, addSpecialTokens) {
			dst = append(dst, token.ID)
		}
		return dst
	}
	if text == "" {
		return dst
	}
	if t.addedTokens.Empty() {
		return t.appendOrdinary(dst, text)
	}
	for _, segment := range t.addedTokens.Split(text) {
		if segment.Added {
			dst = append(dst, segment.ID)
			continue
		}
		dst = t.appendOrdinary(dst, text[segment.Start:segment.End])
	}
	return dst
}

func (t *Tokenizer) encodeOrdinary(dst []Token, text string, base int) []Token {
	if text == "" {
		return dst
	}
	working, rawAt := t.workingInput(text)
	for _, piece := range bytelevel.Split(working) {
		mapped := bytelevel.Map(working[piece.Start:piece.End], piece.Start, rawAt)
		for _, token := range t.model.Tokenize(mapped.Text) {
			value, _ := t.model.IDToToken(token.ID)
			dst = append(dst, Token{
				ID:         token.ID,
				Value:      value,
				Start:      base + mapped.RawOffset(token.Start),
				End:        base + mapped.RawOffset(token.End),
				SequenceID: postprocess.NoSequence,
			})
		}
	}
	return dst
}

func (t *Tokenizer) appendOrdinary(dst []uint32, text string) []uint32 {
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
