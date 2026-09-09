// Package tokenizer joins a BPE model with a fixed GPT-2 ByteLevel
// pre-tokenizer. Public API and validation belong to package bpe.
package tokenizer

import (
	"strings"
	"sync"
	"unsafe"

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
	scratchPool    sync.Pool
}

// Scratch owns all temporary storage used by ID-only tokenization. A value is
// borrowed for one AppendIDs call and reused for every pre-tokenized piece.
type Scratch struct {
	working     []byte
	pieces      []bytelevel.Piece
	mapped      []byte
	segments    []addedtoken.Segment
	spare       []addedtoken.Segment
	sequenceIDs []uint32
	model       model.Scratch
}

// New constructs a ByteLevel tokenizer.
func New(model *model.Model, addPrefixSpace bool, definitions []addedtoken.Definition, processor postprocess.Processor) *Tokenizer {
	addedValues := make(map[uint32]string, len(definitions))
	for _, definition := range definitions {
		addedValues[definition.ID] = definition.Content
	}
	tokenizer := &Tokenizer{
		model:          model,
		addPrefixSpace: addPrefixSpace,
		addedTokens:    addedtoken.New(definitions),
		addedValues:    addedValues,
		postProcessor:  processor,
	}
	tokenizer.scratchPool.New = func() any { return new(Scratch) }
	return tokenizer
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
	if text == "" && (t.postProcessor == nil || !t.postProcessor.AffectsIDs()) {
		return dst
	}
	scratch := t.scratchPool.Get().(*Scratch)
	dst = t.AppendIDsWithScratch(dst, text, addSpecialTokens, scratch)
	t.scratchPool.Put(scratch)
	return dst
}

// AppendIDsWithScratch is the allocation-reusing implementation behind
// AppendIDs. The caller must provide exclusively owned scratch.
func (t *Tokenizer) AppendIDsWithScratch(dst []uint32, text string, addSpecialTokens bool, scratch *Scratch) []uint32 {
	if t.postProcessor != nil && t.postProcessor.AffectsIDs() {
		scratch.sequenceIDs = t.appendSequenceIDs(scratch.sequenceIDs[:0], text, scratch)
		return t.postProcessor.AppendIDs(dst, scratch.sequenceIDs, nil, false, addSpecialTokens)
	}
	return t.appendSequenceIDs(dst, text, scratch)
}

func (t *Tokenizer) appendSequenceIDs(dst []uint32, text string, scratch *Scratch) []uint32 {
	if t.addedTokens.Empty() {
		return t.appendOrdinary(dst, text, scratch)
	}
	scratch.segments, scratch.spare = t.addedTokens.SplitWithScratch(text, scratch.segments, scratch.spare)
	for _, segment := range scratch.segments {
		if segment.Added {
			dst = append(dst, segment.ID)
			continue
		}
		dst = t.appendOrdinary(dst, text[segment.Start:segment.End], scratch)
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
			start, end := mapped.RawRange(token.Start, token.End)
			dst = append(dst, Token{
				ID:         token.ID,
				Value:      value,
				Start:      base + start,
				End:        base + end,
				SequenceID: postprocess.NoSequence,
			})
		}
	}
	return dst
}

func (t *Tokenizer) appendOrdinary(dst []uint32, text string, scratch *Scratch) []uint32 {
	if text == "" {
		return dst
	}
	working := text
	if t.addPrefixSpace && !strings.HasPrefix(text, " ") {
		scratch.working = append(scratch.working[:0], ' ')
		scratch.working = append(scratch.working, text...)
		working = bytesString(scratch.working)
	}
	scratch.pieces = bytelevel.AppendSplit(scratch.pieces[:0], working)
	for _, piece := range scratch.pieces {
		scratch.mapped = bytelevel.AppendMap(scratch.mapped[:0], working[piece.Start:piece.End])
		dst = t.model.AppendIDsWithScratch(dst, bytesString(scratch.mapped), &scratch.model)
	}
	return dst
}

// bytesString creates a call-scoped view. Model tokenization never retains its
// input, and the backing slice is not changed until that call returns.
func bytesString(value []byte) string {
	return unsafe.String(unsafe.SliceData(value), len(value))
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
