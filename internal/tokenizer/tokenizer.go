// Package tokenizer joins a BPE model with a configurable ByteLevel
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
	"github.com/gosimd/bpe/internal/pretokenize"
)

// Token is a post-processable token with byte offsets in the original input.
type Token = postprocess.Token

// Tokenizer is immutable and safe for concurrent use.
type Tokenizer struct {
	model          *model.Model
	addPrefixSpace bool
	addedTokens    *addedtoken.Matcher
	postProcessor  postprocess.Processor
	preTokenizer   *pretokenize.Regexp
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
func New(model *model.Model, addPrefixSpace bool, definitions []addedtoken.Definition, processor postprocess.Processor, splitter *pretokenize.Regexp) *Tokenizer {
	tokenizer := &Tokenizer{
		model:          model,
		addPrefixSpace: addPrefixSpace,
		addedTokens:    addedtoken.New(definitions),
		postProcessor:  processor,
		preTokenizer:   splitter,
	}
	tokenizer.scratchPool.New = func() any { return new(Scratch) }
	return tokenizer
}

// Encode tokenizes and post-processes one or two raw UTF-8 inputs.
func (t *Tokenizer) Encode(first, second string, pair, addSpecialTokens bool) ([]Token, error) {
	firstTokens, err := t.encodeSequence(first)
	if err != nil {
		return nil, err
	}
	var secondTokens []Token
	if pair {
		secondTokens, err = t.encodeSequence(second)
		if err != nil {
			return nil, err
		}
	}
	if t.postProcessor == nil {
		return postprocess.Merge(firstTokens, secondTokens, pair), nil
	}
	return t.postProcessor.Process(firstTokens, secondTokens, pair, addSpecialTokens), nil
}

func (t *Tokenizer) encodeSequence(text string) ([]Token, error) {
	if text == "" {
		return nil, nil
	}
	if t.addedTokens.Empty() {
		return t.encodeOrdinary(nil, text, 0)
	}
	segments := t.addedTokens.Split(text)
	result := make([]Token, 0, len(segments))
	for _, segment := range segments {
		if segment.Added {
			// HF retains the full matched text, including LStrip/RStrip bytes.
			// ByteLevel post-processing must see those bytes to trim offsets.
			result = append(result, Token{ID: segment.ID, Value: text[segment.Start:segment.End], Start: segment.Start, End: segment.End})
			continue
		}
		var err error
		result, err = t.encodeOrdinary(result, text[segment.Start:segment.End], segment.Start)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// AppendIDs appends IDs for raw UTF-8 input to dst.
func (t *Tokenizer) AppendIDs(dst []uint32, text string, addSpecialTokens bool) ([]uint32, error) {
	if text == "" && (t.postProcessor == nil || !t.postProcessor.AffectsIDs()) {
		return dst, nil
	}
	scratch := t.scratchPool.Get().(*Scratch)
	dst, err := t.AppendIDsWithScratch(dst, text, addSpecialTokens, scratch)
	t.scratchPool.Put(scratch)
	return dst, err
}

// AppendIDsWithScratch is the allocation-reusing implementation behind
// AppendIDs. The caller must provide exclusively owned scratch.
func (t *Tokenizer) AppendIDsWithScratch(dst []uint32, text string, addSpecialTokens bool, scratch *Scratch) ([]uint32, error) {
	if t.postProcessor != nil && t.postProcessor.AffectsIDs() {
		var err error
		scratch.sequenceIDs, err = t.appendSequenceIDs(scratch.sequenceIDs[:0], text, scratch)
		if err != nil {
			return dst, err
		}
		return t.postProcessor.AppendIDs(dst, scratch.sequenceIDs, nil, false, addSpecialTokens), nil
	}
	return t.appendSequenceIDs(dst, text, scratch)
}

func (t *Tokenizer) appendSequenceIDs(dst []uint32, text string, scratch *Scratch) ([]uint32, error) {
	if t.addedTokens.Empty() {
		return t.appendOrdinary(dst, text, scratch)
	}
	scratch.segments, scratch.spare = t.addedTokens.SplitWithScratch(text, scratch.segments, scratch.spare)
	for _, segment := range scratch.segments {
		if segment.Added {
			dst = append(dst, segment.ID)
			continue
		}
		var err error
		dst, err = t.appendOrdinary(dst, text[segment.Start:segment.End], scratch)
		if err != nil {
			return dst, err
		}
	}
	return dst, nil
}

func (t *Tokenizer) encodeOrdinary(dst []Token, text string, base int) ([]Token, error) {
	if text == "" {
		return dst, nil
	}
	if t.preTokenizer != nil {
		err := t.preTokenizer.Split(text, func(start, end int) {
			piece := text[start:end]
			var mapped bytelevel.MappedPiece
			if t.addPrefixSpace && !strings.HasPrefix(piece, " ") {
				mapped = bytelevel.MapWithPrefixSpace(piece)
			} else {
				mapped = bytelevel.Map(piece, 0, func(offset int) int { return offset })
			}
			dst = t.encodeMapped(dst, mapped, base+start)
		})
		return dst, err
	}
	working, rawAt := t.workingInput(text)
	for _, piece := range bytelevel.Split(working) {
		mapped := bytelevel.Map(working[piece.Start:piece.End], piece.Start, rawAt)
		dst = t.encodeMapped(dst, mapped, base)
	}
	return dst, nil
}

func (t *Tokenizer) encodeMapped(dst []Token, mapped bytelevel.MappedPiece, base int) []Token {
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
	return dst
}

func (t *Tokenizer) appendOrdinary(dst []uint32, text string, scratch *Scratch) ([]uint32, error) {
	if text == "" {
		return dst, nil
	}
	if t.preTokenizer != nil {
		err := t.preTokenizer.Split(text, func(start, end int) {
			working := t.appendWorkingInput(text[start:end], scratch)
			dst = t.appendMapped(dst, working, scratch)
		})
		return dst, err
	}
	working := t.appendWorkingInput(text, scratch)
	scratch.pieces = bytelevel.AppendSplit(scratch.pieces[:0], working)
	for _, piece := range scratch.pieces {
		dst = t.appendMapped(dst, working[piece.Start:piece.End], scratch)
	}
	return dst, nil
}

func (t *Tokenizer) appendWorkingInput(text string, scratch *Scratch) string {
	working := text
	if t.addPrefixSpace && !strings.HasPrefix(text, " ") {
		scratch.working = append(scratch.working[:0], ' ')
		scratch.working = append(scratch.working, text...)
		working = bytesString(scratch.working)
	}
	return working
}

func (t *Tokenizer) appendMapped(dst []uint32, working string, scratch *Scratch) []uint32 {
	scratch.mapped = bytelevel.AppendMap(scratch.mapped[:0], working)
	return t.model.AppendIDsWithScratch(dst, bytesString(scratch.mapped), &scratch.model)
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
