// Package tokenizer joins a BPE model with a configurable ByteLevel
// pre-tokenizer. Public API and validation belong to package bpe.
package tokenizer

import (
	"sync"

	"github.com/gosimd/bpe/internal/addedtoken"
	"github.com/gosimd/bpe/internal/model"
	"github.com/gosimd/bpe/internal/postprocess"
	"github.com/gosimd/bpe/internal/pretokenize"
)

// Token is a post-processable token with byte offsets in the original input.
type Token = postprocess.Token

// Tokenizer is immutable and safe for concurrent use.
type Tokenizer struct {
	model         *model.Model
	addedTokens   *addedtoken.Matcher
	postProcessor postprocess.Processor
	preTokenizer  *pretokenize.Pipeline
	scratchPool   sync.Pool
}

// Scratch owns BPE, AddedToken and template storage for ID-only tokenization.
// It is borrowed for one AppendIDs call and reused for every piece. The
// pre-tokenization dependency owns its separate scratch pool.
type Scratch struct {
	segments    []addedtoken.Segment
	spare       []addedtoken.Segment
	sequenceIDs []uint32
	model       model.Scratch
}

// New constructs a ByteLevel tokenizer.
func New(model *model.Model, definitions []addedtoken.Definition, processor postprocess.Processor, pipeline *pretokenize.Pipeline) *Tokenizer {
	tokenizer := &Tokenizer{
		model:         model,
		addedTokens:   addedtoken.New(definitions),
		postProcessor: processor,
		preTokenizer:  pipeline,
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
	err := t.preTokenizer.Walk(text, func(piece pretokenize.Piece) error {
		for _, token := range t.model.Tokenize(piece.Text()) {
			value, _ := t.model.IDToToken(token.ID)
			start, end, err := piece.OriginalRange(token.Start, token.End)
			if err != nil {
				return err
			}
			dst = append(dst, Token{
				ID:         token.ID,
				Value:      value,
				Start:      base + start,
				End:        base + end,
				SequenceID: postprocess.NoSequence,
			})
		}
		return nil
	})
	return dst, err
}

func (t *Tokenizer) appendOrdinary(dst []uint32, text string, scratch *Scratch) ([]uint32, error) {
	err := t.preTokenizer.WalkTextUnsafe(text, func(piece string) error {
		dst = t.model.AppendIDsWithScratch(dst, piece, &scratch.model)
		return nil
	})
	return dst, err
}
