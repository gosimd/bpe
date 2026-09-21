// Package pretokenize adapts gosimd/pretok without implementing text splitting,
// ByteLevel transformations, or offset alignment itself.
package pretokenize

import (
	"errors"
	"fmt"

	"github.com/gosimd/pretok"
)

var (
	ErrUninitialized = errors.New("bpe: uninitialized pre-tokenizer pipeline")
	ErrInvalidUTF8   = errors.New("bpe: pre-tokenizer input is not valid UTF-8")
	ErrUnsupported   = errors.New("bpe: unsupported pre-tokenizer configuration")
)

// Pipeline and Piece keep dependency types behind the internal boundary.
type Pipeline struct{ compiled *pretok.Pipeline }
type Piece struct{ value pretok.Piece }

// CompileJSON delegates the unmodified pre_tokenizer value to the dependency.
func CompileJSON(config []byte) (*Pipeline, error) {
	compiled, err := pretok.CompileJSON(config)
	if errors.Is(err, pretok.ErrUnsupportedPreTokenizer) || errors.Is(err, pretok.ErrUnsupportedPattern) {
		return nil, fmt.Errorf("%w: %w", ErrUnsupported, err)
	}
	if err != nil {
		return nil, translate(err)
	}
	return &Pipeline{compiled: compiled}, nil
}

// NewByteLevel compiles once; the dependency owns all three ByteLevel options.
// Offset trimming remains an Encoding post-processor, not a pre-tokenizer job.
func NewByteLevel(prefix, useRegex bool) (*Pipeline, error) {
	return CompileJSON([]byte(fmt.Sprintf(`{"type":"ByteLevel","add_prefix_space":%t,"trim_offsets":false,"use_regex":%t}`, prefix, useRegex)))
}

func (p *Pipeline) Walk(text string, emit func(Piece) error) error {
	if p == nil || p.compiled == nil {
		return ErrUninitialized
	}
	return translate(p.compiled.Walk(text, func(piece pretok.Piece) error { return emit(Piece{value: piece}) }))
}

// WalkTextUnsafe borrows strings only for the synchronous callback. BPE never retains
// them. Use the dependency's unsafe variant to avoid defensive per-piece copies.
func (p *Pipeline) WalkTextUnsafe(text string, emit func(string) error) error {
	if p == nil || p.compiled == nil {
		return ErrUninitialized
	}
	return translate(p.compiled.WalkTextUnsafe(text, emit))
}

// Text and range mapping are valid only during Walk's callback.
func (p Piece) Text() string { return p.value.Text() }
func (p Piece) OriginalRange(start, end int) (int, int, error) {
	span, err := p.value.OriginalRange(start, end)
	return span.Start, span.End, err
}

func translate(err error) error {
	switch {
	case errors.Is(err, pretok.ErrUninitialized):
		return ErrUninitialized
	case errors.Is(err, pretok.ErrInvalidUTF8):
		return ErrInvalidUTF8
	default:
		return err
	}
}
