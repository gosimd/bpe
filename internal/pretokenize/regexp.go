// Package pretokenize adapts pre-tokenization engines to BPE-owned contracts.
// Boundary detection does not normalize text or perform ByteLevel mapping.
package pretokenize

import (
	"errors"
	"fmt"
	"regexp"
	"unicode/utf8"

	"github.com/gosimd/pretok"
)

var (
	ErrUnsupportedPattern = errors.New("bpe: unsupported pre-tokenization pattern")
	ErrUninitialized      = errors.New("bpe: uninitialized regexp pre-tokenizer")
	ErrInvalidUTF8        = errors.New("bpe: pre-tokenizer input is not valid UTF-8")
)

// Regexp owns an immutable compiled expression. Execution state is call-local.
type Regexp struct {
	compiled *pretok.Regexp
	fallback *regexp.Regexp
}

// NewRegexp prefers pretok, falling back only for an unsupported expression.
func NewRegexp(pattern string) (*Regexp, error) {
	compiled, err := pretok.Compile(pattern)
	if err != nil {
		if errors.Is(err, pretok.ErrUnsupportedPattern) {
			fallback, compileErr := regexp.Compile(pattern)
			if compileErr != nil {
				return nil, fmt.Errorf("%w: Go regexp fallback for %q: %w", ErrUnsupportedPattern, pattern, compileErr)
			}
			return &Regexp{fallback: fallback}, nil
		}
		return nil, fmt.Errorf("compile pre-tokenization pattern: %v", err)
	}
	return &Regexp{compiled: compiled}, nil
}

// Split calls emit for each isolated match and maximal unmatched gap, in order.
// Initialization and the entire input are validated before the first emission.
// The callback form lets facades append their own span types without an unsafe
// slice conversion or intermediate span buffer. Neither input nor emit is kept.
func (r *Regexp) Split(input string, emit func(start, end int)) error {
	if r == nil || r.compiled == nil && r.fallback == nil {
		return ErrUninitialized
	}
	if r.fallback != nil {
		if !utf8.ValidString(input) {
			return ErrInvalidUTF8
		}
		previous := 0
		// Scan the complete input: restarting on suffixes changes anchors and
		// word boundaries. Empty matches are boundaries, never empty pieces.
		for _, span := range r.fallback.FindAllStringIndex(input, -1) {
			if span[0] > previous {
				emit(previous, span[0])
			}
			if span[1] > span[0] {
				emit(span[0], span[1])
			}
			previous = span[1]
		}
		if previous < len(input) {
			emit(previous, len(input))
		}
		return nil
	}
	cursor, err := r.compiled.Cursor(input)
	if err != nil {
		if errors.Is(err, pretok.ErrInvalidUTF8) {
			return ErrInvalidUTF8
		}
		return fmt.Errorf("scan pre-tokenization pattern: %v", err)
	}
	// Cursor emits matches only. Restore gaps exactly as pretok.Regexp.Split
	// does; dropping gaps would discard input bytes for partial-match patterns.
	previous := 0
	for span, ok := cursor.Next(); ok; span, ok = cursor.Next() {
		if span.Start > previous {
			emit(previous, span.Start)
		}
		emit(span.Start, span.End)
		previous = span.End
	}
	if previous < len(input) {
		emit(previous, len(input))
	}
	return nil
}
