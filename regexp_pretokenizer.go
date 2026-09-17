package bpe

import (
	"errors"

	"github.com/gosimd/bpe/internal/pretokenize"
)

// RegexpPreTokenizer splits text using a supported pre-tokenization regexp.
// It is immutable, safe for concurrent Split calls with separate destinations,
// and retains neither input nor output buffers. Construct it with
// NewRegexpPreTokenizer; nil and zero values return ErrUninitializedPreTokenizer.
//
// Split alone does not perform ByteLevel mapping, normalization, AddedToken
// matching or BPE. Set TokenizerOptions.PreTokenizer to use it in that pipeline.
type RegexpPreTokenizer struct {
	impl *pretokenize.Regexp
}

// NewRegexpPreTokenizer compiles pattern once through gosimd/pretok. Pass the
// exact regexp string after JSON decoding. Only pretok.ErrUnsupportedPattern
// triggers a fallback to Go regexp. If that compilation also fails, the error
// wraps ErrUnsupportedPattern and the Go syntax error. Patterns are not rewritten.
// The fallback has Go regexp semantics, not general Hugging Face regexp parity.
func NewRegexpPreTokenizer(pattern string) (*RegexpPreTokenizer, error) {
	impl, err := pretokenize.NewRegexp(pattern)
	if err != nil {
		return nil, err
	}
	return &RegexpPreTokenizer{impl: impl}, nil
}

// Split appends byte offsets of isolated matches and maximal unmatched gaps to
// dst, covering text without changing its bytes. Offsets are half-open and
// relative to this text, not to any previously appended input. Pass dst[:0] to
// reuse storage. Empty text adds no offsets. On error, dst and its backing
// storage are unchanged; initialization errors take precedence over UTF-8 errors.
// Empty matches create boundaries without emitting empty pieces. Split uses
// isolated matches with invert=false, not all HF split modes. Patterns supported
// by pretok use its semantics; all other accepted patterns use Go regexp.
func (p *RegexpPreTokenizer) Split(text string, dst []Offset) ([]Offset, error) {
	if p == nil || p.impl == nil {
		return dst, ErrUninitializedPreTokenizer
	}
	err := p.impl.Split(text, func(start, end int) {
		dst = append(dst, Offset{Start: start, End: end})
	})
	if errors.Is(err, pretokenize.ErrInvalidUTF8) {
		return dst, ErrInvalidUTF8
	}
	return dst, err
}
