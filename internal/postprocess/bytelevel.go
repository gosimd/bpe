package postprocess

import (
	"unicode"
	"unicode/utf8"
)

const byteLevelSpace = '\u0120'

// ByteLevel implements Hugging Face's ByteLevel post-processor.
type ByteLevel struct {
	addPrefixSpace bool
	trimOffsets    bool
}

func NewByteLevel(addPrefixSpace, trimOffsets bool) *ByteLevel {
	return &ByteLevel{addPrefixSpace: addPrefixSpace, trimOffsets: trimOffsets}
}

func (*ByteLevel) AffectsIDs() bool { return false }

func (*ByteLevel) AppendIDs(dst, first, second []uint32, pair, _ bool) []uint32 {
	dst = append(dst, first...)
	if pair {
		dst = append(dst, second...)
	}
	return dst
}

func (p *ByteLevel) Process(first, second []Token, pair, _ bool) []Token {
	result := Merge(first, second, pair)
	if !p.trimOffsets {
		return result
	}
	trimOffsets(result[:len(first)], p.addPrefixSpace)
	if pair {
		trimOffsets(result[len(first):], p.addPrefixSpace)
	}
	return result
}

func trimOffsets(tokens []Token, addPrefixSpace bool) {
	for index := range tokens {
		token := &tokens[index]
		leading, trailing := trimCounts(token.Value)
		isFirst := index == 0 || token.Start == 0
		if isFirst && addPrefixSpace && leading == 1 {
			leading = 0
		}

		token.Start = min(token.Start+leading, token.End)
		if trailing > 0 && token.End >= trailing {
			token.End = max(token.End-trailing, token.Start)
		}
	}
}

func trimCounts(value string) (leading, trailing int) {
	for _, r := range value {
		if !isTrimSpace(r) {
			break
		}
		leading++
	}
	for index := len(value); index > 0; {
		r, size := utf8.DecodeLastRuneInString(value[:index])
		if !isTrimSpace(r) {
			break
		}
		trailing++
		index -= size
	}
	return leading, trailing
}

func isTrimSpace(r rune) bool {
	return r == byteLevelSpace || unicode.IsSpace(r)
}
