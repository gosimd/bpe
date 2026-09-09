// Package bytelevel implements the GPT-2 ByteLevel pre-tokenizer. It is kept
// separate from BPE merging so future pre-tokenizer modes cannot alter Model
// semantics.
package bytelevel

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Piece is one isolated GPT-2 regex match in a possibly prefix-space-augmented
// input. Start and End are byte offsets in that working input.
type Piece struct {
	Start int
	End   int
}

// Split implements Hugging Face ByteLevel's standard GPT-2 pattern:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// The scanner is purpose-built because Go's regexp package does not support
// the negative lookahead in the penultimate alternative.
func Split(input string) []Piece {
	return AppendSplit(nil, input)
}

// AppendSplit appends GPT-2 regex pieces to dst. Callers may pass dst[:0] to
// reuse its backing array across independent inputs.
func AppendSplit(dst []Piece, input string) []Piece {
	pieces := dst
	for position := 0; position < len(input); {
		end := contractionEnd(input, position)
		if end > position {
			pieces = append(pieces, Piece{Start: position, End: end})
			position = end
			continue
		}

		if input[position] == ' ' {
			next := position + 1
			if next < len(input) {
				_, size := utf8.DecodeRuneInString(input[next:])
				if size > 0 && !unicode.IsSpace(runeAt(input, next)) {
					end = consumeClass(input, next)
					pieces = append(pieces, Piece{Start: position, End: end})
					position = end
					continue
				}
			}
		}

		r := runeAt(input, position)
		if unicode.IsSpace(r) {
			end = consumeSpace(input, position)
			if end < len(input) {
				// \s+(?!\S) leaves the last whitespace for the following optional
				// space branch when the run is followed by a non-whitespace rune.
				last := previousRuneStart(input, end)
				if last > position {
					pieces = append(pieces, Piece{Start: position, End: last})
					position = last
					continue
				}
			}
			pieces = append(pieces, Piece{Start: position, End: end})
			position = end
			continue
		}

		end = consumeClass(input, position)
		pieces = append(pieces, Piece{Start: position, End: end})
		position = end
	}
	return pieces
}

// AppendMap appends the GPT-2 visible-Unicode representation of input to dst.
// Unlike Map, it does not construct raw-offset boundaries and is intended for
// ID-only tokenization.
func AppendMap(dst []byte, input string) []byte {
	for index := range len(input) {
		dst = utf8.AppendRune(dst, byteToRune(input[index]))
	}
	return dst
}

func contractionEnd(input string, start int) int {
	if input[start] != '\'' {
		return start
	}
	for _, suffix := range [...]string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
		if strings.HasPrefix(input[start:], suffix) {
			return start + len(suffix)
		}
	}
	return start
}

func consumeClass(input string, start int) int {
	first := runeAt(input, start)
	class := runeClass(first)
	position := start + utf8.RuneLen(first)
	for position < len(input) {
		r := runeAt(input, position)
		if runeClass(r) != class {
			break
		}
		position += utf8.RuneLen(r)
	}
	return position
}

func consumeSpace(input string, start int) int {
	position := start
	for position < len(input) {
		r := runeAt(input, position)
		if !unicode.IsSpace(r) {
			break
		}
		position += utf8.RuneLen(r)
	}
	return position
}

func previousRuneStart(input string, end int) int {
	_, size := utf8.DecodeLastRuneInString(input[:end])
	return end - size
}

func runeAt(input string, offset int) rune {
	r, _ := utf8.DecodeRuneInString(input[offset:])
	return r
}

const (
	classLetter = iota
	classNumber
	classOther
	classSpace
)

func runeClass(r rune) int {
	if unicode.IsSpace(r) {
		return classSpace
	}
	if unicode.IsLetter(r) {
		return classLetter
	}
	if unicode.IsNumber(r) {
		return classNumber
	}
	return classOther
}

// MappedPiece is a ByteLevel-mapped piece with Hugging Face-compatible
// alignment ranges from transformed bytes back to the raw input.
type MappedPiece struct {
	Text       string
	alignments []alignment
}

type alignment struct {
	start int
	end   int
}

// Map maps every source byte to the GPT-2 visible-Unicode alphabet. offset is
// the source offset represented by input[0]. rawAt maps an input boundary to a
// boundary in the original user input; it accounts for an optional synthetic
// prefix space.
func Map(input string, offset int, rawAt func(int) int) MappedPiece {
	var builder strings.Builder
	builder.Grow(len(input) * 2)
	alignments := make([]alignment, 0, len(input)*2)
	for runeStart := 0; runeStart < len(input); {
		_, runeSize := utf8.DecodeRuneInString(input[runeStart:])
		runeEnd := runeStart + runeSize
		raw := alignment{
			start: rawAt(offset + runeStart),
			end:   rawAt(offset + runeEnd),
		}
		for byteIndex := runeStart; byteIndex < runeEnd; byteIndex++ {
			r := byteToRune(input[byteIndex])
			builder.WriteRune(r)
			for range utf8.RuneLen(r) {
				alignments = append(alignments, raw)
			}
		}
		runeStart = runeEnd
	}
	return MappedPiece{Text: builder.String(), alignments: alignments}
}

// RawRange maps a non-empty range in Text to Hugging Face-compatible byte
// offsets in the original input. Every transformed byte carries the complete
// source UTF-8 scalar's alignment, so adjacent model tokens may overlap.
func (p MappedPiece) RawRange(start, end int) (int, int) {
	if start < 0 || start >= end || end > len(p.alignments) {
		panic("bytelevel: invalid BPE offset range")
	}
	return p.alignments[start].start, p.alignments[end-1].end
}

func byteToRune(b byte) rune {
	switch {
	case b <= 0x20:
		return 0x100 + rune(b)
	case b >= 0x7F && b <= 0xA0:
		return 0x121 + rune(b) - 0x7F
	case b == 0xAD:
		return 0x143
	default:
		return rune(b)
	}
}
