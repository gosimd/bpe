// Package bytelevel implements the GPT-2 ByteLevel pre-tokenizer. It is kept
// separate from BPE merging so future pre-tokenizer modes cannot alter Model
// semantics.
package bytelevel

import (
	"sort"
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
	pieces := make([]Piece, 0, len(input)/4+1)
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

// MappedPiece is a ByteLevel-mapped piece with a translation from transformed
// byte boundaries back to raw input byte boundaries.
type MappedPiece struct {
	Text       string
	boundaries []boundary
}

type boundary struct {
	transformed int
	raw         int
}

// Map maps every source byte to the GPT-2 visible-Unicode alphabet. offset is
// the source offset represented by input[0]. rawAt maps an input boundary to a
// boundary in the original user input; it accounts for an optional synthetic
// prefix space.
func Map(input string, offset int, rawAt func(int) int) MappedPiece {
	var builder strings.Builder
	builder.Grow(len(input) * 2)
	boundaries := make([]boundary, 1, len(input)*2+1)
	boundaries[0] = boundary{transformed: 0, raw: rawAt(offset)}
	transformed := 0
	for index := range len(input) {
		r := byteToRune(input[index])
		builder.WriteRune(r)
		width := utf8.RuneLen(r)
		for byteIndex := 1; byteIndex <= width; byteIndex++ {
			transformed++
			raw := rawAt(offset + index)
			if byteIndex == width {
				raw = rawAt(offset + index + 1)
			}
			boundaries = append(boundaries, boundary{transformed: transformed, raw: raw})
		}
	}
	return MappedPiece{Text: builder.String(), boundaries: boundaries}
}

// RawOffset maps a boundary in Text to a byte offset in the original input.
// Model offsets always land on one of these boundaries.
func (p MappedPiece) RawOffset(transformed int) int {
	index := sort.Search(len(p.boundaries), func(i int) bool {
		return p.boundaries[i].transformed >= transformed
	})
	if index >= len(p.boundaries) || p.boundaries[index].transformed != transformed {
		panic("bytelevel: BPE offset is not on a byte boundary")
	}
	return p.boundaries[index].raw
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
