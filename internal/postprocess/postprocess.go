// Package postprocess implements the supported immutable tokenizer
// post-processors. Public configuration and validation belong to package bpe.
package postprocess

// NoSequence marks a token inserted by a template rather than originating in
// input sequence A or B.
const NoSequence = -1

// Token is the complete internal representation needed by post-processors.
type Token struct {
	ID         uint32
	Value      string
	Start      int
	End        int
	TypeID     uint32
	Special    bool
	SequenceID int
}

// Processor transforms one or two already tokenized input sequences.
type Processor interface {
	Process(first, second []Token, pair, addSpecialTokens bool) []Token
	AppendIDs(dst, first, second []uint32, pair, addSpecialTokens bool) []uint32
	AffectsIDs() bool
}

// Merge applies the default PostProcessor behavior: concatenate the sequences,
// assign type IDs 0 and 1, and retain their sequence ownership.
func Merge(first, second []Token, pair bool) []Token {
	capacity := len(first)
	if pair {
		capacity += len(second)
	}
	result := make([]Token, 0, capacity)
	result = appendSequence(result, first, 0, 0)
	if pair {
		result = appendSequence(result, second, 1, 1)
	}
	return result
}

func appendSequence(dst, source []Token, typeID uint32, sequenceID int) []Token {
	for _, token := range source {
		token.TypeID = typeID
		token.SequenceID = sequenceID
		dst = append(dst, token)
	}
	return dst
}
