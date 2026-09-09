package bpe

// Encoding is the result of tokenizing one or two raw-text inputs.
//
// All slices have equal length. Every offset is a half-open byte range in the
// corresponding input Go string. Hugging Face-compatible ByteLevel alignment
// can make adjacent token offsets overlap within one multibyte UTF-8 scalar.
// SequenceIDs contains 0 for the first input, 1 for the pair input, and
// NoSequence for template-inserted special tokens.
type Encoding struct {
	IDs               []uint32
	TypeIDs           []uint32
	Tokens            []string
	Offsets           []Offset
	SpecialTokensMask []uint32
	SequenceIDs       []int
}

// NoSequence is the SequenceIDs value used for a token inserted by a
// post-processing template rather than originating in either input.
const NoSequence = -1
