package bpe

// Encoding is the result of tokenizing one raw-text input.
//
// IDs and Offsets have equal length. Every offset is a half-open byte range in
// the original Go string passed to Tokenizer.Encode.
type Encoding struct {
	IDs     []uint32
	Offsets []Offset
}
