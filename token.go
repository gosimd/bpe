package bpe

// Offset identifies a half-open byte range [Start, End) in the input piece.
//
// Go strings are byte-addressable, so offsets are expressed in bytes rather
// than Unicode code points.
type Offset struct {
	Start int
	End   int
}

// Token is one BPE token produced by Model.Tokenize.
//
// Value is the vocabulary string associated with ID. It can differ from the
// corresponding source bytes when the model uses a subword prefix, an
// end-of-word suffix, or byte fallback.
type Token struct {
	ID     uint32
	Value  string
	Offset Offset
}
