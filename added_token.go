package bpe

// AddedToken describes one token recognized before ByteLevel pre-tokenization
// and BPE. ID is the token ID expected by the inference model.
//
// The fields mirror Hugging Face's serialized added_tokens entries. Special is
// metadata for future decoding support; special tokens are still recognized by
// Encode. With the currently supported identity normalization, Normalized
// controls Hugging Face's matching phase but does not change Content.
type AddedToken struct {
	ID         uint32
	Content    string
	SingleWord bool
	LStrip     bool
	RStrip     bool
	Normalized bool
	Special    bool
}

// NewAddedToken constructs an AddedToken with Hugging Face's defaults.
// Regular tokens are normalized by default; special tokens are not.
func NewAddedToken(content string, id uint32, special bool) AddedToken {
	return AddedToken{
		ID:         id,
		Content:    content,
		Normalized: !special,
		Special:    special,
	}
}
