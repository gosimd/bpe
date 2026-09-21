package tokenizer

import (
	"errors"
	"fmt"
	"unicode/utf8"

	"github.com/gosimd/bpe/internal/addedtoken"
	"github.com/gosimd/bpe/internal/model"
)

var ErrInvalidAddedToken = errors.New("bpe: invalid added token")

// ValidateAddedTokens is shared by programmatic and JSON construction.
func ValidateAddedTokens(m *model.Model, tokens []addedtoken.Definition) error {
	contents := make(map[string]int, len(tokens))
	ids := make(map[uint32]int, len(tokens))
	for i, t := range tokens {
		if t.Content == "" || !utf8.ValidString(t.Content) {
			return fmt.Errorf("%w: token %d has empty or invalid UTF-8 content", ErrInvalidAddedToken, i)
		}
		if prev, ok := contents[t.Content]; ok {
			return fmt.Errorf("%w: tokens %d and %d duplicate content %q", ErrInvalidAddedToken, prev, i, t.Content)
		}
		if prev, ok := ids[t.ID]; ok {
			return fmt.Errorf("%w: tokens %d and %d duplicate ID %d", ErrInvalidAddedToken, prev, i, t.ID)
		}
		if id, ok := m.TokenToID(t.Content); ok && id != t.ID {
			return fmt.Errorf("%w: content %q has model ID %d, not %d", ErrInvalidAddedToken, t.Content, id, t.ID)
		}
		if content, ok := m.IDToToken(t.ID); ok && content != t.Content {
			return fmt.Errorf("%w: ID %d belongs to model token %q, not %q", ErrInvalidAddedToken, t.ID, content, t.Content)
		}
		contents[t.Content] = i
		ids[t.ID] = i
	}
	return nil
}
