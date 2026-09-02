package model

import (
	"errors"
	"testing"
)

func TestParseMergesRejectsWhitespaceAmbiguity(t *testing.T) {
	_, err := ParseMerges([]byte("a  b\n"))
	if !errors.Is(err, ErrInvalidMerge) {
		t.Fatalf("ParseMerges() error = %v, want ErrInvalidMerge", err)
	}
}

func FuzzParseModelData(f *testing.F) {
	f.Add([]byte(`{"a":0,"b":1,"ab":2}`), []byte("#version: 0.2\na b\n"))
	f.Add([]byte(`{}`), []byte{})
	f.Add([]byte(`[]`), []byte("a b\n"))

	f.Fuzz(func(t *testing.T, vocabJSON, mergesText []byte) {
		vocab, err := ParseVocabulary(vocabJSON)
		if err != nil {
			return
		}
		merges, err := ParseMerges(mergesText)
		if err != nil {
			return
		}
		_, _ = New(vocab, merges, Options{})
	})
}
