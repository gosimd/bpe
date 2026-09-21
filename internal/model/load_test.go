package model

import (
	"errors"
	"strings"
	"testing"
)

func TestParseVocabularyRejectsMalformedJSON(t *testing.T) {
	for _, data := range []string{
		`{"a":0,"a":1}`, `{"a":0,"\u0061":1}`, `{"\ud800":0}`,
		`{"\udc00":0}`, "{\"\xff\":0}", `{"a":0} null`,
		`null`, `[]`, `{"a":null}`, `{"a":-1}`, `{"a":4294967296}`,
		strings.Repeat("[", 81) + "0" + strings.Repeat("]", 81),
	} {
		vocab, err := ParseVocabulary([]byte(data))
		if vocab != nil || !errors.Is(err, ErrInvalidVocabulary) {
			t.Errorf("ParseVocabulary(%q) = %v, %v", data, vocab, err)
		}
	}
}

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
