package bpe_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/gosimd/bpe"
)

func TestModelAndTokenizerRejectMalformedVocabularyJSON(t *testing.T) {
	for name, vocab := range map[string]string{
		"duplicate":               `{"a":0,"a":1}`,
		"duplicate-escaped":       `{"a":0,"\u0061":1}`,
		"duplicate-supplementary": `{"😀":0,"\ud83d\ude00":1}`,
		"invalid-UTF8":            "{\"\xff\":0}",
		"truncated-UTF8":          "{\"\xe2\x82\":0}",
		"high-surrogate":          `{"\ud800":0}`,
		"low-surrogate":           `{"\udc00":0}`,
		"reversed-surrogates":     `{"\udc00\ud800":0}`,
		"two-high-surrogates":     `{"\ud800\ud800":0}`,
		"high-before-scalar":      `{"\ud800\u0061":0}`,
		"truncated-escape":        `{"\u123":0}`,
		"invalid-escape":          `{"\uZZZZ":0}`,
		"trailing-value":          `{"a":0} null`,
		"trailing-comma":          `{"a":0,}`,
	} {
		t.Run(name, func(t *testing.T) {
			model, err := bpe.ParseModel([]byte(vocab), nil, bpe.ModelOptions{})
			if model != nil || !errors.Is(err, bpe.ErrInvalidVocabulary) {
				t.Errorf("ParseModel = %v, %v; want nil, ErrInvalidVocabulary", model, err)
			}
			config := `{"model":{"type":"BPE","vocab":` + vocab + `,"merges":[]}}`
			tokenizer, err := bpe.ParseTokenizer([]byte(config))
			if tokenizer != nil || !errors.Is(err, bpe.ErrInvalidTokenizer) {
				t.Errorf("ParseTokenizer = %v, %v; want nil, ErrInvalidTokenizer", tokenizer, err)
			}

			dir := t.TempDir()
			vocabPath, mergesPath := filepath.Join(dir, "vocab.json"), filepath.Join(dir, "merges.txt")
			if err := os.WriteFile(vocabPath, []byte(vocab), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(mergesPath, nil, 0o600); err != nil {
				t.Fatal(err)
			}
			model, err = bpe.OpenModel(vocabPath, mergesPath, bpe.ModelOptions{})
			if model != nil || !errors.Is(err, bpe.ErrInvalidVocabulary) {
				t.Errorf("OpenModel = %v, %v; want nil, ErrInvalidVocabulary", model, err)
			}
		})
	}
}

func TestModelAndTokenizerPreserveVocabularyJSON(t *testing.T) {
	for name, vocab := range map[string]string{
		"literal": `{"😀":0,"�":1,"猫":4294967295,"\\ud800":3}`,
		"escaped": `{"\uD83D\uDE00":0,"\ufffd":1,"\u732b":4294967295,"\\ud800":3}`,
	} {
		t.Run(name, func(t *testing.T) {
			model, err := bpe.ParseModel([]byte(vocab), nil, bpe.ModelOptions{})
			if err != nil {
				t.Fatal(err)
			}
			config := `{"model":{"type":"BPE","vocab":` + vocab + `,"merges":[]}}`
			tokenizer, err := bpe.ParseTokenizer([]byte(config))
			if err != nil {
				t.Fatal(err)
			}
			for token, id := range map[string]uint32{"😀": 0, "�": 1, "猫": 4294967295, `\ud800`: 3} {
				if got, ok := model.TokenToID(token); !ok || got != id {
					t.Fatalf("TokenToID(%q) = %d, %v; want %d", token, got, ok, id)
				}
				if got, ok := model.IDToToken(id); !ok || got != token {
					t.Fatalf("IDToToken(%d) = %q, %v; want %q", id, got, ok, token)
				}
				if token == `\ud800` {
					continue // Literal escape text is a vocab entry, not an initial scalar.
				}
				got, err := tokenizer.Encode(token)
				if err != nil || len(got.IDs) != 1 || got.IDs[0] != id || got.Tokens[0] != token || got.Offsets[0] != (bpe.Offset{Start: 0, End: len(token)}) {
					t.Fatalf("Encode(%q) = %+v, %v", token, got, err)
				}
			}
		})
	}
}
