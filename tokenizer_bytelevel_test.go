package bpe_test

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/gosimd/bpe"
)

// Use the pinned model alphabet instead of reimplementing ByteLevel in tests.
func testByteAlphabet(t testing.TB) map[string]uint32 {
	t.Helper()
	data, err := os.ReadFile("testdata/gpt2/vocab.json")
	if err != nil {
		t.Fatal(err)
	}
	var full map[string]uint32
	if err := json.Unmarshal(data, &full); err != nil {
		t.Fatal(err)
	}
	vocab := make(map[string]uint32, 256)
	for value, id := range full {
		if id < 256 {
			vocab[value] = id
		}
	}
	if len(vocab) != 256 {
		t.Fatal("unexpected GPT-2 byte alphabet")
	}
	return vocab
}

func TestTokenizerByteLevelPrefixHFFixtures(t *testing.T) {
	data, err := os.ReadFile("testdata/hf_bytelevel_prefix_v0.23.2.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Version                            string `json:"tokenizers_version"`
		Offsets                            string
		Normalization, Padding, Truncation bool
		Source                             struct{ Path, SHA256 string }
		Cases                              []struct {
			Text    string
			Prefix  bool `json:"add_prefix_space"`
			Trim    bool `json:"trim_offsets"`
			IDs     []uint32
			Tokens  []string
			Offsets [][2]int
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Version != "0.23.2" || fixture.Offsets != "byte" || fixture.Normalization || fixture.Padding || fixture.Truncation || len(fixture.Cases) != 28 {
		t.Fatal("unexpected fixture metadata")
	}
	source, err := os.ReadFile(fixture.Source.Path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(source)) != fixture.Source.SHA256 {
		t.Fatal("source hash mismatch")
	}
	model := mustModel(t, testByteAlphabet(t), nil, bpe.ModelOptions{})
	for i, c := range fixture.Cases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			options := bpe.TokenizerOptions{AddPrefixSpace: c.Prefix, AddedTokens: []bpe.AddedToken{bpe.NewAddedToken("<tag>", 256, true)}}
			if c.Trim {
				options.PostProcessor = bpe.ByteLevelPostProcessor{AddPrefixSpace: c.Prefix, TrimOffsets: true}
			}
			tokenizer, err := bpe.NewTokenizerWithOptions(model, options)
			if err != nil {
				t.Fatal(err)
			}
			got, err := tokenizer.Encode(c.Text)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.IDs, c.IDs) || !slices.Equal(got.Tokens, c.Tokens) || len(got.Offsets) != len(c.Offsets) {
				t.Fatalf("%q: encoding %+v, want %+v", c.Text, got, c)
			}
			for j, span := range got.Offsets {
				if [2]int{span.Start, span.End} != c.Offsets[j] {
					t.Fatalf("%q: token %d offset %v, want %v", c.Text, j, span, c.Offsets[j])
				}
			}
			ids, err := tokenizer.AppendIDs([]uint32{999}, c.Text)
			if err != nil || ids[0] != 999 || !slices.Equal(ids[1:], c.IDs) {
				t.Fatalf("AppendIDs: %v, %v", ids, err)
			}
		})
	}
}
