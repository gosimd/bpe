package bpe_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/gosimd/bpe"
	"github.com/gosimd/bpe/internal/hforacle"
)

type hfReferenceConfiguration struct {
	referencePath string
	repository    string
}

func TestHFGPT2WikiText2(t *testing.T) {
	configuration := configuredHFReference(t)
	model, err := bpe.OpenModel(
		filepath.Join(configuration.repository, "testdata", "gpt2", "vocab.json"),
		filepath.Join(configuration.repository, "testdata", "gpt2", "merges.txt"),
		bpe.ModelOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	endOfText := bpe.NewAddedToken("<|endoftext|>", 50256, true)
	// This is the exact flag serialized in the pinned GPT-2 tokenizer.json.
	endOfText.Normalized = true
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddPrefixSpace: false,
		AddedTokens:    []bpe.AddedToken{endOfText},
		PostProcessor: bpe.ByteLevelPostProcessor{
			AddPrefixSpace: true,
			TrimOffsets:    false,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("Model", func(t *testing.T) {
		walkHFReference(t, configuration, func(fixture hforacle.Case) {
			for pieceIndex := range fixture.Record.PreTokenization {
				compareModelPiece(t, model, fixture, pieceIndex)
			}
		})
	})

	t.Run("Tokenizer", func(t *testing.T) {
		walkHFReference(t, configuration, func(fixture hforacle.Case) {
			got, err := tokenizer.EncodeWithOptions(fixture.Text, bpe.EncodeOptions{AddSpecialTokens: false})
			if err != nil {
				failHF(t, "Tokenizer", fixture, -1, -1, "error", err, nil)
			}
			compareEncoding(t, model, fixture, got)
		})
	})

	t.Run("AppendIDs", func(t *testing.T) {
		var flattened []uint32
		walkHFReference(t, configuration, func(fixture hforacle.Case) {
			wantRange := fixture.Record.OutputTokenRange
			if len(flattened) != wantRange[0] {
				failHF(t, "AppendIDs", fixture, -1, -1, "record start boundary", len(flattened), wantRange[0])
			}
			start := len(flattened)
			var err error
			flattened, err = tokenizer.AppendIDsWithOptions(flattened, fixture.Text, bpe.EncodeOptions{AddSpecialTokens: false})
			if err != nil {
				failHF(t, "AppendIDs", fixture, -1, -1, "error", err, nil)
			}
			if len(flattened) != wantRange[1] {
				failHF(t, "AppendIDs", fixture, -1, -1, "record end boundary", len(flattened), wantRange[1])
			}
			compareAppendedIDs(t, model, fixture, flattened[start:])
		})
	})
}

func configuredHFReference(t *testing.T) hfReferenceConfiguration {
	t.Helper()
	referencePath := os.Getenv(hforacle.ReferenceEnvironment)
	if referencePath == "" {
		t.Skipf("full Hugging Face corpus oracle is opt-in; run make test-reference (missing %s)", hforacle.ReferenceEnvironment)
	}
	repository := os.Getenv(hforacle.RepositoryEnvironment)
	if repository == "" {
		t.Fatalf("%s must name the repository root", hforacle.RepositoryEnvironment)
	}
	return hfReferenceConfiguration{referencePath: referencePath, repository: repository}
}

func walkHFReference(t *testing.T, configuration hfReferenceConfiguration, visit func(hforacle.Case)) {
	t.Helper()
	reader, err := hforacle.Open(
		configuration.referencePath,
		filepath.Join(configuration.repository, "testdata", "wikitext2", "validation.jsonl"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	}()
	if err := reader.Metadata.ValidateTokenizer(filepath.Join(configuration.repository, "testdata", "gpt2", "tokenizer.json")); err != nil {
		t.Fatal(err)
	}

	for {
		fixture, ok, err := reader.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			return
		}
		visit(fixture)
	}
}

func compareModelPiece(t *testing.T, model *bpe.Model, fixture hforacle.Case, pieceIndex int) {
	t.Helper()
	piece := fixture.Record.PreTokenization[pieceIndex]
	if piece.Index != pieceIndex {
		failHF(t, "Model", fixture, pieceIndex, -1, "piece index", piece.Index, pieceIndex)
	}
	if piece.Model.Source == "added_token" {
		return
	}
	if piece.Model.Source != "bpe" {
		failHF(t, "Model", fixture, pieceIndex, -1, "piece source", piece.Model.Source, "bpe or added_token")
	}
	got, err := model.Tokenize(piece.Value)
	if err != nil {
		failHF(t, "Model", fixture, pieceIndex, -1, "error", err, nil)
	}
	want := piece.Model
	common := min(len(got), len(want.IDs))
	for tokenIndex := range common {
		wantToken := vocabularyToken(t, model, want.IDs[tokenIndex])
		if got[tokenIndex].ID != want.IDs[tokenIndex] || got[tokenIndex].Value != wantToken {
			failHF(t, "Model", fixture, pieceIndex, tokenIndex, "token/ID",
				fmt.Sprintf("%q/%d", got[tokenIndex].Value, got[tokenIndex].ID),
				fmt.Sprintf("%q/%d", wantToken, want.IDs[tokenIndex]))
		}
		gotOffset := [2]int{got[tokenIndex].Offset.Start, got[tokenIndex].Offset.End}
		if tokenIndex >= len(want.LocalOffsets) {
			failHF(t, "Model", fixture, pieceIndex, tokenIndex, "local offset count", len(got), len(want.LocalOffsets))
		}
		if gotOffset != want.LocalOffsets[tokenIndex] {
			failHF(t, "Model", fixture, pieceIndex, tokenIndex, "local byte offset", gotOffset, want.LocalOffsets[tokenIndex])
		}
	}
	if len(got) != len(want.IDs) {
		failHF(t, "Model", fixture, pieceIndex, common, "token count", len(got), len(want.IDs))
	}
	if len(want.LocalOffsets) != len(want.IDs) {
		failHF(t, "Model oracle", fixture, pieceIndex, common, "ID/local-offset count", len(want.IDs), len(want.LocalOffsets))
	}
}

func compareEncoding(t *testing.T, model *bpe.Model, fixture hforacle.Case, got bpe.Encoding) {
	t.Helper()
	want := fixture.Record.Encoding
	wantLength := len(want.IDs)
	lengths := []struct {
		name string
		got  int
		want int
	}{
		{"IDs", len(got.IDs), wantLength},
		{"tokens", len(got.Tokens), wantLength},
		{"offsets", len(got.Offsets), len(want.Offsets)},
		{"type IDs", len(got.TypeIDs), len(want.TypeIDs)},
		{"sequence IDs", len(got.SequenceIDs), len(want.SequenceIDs)},
		{"special-token mask", len(got.SpecialTokensMask), len(want.SpecialTokensMask)},
	}
	for _, length := range lengths {
		if length.got != length.want {
			tokenIndex := min(length.got, length.want)
			failHF(t, "Tokenizer", fixture, pieceIndexForToken(fixture.Record, tokenIndex), tokenIndex, length.name+" count", length.got, length.want)
		}
	}

	for tokenIndex := range wantLength {
		pieceIndex := pieceIndexForToken(fixture.Record, tokenIndex)
		wantToken := vocabularyToken(t, model, want.IDs[tokenIndex])
		if got.IDs[tokenIndex] != want.IDs[tokenIndex] || got.Tokens[tokenIndex] != wantToken {
			failHF(t, "Tokenizer", fixture, pieceIndex, tokenIndex, "token/ID",
				fmt.Sprintf("%q/%d", got.Tokens[tokenIndex], got.IDs[tokenIndex]),
				fmt.Sprintf("%q/%d", wantToken, want.IDs[tokenIndex]))
		}
		gotOffset := [2]int{got.Offsets[tokenIndex].Start, got.Offsets[tokenIndex].End}
		if gotOffset != want.Offsets[tokenIndex] {
			failHF(t, "Tokenizer", fixture, pieceIndex, tokenIndex, "record-relative byte offset", gotOffset, want.Offsets[tokenIndex])
		}
		if got.TypeIDs[tokenIndex] != want.TypeIDs[tokenIndex] {
			failHF(t, "Tokenizer", fixture, pieceIndex, tokenIndex, "type ID", got.TypeIDs[tokenIndex], want.TypeIDs[tokenIndex])
		}
		wantSequenceID := bpe.NoSequence
		if want.SequenceIDs[tokenIndex] != nil {
			wantSequenceID = *want.SequenceIDs[tokenIndex]
		}
		if got.SequenceIDs[tokenIndex] != wantSequenceID {
			failHF(t, "Tokenizer", fixture, pieceIndex, tokenIndex, "sequence ID", got.SequenceIDs[tokenIndex], wantSequenceID)
		}
		if got.SpecialTokensMask[tokenIndex] != want.SpecialTokensMask[tokenIndex] {
			failHF(t, "Tokenizer", fixture, pieceIndex, tokenIndex, "special-token mask", got.SpecialTokensMask[tokenIndex], want.SpecialTokensMask[tokenIndex])
		}
	}
}

func compareAppendedIDs(t *testing.T, model *bpe.Model, fixture hforacle.Case, got []uint32) {
	t.Helper()
	want := fixture.Record.Encoding.IDs
	common := min(len(got), len(want))
	for tokenIndex := range common {
		if got[tokenIndex] != want[tokenIndex] {
			pieceIndex := pieceIndexForToken(fixture.Record, tokenIndex)
			failHF(t, "AppendIDs", fixture, pieceIndex, tokenIndex, "token/ID",
				fmt.Sprintf("%q/%d", vocabularyToken(t, model, got[tokenIndex]), got[tokenIndex]),
				fmt.Sprintf("%q/%d", vocabularyToken(t, model, want[tokenIndex]), want[tokenIndex]))
		}
	}
	if len(got) != len(want) {
		failHF(t, "AppendIDs", fixture, pieceIndexForToken(fixture.Record, common), common, "record token count", len(got), len(want))
	}
}

func pieceIndexForToken(record hforacle.Record, tokenIndex int) int {
	position := 0
	for _, piece := range record.PreTokenization {
		end := position + len(piece.Model.IDs)
		if tokenIndex >= position && tokenIndex < end {
			return piece.Index
		}
		position = end
	}
	return -1
}

func vocabularyToken(t *testing.T, model *bpe.Model, id uint32) string {
	t.Helper()
	token, ok := model.IDToToken(id)
	if !ok {
		t.Fatalf("oracle references vocabulary ID %d which is absent from the pinned model", id)
	}
	return token
}

func failHF(t *testing.T, stage string, fixture hforacle.Case, pieceIndex, tokenIndex int, field string, got, want any) {
	t.Helper()
	t.Fatalf("Hugging Face %s mismatch\nrecord index: %d\noriginal text: %q\npiece index: %d\ntoken index: %d\nfirst differing %s:\n  got:  %v\n  want: %v",
		stage, fixture.Record.Index, fixture.Text, pieceIndex, tokenIndex, field, got, want)
}
