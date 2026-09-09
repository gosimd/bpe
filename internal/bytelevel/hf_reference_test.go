package bytelevel

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/gosimd/bpe/internal/hforacle"
)

func TestHFGPT2WikiText2ByteLevel(t *testing.T) {
	reader := openHFReference(t)
	defer func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	}()

	for {
		fixture, ok, err := reader.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		compareByteLevelRecord(t, fixture)
	}
}

func compareByteLevelRecord(t *testing.T, fixture hforacle.Case) {
	t.Helper()
	want := fixture.Record.PreTokenization
	got := Split(fixture.Text)
	common := min(len(got), len(want))
	for pieceIndex := range common {
		if want[pieceIndex].Model.Source != "bpe" {
			failByteLevel(t, fixture, pieceIndex, "piece source", want[pieceIndex].Model.Source, "bpe")
		}
		piece := got[pieceIndex]
		mapped := Map(fixture.Text[piece.Start:piece.End], piece.Start, func(offset int) int { return offset })
		if mapped.Text != want[pieceIndex].Value {
			byteIndex := firstDifferentByte(mapped.Text, want[pieceIndex].Value)
			failByteLevel(t, fixture, pieceIndex, fmt.Sprintf("transformed piece byte %d", byteIndex), mapped.Text, want[pieceIndex].Value)
		}
		gotOffsets := [2]int{piece.Start, piece.End}
		if gotOffsets != want[pieceIndex].Offsets {
			failByteLevel(t, fixture, pieceIndex, "record-relative byte offset", gotOffsets, want[pieceIndex].Offsets)
		}
	}
	if len(got) != len(want) {
		failByteLevel(t, fixture, common, "piece count", len(got), len(want))
	}
}

func openHFReference(t *testing.T) *hforacle.Reader {
	t.Helper()
	referencePath := os.Getenv(hforacle.ReferenceEnvironment)
	if referencePath == "" {
		t.Skipf("full Hugging Face corpus oracle is opt-in; run make test-reference (missing %s)", hforacle.ReferenceEnvironment)
	}
	repository := os.Getenv(hforacle.RepositoryEnvironment)
	if repository == "" {
		t.Fatalf("%s must name the repository root", hforacle.RepositoryEnvironment)
	}
	reader, err := hforacle.Open(referencePath, filepath.Join(repository, "testdata", "wikitext2", "validation.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if err := reader.Metadata.ValidateTokenizer(filepath.Join(repository, "testdata", "gpt2", "tokenizer.json")); err != nil {
		reader.Close()
		t.Fatal(err)
	}
	return reader
}

func failByteLevel(t *testing.T, fixture hforacle.Case, pieceIndex int, field string, got, want any) {
	t.Helper()
	t.Fatalf("Hugging Face ByteLevel mismatch\nrecord index: %d\noriginal text: %q\npiece index: %d\nfirst differing %s:\n  got:  %v\n  want: %v",
		fixture.Record.Index, fixture.Text, pieceIndex, field, got, want)
}

func firstDifferentByte(got, want string) int {
	common := min(len(got), len(want))
	for index := range common {
		if got[index] != want[index] {
			return index
		}
	}
	return common
}
