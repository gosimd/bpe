package pretokenize

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/gosimd/bpe/internal/benchsuite"
	"github.com/gosimd/bpe/internal/hforacle"
)

func TestHFSuiteByteLevel(t *testing.T) {
	path := os.Getenv(benchsuite.Environment)
	if path == "" {
		t.Skip("run make test-reference (missing BPE_BENCH_SUITE)")
	}
	root := os.Getenv(hforacle.RepositoryEnvironment)
	if root == "" {
		t.Fatal("BPE_HF_REPOSITORY must name the repository root")
	}
	suite, err := benchsuite.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	suite, err = suite.Select(os.Getenv("BPE_BENCH_MODELS"), os.Getenv("BPE_BENCH_CORPORA"))
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range suite.Models {
		if spec.Pipeline != "gpt2" {
			continue
		}
		if _, err := benchsuite.ModelOptions(filepath.Join(root, spec.Tokenizer), spec.Pipeline); err != nil {
			t.Fatal(err)
		}
		for _, corpus := range suite.Corpora {
			t.Run(spec.Name+"/"+corpus.Name, func(t *testing.T) {
				if err := benchsuite.ValidateCorpus(root, corpus); err != nil {
					t.Fatal(err)
				}
				reader, err := hforacle.Open(benchsuite.ReferencePath(root, spec, corpus), corpus.CorpusPath(root))
				if err != nil {
					t.Fatal(err)
				}
				if err := reader.Metadata.ValidateTokenizer(filepath.Join(root, spec.Tokenizer)); err != nil {
					reader.Close()
					t.Fatal(err)
				}
				checkHFReference(t, reader)
			})
		}
	}
}

func checkHFReference(t *testing.T, reader *hforacle.Reader) {
	defer func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	}()
	pipeline, err := NewByteLevel(false, true)
	if err != nil {
		t.Fatal(err)
	}

	for {
		fixture, ok, err := reader.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		// AddedToken extraction precedes ByteLevel and is checked by the public
		// suite. A raw ByteLevel split cannot reproduce those extracted records.
		hasAddedToken := false
		for _, piece := range fixture.Record.PreTokenization {
			if piece.Model.Source == "added_token" {
				hasAddedToken = true
				break
			}
		}
		if hasAddedToken {
			continue
		}
		compareByteLevelRecord(t, pipeline, fixture)
	}
}

func compareByteLevelRecord(t *testing.T, pipeline *Pipeline, fixture hforacle.Case) {
	t.Helper()
	want := fixture.Record.PreTokenization
	pieceIndex := 0
	err := pipeline.Walk(fixture.Text, func(piece Piece) error {
		if pieceIndex >= len(want) {
			failByteLevel(t, fixture, pieceIndex, "extra piece", piece.Text(), "end of input")
		}
		if want[pieceIndex].Model.Source != "bpe" {
			failByteLevel(t, fixture, pieceIndex, "piece source", want[pieceIndex].Model.Source, "bpe")
		}
		if piece.Text() != want[pieceIndex].Value {
			byteIndex := firstDifferentByte(piece.Text(), want[pieceIndex].Value)
			failByteLevel(t, fixture, pieceIndex, fmt.Sprintf("transformed piece byte %d", byteIndex), piece.Text(), want[pieceIndex].Value)
		}
		start, end, err := piece.OriginalRange(0, len(piece.Text()))
		if err != nil {
			return err
		}
		gotOffsets := [2]int{start, end}
		if gotOffsets != want[pieceIndex].Offsets {
			failByteLevel(t, fixture, pieceIndex, "record-relative byte offset", gotOffsets, want[pieceIndex].Offsets)
		}
		pieceIndex++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if pieceIndex != len(want) {
		failByteLevel(t, fixture, pieceIndex, "piece count", pieceIndex, len(want))
	}
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
