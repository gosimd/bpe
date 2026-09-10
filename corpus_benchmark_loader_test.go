package bpe_test

import (
	"path/filepath"
	"testing"

	"github.com/gosimd/bpe/internal/benchsuite"
)

func TestBenchmarkModelOnlyLoader(t *testing.T) {
	root, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	const base = "internal/benchsuite/testdata/tiny/"
	data := loadCorpusBenchmarkData(t, root, benchsuite.Model{
		Name: "tiny", Tokenizer: base + "tokenizer.json", Vocab: base + "vocab.json", Merges: base + "merges.txt", Pipeline: "model-only",
	}, filepath.Join(root, base, "corpus.jsonl"), filepath.Join(root, base, "oracle.jsonl"))
	if data.tokenizer != nil || data.addedTokens != nil {
		t.Fatal("model-only case constructed a GPT-2 pipeline")
	}
	if len(data.records) != 3 || data.records[1].text != "" {
		t.Fatal("lost empty record")
	}
	if data.modelPieces[0].value != "ab" || data.modelPieces[0].ids[0] != 399999 {
		t.Fatal("did not load the model's own normalized HF pieces and high IDs")
	}
	data.verify(t)
}
