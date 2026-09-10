package benchsuite

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModelFiles(t *testing.T) {
	suite, err := Load("testdata/suite.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range suite.Models {
		if err := ValidateModelFiles("../..", spec); err != nil {
			t.Fatal(err)
		}
	}
	root := t.TempDir()
	spec := Model{Name: "tiny", Tokenizer: "tokenizer.json", Vocab: "vocab.json", Merges: "merges.txt"}
	for _, file := range []string{spec.Tokenizer, spec.Vocab, spec.Merges} {
		data, err := os.ReadFile(filepath.Join("testdata/tiny", file))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, file), data, 0644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, spec.Merges), []byte("#version: 0.2\nb a\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ValidateModelFiles(root, spec); err == nil {
		t.Fatal("accepted changed merge order/pair")
	}
	if err := os.WriteFile(filepath.Join(root, spec.Vocab), []byte(`{"a":1}`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ValidateModelFiles(root, spec); err == nil {
		t.Fatal("accepted mismatched vocabulary")
	}
}

func TestModelOptions(t *testing.T) {
	data := []byte(`{"normalizer":{"type":"NFC"},"model":{"type":"BPE","vocab":{"a":399999},"merges":[],"unk_token":"<unk>","continuing_subword_prefix":"##","end_of_word_suffix":"</w>","fuse_unk":true,"byte_fallback":true,"ignore_merges":true}}`)
	options, err := parseModelOptions(data, "model-only")
	if err != nil {
		t.Fatal(err)
	}
	if options.UnknownToken != "<unk>" || options.ContinuingSubwordPrefix != "##" || options.EndOfWordSuffix != "</w>" || !options.FuseUnknown || !options.ByteFallback || !options.IgnoreMerges {
		t.Fatalf("lost BPE options: %+v", options)
	}
	if _, err := parseModelOptions(data, "gpt2"); err == nil {
		t.Fatal("accepted NFC as GPT-2 pipeline")
	}
	for _, bad := range []string{`{}`, strings.Replace(string(data), `"BPE"`, `"Unigram"`, 1), strings.Replace(string(data), `"type":"BPE"`, `"type":"BPE","dropout":0.1`, 1)} {
		if _, err := parseModelOptions([]byte(bad), "model-only"); err == nil {
			t.Fatalf("accepted unsupported model %s", bad)
		}
	}
	if _, err := ModelOptions("../../testdata/gpt2/tokenizer.json", "gpt2"); err != nil {
		t.Fatal(err)
	}
}
