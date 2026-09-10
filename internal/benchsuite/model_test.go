package benchsuite

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPinnedLargeModelAssets(t *testing.T) {
	suite, err := Load("../../benchmarks/suite.json")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][2]int{
		"qwen2-5": {151643, 151387},
		"qwen3-8": {248044, 247587},
		"bloom":   {250680, 250434},
	}
	for _, spec := range suite.Models {
		counts, ok := want[spec.Name]
		if !ok {
			continue
		}
		delete(want, spec.Name)
		t.Run(spec.Name, func(t *testing.T) {
			if spec.Pipeline != "model-only" {
				t.Fatal("full Go pipeline is not implemented")
			}
			path := filepath.Join("../..", spec.Tokenizer)
			dir := filepath.Dir(path)
			sums, err := os.ReadFile(filepath.Join(dir, "SHA256SUMS"))
			if err != nil {
				t.Fatal(err)
			}
			lines := strings.Split(strings.TrimSpace(string(sums)), "\n")
			if len(lines) != 4 {
				t.Fatal("expected hashes for tokenizer, vocab, merges and license")
			}
			for _, line := range lines {
				fields := strings.Fields(line)
				if len(fields) != 2 {
					t.Fatalf("malformed checksum line %q", line)
				}
				if _, err := checkedFile(dir, fields[1], fields[0]); err != nil {
					t.Fatal(err)
				}
			}
			if err := ValidateModelFiles("../..", spec); err != nil {
				t.Fatal(err)
			}
			if _, err := ModelOptions(path, spec.Pipeline); err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var tokenizer struct {
				Model struct {
					Vocab  map[string]uint32 `json:"vocab"`
					Merges []json.RawMessage `json:"merges"`
				} `json:"model"`
			}
			if err := json.Unmarshal(data, &tokenizer); err != nil {
				t.Fatal(err)
			}
			if len(tokenizer.Model.Vocab) != counts[0] || len(tokenizer.Model.Merges) != counts[1] {
				t.Fatalf("unexpected vocab/merges counts: %d/%d", len(tokenizer.Model.Vocab), len(tokenizer.Model.Merges))
			}
		})
	}
	if len(want) != 0 {
		t.Fatalf("pinned models absent from catalogue: %v", want)
	}
}

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
