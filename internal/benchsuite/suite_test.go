package benchsuite

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCatalogueAndSelection(t *testing.T) {
	suite, err := Load("../../benchmarks/suite.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(suite.Models) != 1 || len(suite.Corpora) != 5 {
		t.Fatalf("unexpected catalogue size: %d/%d", len(suite.Models), len(suite.Corpora))
	}
	// A second model with different assets/pipeline must remain independent.
	suite.Models = append(suite.Models, Model{Name: "other", Tokenizer: "other/tokenizer.json", Vocab: "other/vocab.json", Merges: "other/merges.txt", Pipeline: "model-only"})
	selected, err := suite.Select("other,gpt2", "zh-4k,en-4k")
	if err != nil {
		t.Fatal(err)
	}
	if len(selected.Models) != 2 || selected.Models[1].Name != "other" || len(selected.Corpora) != 2 || selected.Corpora[0].Name != "en-4k" {
		t.Fatalf("selection must preserve catalogue order: %+v", selected)
	}
	if _, err := suite.Select("typo", ""); err == nil {
		t.Fatal("accepted unknown model")
	}
	if _, err := suite.Select("", "typo"); err == nil {
		t.Fatal("accepted unknown corpus")
	}
	if ReferencePath("root", selected.Models[0], selected.Corpora[0]) == ReferencePath("root", selected.Models[1], selected.Corpora[0]) {
		t.Fatal("model oracle paths collide")
	}
}

func TestTwoModelCatalogue(t *testing.T) {
	suite, err := Load("testdata/suite.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(suite.Models) != 2 || suite.Models[0].Pipeline != "gpt2" || suite.Models[1].Pipeline != "model-only" {
		t.Fatalf("unexpected two-model smoke suite: %+v", suite)
	}
}

func TestInvalidCatalogue(t *testing.T) {
	valid := `{"schema":1,"models":[{"name":"tiny","tokenizer":"t.json","vocab":"v.json","merges":"m.txt","pipeline":"model-only"}],"corpora":[{"name":"text","path":"c.jsonl"}]}`
	for _, tc := range []struct{ name, text string }{
		{"unknown field", strings.Replace(valid, `"schema":1`, `"schema":1,"typo":true`, 1)},
		{"version", strings.Replace(valid, `"schema":1`, `"schema":2`, 1)},
		{"pipeline", strings.Replace(valid, `"model-only"`, `"qwen"`, 1)},
		{"path escape", strings.Replace(valid, `"t.json"`, `"../t.json"`, 1)},
		{"name escape", strings.Replace(valid, `"tiny"`, `"../tiny"`, 1)},
		{"ambiguous corpus", strings.Replace(valid, `"path":"c.jsonl"`, `"path":"c.jsonl","pool":"ru-wikipedia","request_bytes":4096`, 1)},
		{"trailing", valid + `{}`},
		{"empty", `{"schema":1,"models":[],"corpora":[]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "suite.json")
			if err := os.WriteFile(path, []byte(tc.text), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path); err == nil {
				t.Fatal("accepted invalid catalogue")
			}
		})
	}
}

func TestVendoredPools(t *testing.T) {
	for _, pool := range []string{"en-wikitext2", "ru-wikipedia", "ar-wikipedia", "ja-wikipedia", "zh-wikipedia"} {
		t.Run(pool, func(t *testing.T) {
			var whole string
			for _, size := range []int{1048576, 256, 4096, 65536} {
				records, err := PoolRecords("../..", Corpus{Pool: pool, RequestBytes: size})
				if err != nil {
					t.Fatal(err)
				}
				joined := strings.Join(records, "")
				if whole == "" {
					whole = joined
				} else if joined != whole {
					t.Fatalf("request size %d changed pool bytes", size)
				}
			}
		})
	}
	if _, err := PoolRecords("../..", Corpus{Pool: "ru-wikipedia", RequestBytes: 17}); err == nil {
		t.Fatal("accepted unpinned request size")
	}
	if _, err := PoolRecords("../..", Corpus{Pool: "missing", RequestBytes: 4096}); err == nil {
		t.Fatal("accepted missing pool")
	}
}

func TestCheckedFile(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "text"), []byte("original"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := checkedFile(root, "text", digest([]byte("changed"))); err == nil {
		t.Fatal("accepted changed file")
	}
	if _, err := checkedFile(root, "../text", ""); err == nil {
		t.Fatal("accepted path outside pool")
	}
}

func TestPrepareCorpusDeterministic(t *testing.T) {
	// Copy the small dependency set into a temporary repository; generation must
	// not modify the committed source or require the sibling checkout.
	root := t.TempDir()
	dir := filepath.Join(root, "testdata/corpora/text")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"MANIFEST.json", "en-wikitext2.txt.gz", "en-wikitext2.source.json", "en-wikitext2-4096.json"} {
		data, err := os.ReadFile(filepath.Join("../../testdata/corpora/text", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0644); err != nil {
			t.Fatal(err)
		}
	}
	c := Corpus{Name: "en-4k", Pool: "en-wikitext2", RequestBytes: 4096}
	if err := PrepareCorpus(root, c); err != nil {
		t.Fatal(err)
	}
	first, err := os.ReadFile(c.CorpusPath(root))
	if err != nil {
		t.Fatal(err)
	}
	if err := PrepareCorpus(root, c); err != nil {
		t.Fatal(err)
	}
	second, err := os.ReadFile(c.CorpusPath(root))
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != string(second) {
		t.Fatal("nondeterministic JSONL")
	}
	if err := ValidateCorpus(root, c); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(c.CorpusPath(root), []byte("stale"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ValidateCorpus(root, c); err == nil {
		t.Fatal("accepted stale derived JSONL")
	}
	want, err := PoolRecords(root, c)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSuffix(string(first), "\n"), "\n")
	if len(lines) != len(want) {
		t.Fatal("record count changed")
	}
	for i, line := range lines {
		var record struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatal(err)
		}
		if record.Text != want[i] {
			t.Fatalf("changed record %d", i)
		}
	}
	// Empty records in a caller-provided JSONL must survive untouched.
	path := filepath.Join(root, "raw.jsonl")
	raw := []byte("{\"text\":\"\"}\n{\"text\":\" x \\n\"}\n")
	if err := os.WriteFile(path, raw, 0644); err != nil {
		t.Fatal(err)
	}
	if err := PrepareCorpus(root, Corpus{Name: "raw", Path: "raw.jsonl"}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(raw) {
		t.Fatal("changed original JSONL")
	}
}
