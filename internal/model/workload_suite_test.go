package model_test

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/gosimd/bpe/internal/benchsuite"
	"github.com/gosimd/bpe/internal/hforacle"
	"github.com/gosimd/bpe/internal/model"
)

// TestModelWorkload is an explicit, untimed diagnostic. Normal go test skips
// it; generation and oracle loading remain outside the ordinary test workflow.
func TestModelWorkload(t *testing.T) {
	out := os.Getenv("BPE_WORKLOAD_OUTPUT")
	if out == "" {
		t.Skip("run make analyze-model-workload")
	}
	root := os.Getenv(hforacle.RepositoryEnvironment)
	if root == "" {
		t.Fatal("BPE_HF_REPOSITORY is required")
	}
	suite, err := benchsuite.Load(os.Getenv(benchsuite.Environment))
	if err != nil {
		t.Fatal(err)
	}
	suite, err = suite.Select(os.Getenv("BPE_BENCH_MODELS"), os.Getenv("BPE_BENCH_CORPORA"))
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range suite.Models {
		t.Run(spec.Name, func(t *testing.T) {
			if err := benchsuite.ValidateModelFiles(root, spec); err != nil {
				t.Fatal(err)
			}
			read := func(path string) []byte {
				data, err := os.ReadFile(filepath.Join(root, path))
				if err != nil {
					t.Fatal(err)
				}
				return data
			}
			vocab, err := model.ParseVocabulary(read(spec.Vocab))
			if err != nil {
				t.Fatal(err)
			}
			merges, err := model.ParseMerges(read(spec.Merges))
			if err != nil {
				t.Fatal(err)
			}
			options, err := benchsuite.ModelOptions(filepath.Join(root, spec.Tokenizer), spec.Pipeline)
			if err != nil {
				t.Fatal(err)
			}
			compiled, err := model.New(vocab, merges, options)
			if err != nil {
				t.Fatal(err)
			}
			for _, corpus := range suite.Corpora {
				t.Run(corpus.Name, func(t *testing.T) {
					if err := benchsuite.ValidateCorpus(root, corpus); err != nil {
						t.Fatal(err)
					}
					path := benchsuite.ReferencePath(root, spec, corpus)
					reader, err := hforacle.Open(path, corpus.CorpusPath(root))
					if err != nil {
						t.Fatal(err)
					}
					defer reader.Close()
					if err := reader.Metadata.ValidateTokenizer(filepath.Join(root, spec.Tokenizer)); err != nil {
						t.Fatal(err)
					}
					analyzer := model.NewWorkloadAnalyzer(compiled)
					skipped := 0
					for {
						fixture, ok, err := reader.Next()
						if err != nil {
							t.Fatal(err)
						}
						if !ok {
							break
						}
						for _, piece := range fixture.Record.PreTokenization {
							if piece.Model.Source == "added_token" {
								skipped++
								continue
							}
							if piece.Model.Source != "bpe" {
								t.Fatalf("unknown source %q", piece.Model.Source)
							}
							got, err := analyzer.Analyze(piece.Value)
							if err != nil {
								t.Fatal(err)
							}
							if len(got) != len(piece.Model.IDs) || len(got) != len(piece.Model.LocalOffsets) {
								t.Fatalf("record %d piece %d %q: token count differs from HF", fixture.Record.Index, piece.Index, piece.Value)
							}
							for i, token := range got {
								if token.ID != piece.Model.IDs[i] || [2]int{token.Start, token.End} != piece.Model.LocalOffsets[i] {
									t.Fatalf("record %d piece %d %q: token %d = %+v differs from HF", fixture.Record.Index, piece.Index, piece.Value, i, token)
								}
							}
						}
					}
					hashes := make(map[string]string)
					for _, name := range []string{spec.Tokenizer, spec.Vocab, spec.Merges, "internal/model/model.go", "internal/model/encode.go", "internal/model/merge.go", "internal/model/workload_test.go", "internal/model/workload_generated_test.go", "internal/model/workload_suite_test.go", "tools/workload-instrument/main.go"} {
						hashes[name] = fmt.Sprintf("%x", sha256.Sum256(read(name)))
					}
					oracle, err := os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
					hashes["oracle"] = fmt.Sprintf("%x", sha256.Sum256(oracle))
					result := struct {
						Schema   int                  `json:"schema"`
						Model    string               `json:"model"`
						Corpus   string               `json:"corpus"`
						Metadata hforacle.Metadata    `json:"oracle_metadata"`
						Hashes   map[string]string    `json:"sha256"`
						Skipped  int                  `json:"excluded_added_token_pieces"`
						Stats    model.WorkloadReport `json:"stats"`
					}{1, spec.Name, corpus.Name, reader.Metadata, hashes, skipped, analyzer.Finish()}
					data, err := json.MarshalIndent(result, "", "  ")
					if err != nil {
						t.Fatal(err)
					}
					dir := filepath.Join(out, spec.Name)
					if err := os.MkdirAll(dir, 0755); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(filepath.Join(dir, corpus.Name+".json"), append(data, '\n'), 0644); err != nil {
						t.Fatal(err)
					}
					t.Logf("%d pieces, %d merges, %d merge lookups", result.Stats.Pieces, result.Stats.Merges.Sum, result.Stats.Lookup.Queries)
				})
			}
		})
	}
}
