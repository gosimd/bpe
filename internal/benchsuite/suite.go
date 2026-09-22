// Package benchsuite is repository-only benchmark configuration and corpus tooling.
// It does not implement the public tokenizer.json loading API.
package benchsuite

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf8"
)

const Environment = "BPE_BENCH_SUITE"

type Model struct {
	Name      string `json:"name"`
	Tokenizer string `json:"tokenizer"`
	Vocab     string `json:"vocab"`
	Merges    string `json:"merges"`
	// Pipeline is model-only, the GPT-2 preset, or an explicit JSON pipeline.
	Pipeline string `json:"pipeline"`
}

type Corpus struct {
	Name         string `json:"name"`
	Path         string `json:"path,omitempty"`
	Pool         string `json:"pool,omitempty"`
	RequestBytes int    `json:"request_bytes,omitempty"`
}

type Suite struct {
	Schema  int      `json:"schema"`
	Models  []Model  `json:"models"`
	Corpora []Corpus `json:"corpora"`
}

var namePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// Load reads a catalogue. All asset paths are relative to the repository root,
// independent of the catalogue's own location.
func Load(path string) (Suite, error) {
	var suite Suite
	data, err := os.ReadFile(path)
	if err != nil {
		return suite, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&suite); err != nil {
		return suite, err
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return suite, fmt.Errorf("trailing suite JSON")
	}
	if suite.Schema != 1 || len(suite.Models) == 0 || len(suite.Corpora) == 0 {
		return suite, fmt.Errorf("suite requires schema 1 and nonempty models and corpora")
	}
	seen := map[string]bool{}
	for _, m := range suite.Models {
		if !namePattern.MatchString(m.Name) || seen[m.Name] {
			return suite, fmt.Errorf("invalid or duplicate model name %q", m.Name)
		}
		seen[m.Name] = true
		if m.Pipeline != "model-only" && m.Pipeline != "gpt2" && m.Pipeline != "json" {
			return suite, fmt.Errorf("model %s: unsupported pipeline %q", m.Name, m.Pipeline)
		}
		for _, path := range []string{m.Tokenizer, m.Vocab, m.Merges} {
			if !filepath.IsLocal(path) {
				return suite, fmt.Errorf("model %s: expected repository-relative path, got %q", m.Name, path)
			}
		}
	}
	clear(seen)
	for _, c := range suite.Corpora {
		if !namePattern.MatchString(c.Name) || seen[c.Name] {
			return suite, fmt.Errorf("invalid or duplicate corpus name %q", c.Name)
		}
		seen[c.Name] = true
		if c.Path != "" {
			if !filepath.IsLocal(c.Path) || c.Pool != "" || c.RequestBytes != 0 {
				return suite, fmt.Errorf("corpus %s: path must be local and cannot be combined with pool", c.Name)
			}
		} else if !namePattern.MatchString(c.Pool) || c.RequestBytes <= 0 {
			return suite, fmt.Errorf("corpus %s: specify path or pool and request_bytes", c.Name)
		}
	}
	return suite, nil
}

// Select preserves catalogue order and fails on typos instead of silently
// running an empty or different benchmark. Empty selectors mean all entries.
func (s Suite) Select(models, corpora string) (Suite, error) {
	selectNames := func(selector string, names []string) (map[string]bool, error) {
		result := map[string]bool{}
		if selector == "" {
			for _, n := range names {
				result[n] = true
			}
			return result, nil
		}
		for _, n := range strings.Split(selector, ",") {
			found := false
			for _, known := range names {
				if known == n {
					found = true
					break
				}
			}
			if !found {
				return nil, fmt.Errorf("unknown selection %q", n)
			}
			result[n] = true
		}
		return result, nil
	}
	modelNames, corpusNames := []string{}, []string{}
	for _, m := range s.Models {
		modelNames = append(modelNames, m.Name)
	}
	for _, c := range s.Corpora {
		corpusNames = append(corpusNames, c.Name)
	}
	ms, err := selectNames(models, modelNames)
	if err != nil {
		return Suite{}, err
	}
	cs, err := selectNames(corpora, corpusNames)
	if err != nil {
		return Suite{}, err
	}
	selected := Suite{Schema: s.Schema}
	for _, m := range s.Models {
		if ms[m.Name] {
			selected.Models = append(selected.Models, m)
		}
	}
	for _, c := range s.Corpora {
		if cs[c.Name] {
			selected.Corpora = append(selected.Corpora, c)
		}
	}
	return selected, nil
}

func (c Corpus) CorpusPath(root string) string {
	if c.Path != "" {
		return filepath.Join(root, c.Path)
	}
	return filepath.Join(root, "testdata/generated/corpora", c.Name+".jsonl")
}

func ReferencePath(root string, model Model, corpus Corpus) string {
	return filepath.Join(root, "testdata/generated/suite", model.Name, corpus.Name+".jsonl")
}

type poolManifest struct {
	Schema  int `json:"schema"`
	Corpora []struct {
		ID               string `json:"id"`
		File             string `json:"file"`
		SHA256           string `json:"sha256"`
		CompressedSHA256 string `json:"compressed_sha256"`
		Bytes            int    `json:"bytes"`
		Scalars          int    `json:"scalars"`
		SourceFile       string `json:"source_file"`
		SourceSHA256     string `json:"source_sha256"`
		Batches          []struct {
			Size     int    `json:"size"`
			File     string `json:"file"`
			SHA256   string `json:"sha256"`
			Requests int    `json:"requests"`
		} `json:"batches"`
	} `json:"corpora"`
}

// PoolRecords validates the vendored hashes and exact UTF-8 request boundaries.
// It returns slices of one shared pool, not independently copied request buffers.
func PoolRecords(root string, c Corpus) ([]string, error) {
	dir := filepath.Join(root, "testdata/corpora/text")
	manifestBytes, err := os.ReadFile(filepath.Join(dir, "MANIFEST.json"))
	if err != nil {
		return nil, err
	}
	var manifest poolManifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return nil, err
	}
	if manifest.Schema != 1 {
		return nil, fmt.Errorf("unsupported text corpus schema %d", manifest.Schema)
	}
	for _, p := range manifest.Corpora {
		if p.ID != c.Pool {
			continue
		}
		if _, err := checkedFile(dir, p.SourceFile, p.SourceSHA256); err != nil {
			return nil, err
		}
		compressed, err := checkedFile(dir, p.File, p.CompressedSHA256)
		if err != nil {
			return nil, err
		}
		gz, err := gzip.NewReader(bytes.NewReader(compressed))
		if err != nil {
			return nil, err
		}
		raw, err := io.ReadAll(gz)
		closeErr := gz.Close()
		if err != nil {
			return nil, err
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if len(raw) != p.Bytes || !utf8.Valid(raw) || utf8.RuneCount(raw) != p.Scalars || digest(raw) != p.SHA256 {
			return nil, fmt.Errorf("pool %s: text size, UTF-8 or SHA-256 mismatch", p.ID)
		}
		text := string(raw)
		for _, b := range p.Batches {
			if b.Size != c.RequestBytes {
				continue
			}
			data, err := checkedFile(dir, b.File, b.SHA256)
			if err != nil {
				return nil, err
			}
			var spans []struct {
				Start int `json:"start"`
				End   int `json:"end"`
			}
			if err := json.Unmarshal(data, &spans); err != nil {
				return nil, err
			}
			if len(spans) != b.Requests || len(spans) == 0 {
				return nil, fmt.Errorf("pool %s: request count mismatch", p.ID)
			}
			records := make([]string, 0, len(spans))
			end := 0
			for i, span := range spans {
				if span.Start != end || span.End <= span.Start || span.End > len(text) || span.End-span.Start > b.Size || !utf8.ValidString(text[span.Start:span.End]) {
					return nil, fmt.Errorf("pool %s: invalid request %d boundary", p.ID, i)
				}
				records = append(records, text[span.Start:span.End])
				end = span.End
			}
			if end != len(text) {
				return nil, fmt.Errorf("pool %s: incomplete requests", p.ID)
			}
			return records, nil
		}
		return nil, fmt.Errorf("pool %s: no request size %d", p.ID, c.RequestBytes)
	}
	return nil, fmt.Errorf("unknown pool %q", c.Pool)
}

// PrepareCorpus materializes deterministic JSONL for the Rust oracle. Original
// JSONL corpora are left untouched, including their empty records.
func PrepareCorpus(root string, c Corpus) error {
	if c.Path != "" {
		_, err := os.Stat(c.CorpusPath(root))
		return err
	}
	data, err := poolJSONL(root, c)
	if err != nil {
		return err
	}
	path := c.CorpusPath(root)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// ValidateCorpus prevents stale derived JSONL from being benchmarked under a
// changed pool/request-size configuration. It does not regenerate anything.
func ValidateCorpus(root string, c Corpus) error {
	if c.Path != "" {
		// hforacle.Open validates original JSONL hashes.
		return nil
	}
	want, err := poolJSONL(root, c)
	if err != nil {
		return err
	}
	got, err := os.ReadFile(c.CorpusPath(root))
	if err != nil {
		return err
	}
	if !bytes.Equal(got, want) {
		return fmt.Errorf("corpus %s: stale derived JSONL; run make generate-suite-fixtures", c.Name)
	}
	return nil
}

func poolJSONL(root string, c Corpus) ([]byte, error) {
	records, err := PoolRecords(root, c)
	if err != nil {
		return nil, err
	}
	var data bytes.Buffer
	encoder := json.NewEncoder(&data)
	encoder.SetEscapeHTML(false)
	for _, text := range records {
		if err := encoder.Encode(struct {
			Text string `json:"text"`
		}{text}); err != nil {
			return nil, err
		}
	}
	return data.Bytes(), nil
}

func checkedFile(root, path, hash string) ([]byte, error) {
	if !filepath.IsLocal(path) {
		return nil, fmt.Errorf("non-local pool path %q", path)
	}
	data, err := os.ReadFile(filepath.Join(root, path))
	if err != nil {
		return nil, err
	}
	if digest(data) != hash {
		return nil, fmt.Errorf("SHA-256 mismatch: %s", path)
	}
	return data, nil
}

func digest(data []byte) string { hash := sha256.Sum256(data); return hex.EncodeToString(hash[:]) }
