package benchsuite

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/gosimd/bpe/internal/model"
)

// ValidateModelFiles requires exported vocab/merges to describe exactly the
// model hashed into the HF oracle, even for rules the selected corpus never uses.
func ValidateModelFiles(root string, spec Model) error {
	data, err := os.ReadFile(filepath.Join(root, spec.Tokenizer))
	if err != nil {
		return err
	}
	var tokenizer struct {
		Model struct {
			Vocab  map[string]uint32 `json:"vocab"`
			Merges []json.RawMessage `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &tokenizer); err != nil {
		return err
	}
	vocabData, err := os.ReadFile(filepath.Join(root, spec.Vocab))
	if err != nil {
		return err
	}
	vocab, err := model.ParseVocabulary(vocabData)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(vocab, tokenizer.Model.Vocab) {
		return fmt.Errorf("model %s: vocab.json differs from tokenizer.json", spec.Name)
	}
	mergesData, err := os.ReadFile(filepath.Join(root, spec.Merges))
	if err != nil {
		return err
	}
	merges, err := model.ParseMerges(mergesData)
	if err != nil {
		return err
	}
	if len(merges) != len(tokenizer.Model.Merges) {
		return fmt.Errorf("model %s: merges count differs from tokenizer.json", spec.Name)
	}
	for i, raw := range tokenizer.Model.Merges {
		var pair []string
		if err := json.Unmarshal(raw, &pair); err != nil {
			var joined string
			if err := json.Unmarshal(raw, &joined); err != nil {
				return fmt.Errorf("model %s: invalid JSON merge %d: %w", spec.Name, i, err)
			}
			pair = strings.Split(joined, " ")
		}
		if len(pair) != 2 || pair[0] != merges[i].Left || pair[1] != merges[i].Right {
			return fmt.Errorf("model %s: merge %d differs from tokenizer.json", spec.Name, i)
		}
	}
	return nil
}

// ModelOptions reads only BPE options for the benchmark's OpenModel call. It is
// intentionally not a general pipeline loader. No vocabulary-size assumptions
// or narrowing of IDs/ranks are made here.
func ModelOptions(path string, pipeline string) (model.Options, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return model.Options{}, err
	}
	return parseModelOptions(data, pipeline)
}

func parseModelOptions(data []byte, pipeline string) (model.Options, error) {
	var config struct {
		Model struct {
			Type         string          `json:"type"`
			Vocab        json.RawMessage `json:"vocab"`
			Merges       json.RawMessage `json:"merges"`
			Dropout      *float64        `json:"dropout"`
			Unknown      string          `json:"unk_token"`
			Prefix       string          `json:"continuing_subword_prefix"`
			Suffix       string          `json:"end_of_word_suffix"`
			Fuse         bool            `json:"fuse_unk"`
			ByteFallback bool            `json:"byte_fallback"`
			IgnoreMerges bool            `json:"ignore_merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return model.Options{}, err
	}
	m := config.Model
	// Older GPT-2 JSON uses the untagged BPE representation.
	if (m.Type != "BPE" && m.Type != "") || len(m.Vocab) == 0 || len(m.Merges) == 0 {
		return model.Options{}, fmt.Errorf("benchmark requires a BPE model with vocab and merges")
	}
	if m.Dropout != nil && *m.Dropout != 0 {
		return model.Options{}, fmt.Errorf("BPE dropout is not supported by the deterministic Go benchmark")
	}
	if pipeline != "model-only" && pipeline != "gpt2" {
		return model.Options{}, fmt.Errorf("unsupported pipeline %q", pipeline)
	}
	if pipeline == "gpt2" {
		// This preset deliberately cannot be used to mislabel another pipeline
		// as GPT-2. Model-only benchmarks use actual transformed HF pieces.
		var actual map[string]any
		if err := json.Unmarshal(data, &actual); err != nil {
			return model.Options{}, err
		}
		var expected map[string]any
		_ = json.Unmarshal([]byte(`{
		  "normalizer":null,
		  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
		  "post_processor":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
		  "added_tokens":[{"id":50256,"special":true,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true}]
		}`), &expected)
		for _, field := range []string{"pre_tokenizer", "post_processor"} {
			if value, ok := actual[field].(map[string]any); ok {
				if _, exists := value["use_regex"]; !exists {
					value["use_regex"] = true
				}
			}
		}
		for field, want := range expected {
			if !reflect.DeepEqual(actual[field], want) {
				return model.Options{}, fmt.Errorf("gpt2 preset: unsupported %s; use model-only for other pipelines", field)
			}
		}
	}
	return model.Options{UnknownToken: m.Unknown, ContinuingSubwordPrefix: m.Prefix, EndOfWordSuffix: m.Suffix, FuseUnknown: m.Fuse, ByteFallback: m.ByteFallback, IgnoreMerges: m.IgnoreMerges}, nil
}
