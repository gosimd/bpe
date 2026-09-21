package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/gosimd/bpe/internal/addedtoken"
	"github.com/gosimd/bpe/internal/model"
	"github.com/gosimd/bpe/internal/pretokenize"
)

// ParseJSON loads the supported encoding subset of HF tokenizer.json. The
// compiled result owns its state and does not retain the input byte slice.
func ParseJSON(data []byte) (*Tokenizer, error) {
	t, err := parseJSON(data)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidTokenizer, err)
	}
	return t, nil
}

func parseJSON(data []byte) (*Tokenizer, error) {
	if err := validateJSON(data); err != nil {
		return nil, err
	}
	o, err := readObject(data)
	if err != nil {
		return nil, err
	}
	version, err := field(o, "version", "1.0", false)
	if err != nil {
		return nil, err
	}
	if version != "1.0" {
		return nil, fmt.Errorf("%w: version %q", ErrUnsupportedConfiguration, version)
	}
	for _, key := range []string{"normalizer", "padding", "truncation"} {
		if !isNull(o.take(key)) {
			return nil, fmt.Errorf("%w: %s must be null or absent", ErrUnsupportedConfiguration, key)
		}
	}
	// Decoder metadata has no effect on Encode/AppendIDs. This library does not
	// expose decoding; accepting it is not a claim of decoder support.
	if raw := o.take("decoder"); !isNull(raw) {
		d, err := readObject(raw)
		if err != nil {
			return nil, fmt.Errorf("decoder: %w", err)
		}
		if typ, err := field(d, "type", "", true); err != nil || typ == "" {
			return nil, fmt.Errorf("decoder: nonempty type required")
		}
	}
	m, err := parseModel(o.take("model"))
	if err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	added, err := parseAdded(o.take("added_tokens"), m)
	if err != nil {
		return nil, fmt.Errorf("added_tokens: %w", err)
	}
	pre := o.take("pre_tokenizer")
	if len(pre) == 0 {
		pre = json.RawMessage("null")
	}
	pipeline, err := pretokenize.CompileJSON(pre)
	if err != nil {
		if errors.Is(err, pretokenize.ErrUnsupported) {
			return nil, fmt.Errorf("%w: pre_tokenizer: %w", ErrUnsupportedConfiguration, err)
		}
		return nil, fmt.Errorf("pre_tokenizer: %w", err)
	}
	if len(added) > 0 && needsOriginalContext(pre) {
		return nil, fmt.Errorf("%w: pre_tokenizer Metaspace(first) with added_tokens requires segment context unavailable in pretok v0.2.0", ErrUnsupportedConfiguration)
	}
	processor, err := parsePostProcessor(o.take("post_processor"))
	if err != nil {
		return nil, fmt.Errorf("post_processor: %w", err)
	}
	if err := o.done(); err != nil {
		return nil, err
	}
	return New(m, added, processor, pipeline), nil
}

// Inspect only for the integration constraint; never rewrite the JSON passed
// to pretok. Its compiler already validated the complete component tree.
func needsOriginalContext(raw []byte) bool {
	var c struct {
		Type     string
		Prepend  string            `json:"prepend_scheme"`
		Children []json.RawMessage `json:"pretokenizers"`
	}
	_ = json.Unmarshal(raw, &c)
	if c.Type == "Metaspace" && c.Prepend == "first" {
		return true
	}
	for _, child := range c.Children {
		if needsOriginalContext(child) {
			return true
		}
	}
	return false
}

func parseModel(raw []byte) (*model.Model, error) {
	o, err := readObject(raw)
	if err != nil {
		return nil, err
	}
	typ, err := field(o, "type", "BPE", false)
	if err != nil {
		return nil, err
	}
	if typ != "BPE" {
		return nil, fmt.Errorf("%w: model type %q", ErrUnsupportedConfiguration, typ)
	}
	if dropout := o.take("dropout"); !isNull(dropout) {
		var d float64
		if err := json.Unmarshal(dropout, &d); err != nil {
			return nil, fmt.Errorf("dropout: %w", err)
		}
		if d != 0 {
			return nil, fmt.Errorf("%w: nonzero BPE dropout", ErrUnsupportedConfiguration)
		}
	}
	var options model.Options
	for _, f := range []struct {
		key string
		dst *string
	}{{"unk_token", &options.UnknownToken}, {"continuing_subword_prefix", &options.ContinuingSubwordPrefix}, {"end_of_word_suffix", &options.EndOfWordSuffix}} {
		if v := o.take(f.key); !isNull(v) {
			if err := json.Unmarshal(v, f.dst); err != nil {
				return nil, fmt.Errorf("%s: %w", f.key, err)
			}
			if f.key == "unk_token" && *f.dst == "" {
				return nil, fmt.Errorf("%w: empty unk_token cannot be represented", ErrUnsupportedConfiguration)
			}
		}
	}
	for _, f := range []struct {
		key string
		dst *bool
	}{{"fuse_unk", &options.FuseUnknown}, {"byte_fallback", &options.ByteFallback}, {"ignore_merges", &options.IgnoreMerges}} {
		// HF treats null model flags as unspecified.
		if v := o.take(f.key); !isNull(v) {
			if err := json.Unmarshal(v, f.dst); err != nil {
				return nil, fmt.Errorf("%s: %w", f.key, err)
			}
		}
	}
	vocab, err := model.ParseVocabulary(o.take("vocab"))
	if err != nil {
		return nil, err
	}
	mergesRaw, err := field(o, "merges", []json.RawMessage(nil), true)
	if err != nil {
		return nil, err
	}
	merges := make([]model.Merge, len(mergesRaw))
	format := byte(0)
	for i, v := range mergesRaw {
		var parts []string
		if len(v) > 0 && v[0] == '"' {
			if format == '[' {
				return nil, fmt.Errorf("%w: mixed merge formats", model.ErrInvalidMerge)
			}
			format = '"'
			var s string
			if err := json.Unmarshal(v, &s); err != nil {
				return nil, err
			}
			parts = strings.Split(s, " ")
		} else {
			if format == '"' {
				return nil, fmt.Errorf("%w: mixed merge formats", model.ErrInvalidMerge)
			}
			format = '['
			var err error
			parts, err = array[string](v)
			if err != nil {
				return nil, fmt.Errorf("%w: merge %d: %v", model.ErrInvalidMerge, i, err)
			}
		}
		if len(parts) != 2 {
			return nil, fmt.Errorf("%w: merge %d requires two strings", model.ErrInvalidMerge, i)
		}
		merges[i] = model.Merge{Left: parts[0], Right: parts[1]}
	}
	if err := o.done(); err != nil {
		return nil, err
	}
	return model.New(vocab, merges, options)
}

func parseAdded(raw []byte, m *model.Model) ([]addedtoken.Definition, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	if isNull(raw) {
		return nil, fmt.Errorf("expected array, not null")
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, err
	}
	result := make([]addedtoken.Definition, len(entries))
	next := uint64(m.VocabSize())
	for i, raw := range entries {
		o, err := readObject(raw)
		if err != nil {
			return nil, err
		}
		t := &result[i]
		t.ID, err = field(o, "id", uint32(0), true)
		if err != nil {
			return nil, err
		}
		t.Content, err = field(o, "content", "", true)
		if err != nil {
			return nil, err
		}
		// Serialized HF AddedToken fields are explicit, including normalized.
		for _, f := range []struct {
			key string
			dst *bool
		}{{"single_word", &t.SingleWord}, {"lstrip", &t.LStrip}, {"rstrip", &t.RStrip}, {"normalized", &t.Normalized}} {
			*f.dst, err = field(o, f.key, false, true)
			if err != nil {
				return nil, err
			}
		}
		if _, err := field(o, "special", false, true); err != nil {
			return nil, err
		}
		if err := o.done(); err != nil {
			return nil, err
		}
		// HF allocates IDs in serialized order rather than trusting supplied IDs.
		// Reject mismatches instead of silently changing IDs expected by inference.
		if _, exists := m.TokenToID(t.Content); !exists {
			if uint64(t.ID) != next {
				return nil, fmt.Errorf("%w: token %d ID %d, HF would assign %d", ErrInvalidAddedToken, i, t.ID, next)
			}
			next++
		}
	}
	if err := ValidateAddedTokens(m, result); err != nil {
		return nil, err
	}
	return result, nil
}
