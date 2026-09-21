package bpe_test

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gosimd/bpe"
)

const minimalTokenizerJSON = `{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"猫":3," ":4},"merges":[["a","b"]]},"pre_tokenizer":null}`

func tokenizerJSON(t testing.TB, fields map[string]json.RawMessage) []byte {
	t.Helper()
	var config map[string]json.RawMessage
	if err := json.Unmarshal([]byte(minimalTokenizerJSON), &config); err != nil {
		t.Fatal(err)
	}
	for k, v := range fields {
		if v == nil {
			delete(config, k)
		} else {
			config[k] = v
		}
	}
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestParseTokenizerIdentityAndModelFormats(t *testing.T) {
	for _, data := range [][]byte{
		[]byte(minimalTokenizerJSON),
		tokenizerJSON(t, map[string]json.RawMessage{"pre_tokenizer": nil}),
		[]byte(strings.Replace(minimalTokenizerJSON, `[["a","b"]]`, `["a b"]`, 1)),
		[]byte(strings.Replace(minimalTokenizerJSON, `"type":"BPE",`, ``, 1)),
	} {
		tok, err := bpe.ParseTokenizer(data)
		if err != nil {
			t.Fatal(err)
		}
		e, err := tok.Encode("ab ab猫")
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(e.IDs, []uint32{2, 4, 2, 3}) || !slices.Equal(e.Offsets, []bpe.Offset{{0, 2}, {2, 3}, {3, 5}, {5, 8}}) {
			t.Fatal(e)
		}
	}
	tok, err := bpe.ParseTokenizer(tokenizerJSON(t, map[string]json.RawMessage{"pre_tokenizer": json.RawMessage(`{"type":"WhitespaceSplit"}`)}))
	if err != nil {
		t.Fatal(err)
	}
	e, err := tok.Encode("ab ab")
	if err != nil || !slices.Equal(e.IDs, []uint32{2, 2}) || !slices.Equal(e.Offsets, []bpe.Offset{{0, 2}, {3, 5}}) {
		t.Fatal(e, err)
	}
}

func TestOpenTokenizerGPT2(t *testing.T) {
	loaded, err := bpe.OpenTokenizer("testdata/gpt2/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	model, err := bpe.OpenModel("testdata/gpt2/vocab.json", "testdata/gpt2/merges.txt", bpe.ModelOptions{})
	if err != nil {
		t.Fatal(err)
	}
	end := bpe.NewAddedToken("<|endoftext|>", 50256, true)
	end.Normalized = true
	manual, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{AddedTokens: []bpe.AddedToken{end}, PostProcessor: bpe.ByteLevelPostProcessor{AddPrefixSpace: true}})
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"", "Hello, world!", "é🙂 العربية 猫", "a<|endoftext|>b", " \t\r\n"} {
		got, err := loaded.EncodePair(text, " ab")
		if err != nil {
			t.Fatal(err)
		}
		want, err := manual.EncodePair(text, " ab")
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("%q: %+v != %+v (%v)", text, got, want, err)
		}
	}
	if _, err := bpe.OpenTokenizer("testdata/does-not-exist.json"); !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}
	for _, path := range []string{"testdata/qwen2.5/tokenizer.json", "testdata/qwen3.8/tokenizer.json"} {
		_, err := bpe.OpenTokenizer(path)
		if !errors.Is(err, bpe.ErrUnsupportedConfiguration) || !strings.Contains(err.Error(), "normalizer") {
			t.Fatalf("%s: %v", path, err)
		}
	}
	if _, err := bpe.OpenTokenizer("testdata/bloom/tokenizer.json"); err != nil {
		t.Fatal(err)
	}
}

func TestTokenizerJSONModelOptions(t *testing.T) {
	for _, c := range []struct {
		name, model, text string
		ids               []uint32
	}{
		{"fused-unknown", `{"type":"BPE","vocab":{"<unk>":0},"merges":[],"unk_token":"<unk>","fuse_unk":true}`, "xy", []uint32{0}},
		{"byte-fallback", `{"type":"BPE","vocab":{"<0x61>":0},"merges":[],"byte_fallback":true}`, "a", []uint32{0}},
		{"prefix", `{"type":"BPE","vocab":{"a":0,"##b":1,"ab":2},"merges":[["a","##b"]],"continuing_subword_prefix":"##"}`, "ab", []uint32{2}},
		{"suffix", `{"type":"BPE","vocab":{"a":0,"b</w>":1,"ab</w>":2},"merges":[["a","b</w>"]],"end_of_word_suffix":"</w>"}`, "ab", []uint32{2}},
		{"ignore-merges", `{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[],"ignore_merges":true,"dropout":0}`, "ab", []uint32{2}},
		{"large-ID", `{"type":"BPE","vocab":{"a":4294967295},"merges":[]}`, "a", []uint32{4294967295}},
	} {
		t.Run(c.name, func(t *testing.T) {
			tok, err := bpe.ParseTokenizer(tokenizerJSON(t, map[string]json.RawMessage{"model": json.RawMessage(c.model)}))
			if err != nil {
				t.Fatal(err)
			}
			e, err := tok.Encode(c.text)
			if err != nil || !slices.Equal(e.IDs, c.ids) || !slices.Equal(e.Offsets, []bpe.Offset{{0, len(c.text)}}) {
				t.Fatal(e, err)
			}
			ids, err := tok.AppendIDs(nil, c.text)
			if err != nil || !slices.Equal(ids, c.ids) {
				t.Fatal(ids, err)
			}
		})
	}
}

func TestTokenizerJSONTemplateValidation(t *testing.T) {
	valid := `{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[S]","type_id":1}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"[S]":{"id":"[S]","ids":[99],"tokens":["[S]"]}}}`
	for _, raw := range []string{
		strings.Replace(valid, `"ids":[99]`, `"ids":[null]`, 1),
		strings.Replace(valid, `"tokens":["[S]"]`, `"tokens":[null]`, 1),
		strings.Replace(valid, `"tokens":["[S]"]`, `"tokens":[]`, 1),
		strings.Replace(valid, `"type_id":0`, `"type_id":null`, 1),
		strings.Replace(valid, `"id":"A"`, `"id":"C"`, 1),
		strings.Replace(valid, `"id":"A"`, `"id":"B"`, 1),
		strings.Replace(valid, `"id":"B"`, `"id":"A"`, 1),
		strings.Replace(valid, `"[S]":{"id":"[S]"`, `"[S]":{"id":"[X]"`, 1),
	} {
		if tok, err := bpe.ParseTokenizer(tokenizerJSON(t, map[string]json.RawMessage{"post_processor": json.RawMessage(raw)})); tok != nil || !errors.Is(err, bpe.ErrInvalidTokenizer) {
			t.Fatal(tok, err, raw)
		}
	}
}

func TestTokenizerJSONAddedTokens(t *testing.T) {
	// Both extraction phases and single-word rules survive JSON loading.
	added := `[{"id":5,"content":"<tag>","special":false,"normalized":true,"single_word":true,"lstrip":true,"rstrip":true}]`
	tok, err := bpe.ParseTokenizer(tokenizerJSON(t, map[string]json.RawMessage{"added_tokens": json.RawMessage(added)}))
	if err != nil {
		t.Fatal(err)
	}
	e, err := tok.Encode("ab <tag> ab")
	if err != nil || !slices.Equal(e.IDs, []uint32{2, 5, 2}) || !slices.Equal(e.Offsets, []bpe.Offset{{0, 2}, {2, 9}, {9, 11}}) {
		t.Fatal(e, err)
	}
	e, err = tok.Encode("a<tag>b")
	if err != nil || slices.Contains(e.IDs, uint32(5)) {
		t.Fatal(e, err)
	}
	for _, bad := range []string{
		strings.Replace(added, `"id":5`, `"id":2`, 1),
		strings.Replace(added, `"content":"<tag>"`, `"content":""`, 1),
		strings.Replace(added, `"content":"<tag>"`, `"content":"ab"`, 1),
		strings.TrimSuffix(added, "]") + "," + strings.TrimPrefix(added, "["),
	} {
		if _, err := bpe.ParseTokenizer(tokenizerJSON(t, map[string]json.RawMessage{"added_tokens": json.RawMessage(bad)})); !errors.Is(err, bpe.ErrInvalidAddedToken) {
			t.Fatal(err)
		}
	}
}

func TestParseTokenizerRejectsInvalidConfiguration(t *testing.T) {
	for name, data := range map[string]string{
		"empty": "", "null": "null", "array": "[]", "trailing": minimalTokenizerJSON + " null",
		"duplicate":     strings.Replace(minimalTokenizerJSON, `"a":0`, `"a":0,"a":1`, 1),
		"surrogate":     strings.Replace(minimalTokenizerJSON, `"猫"`, `"\ud800"`, 1),
		"UTF8":          strings.Replace(minimalTokenizerJSON, "猫", "\xff", 1),
		"missing-model": `{}`, "null-model": `{"model":null}`,
		"null-vocab-ID":    strings.Replace(minimalTokenizerJSON, `"a":0`, `"a":null`, 1),
		"negative-ID":      strings.Replace(minimalTokenizerJSON, `"a":0`, `"a":-1`, 1),
		"duplicate-ID":     strings.Replace(minimalTokenizerJSON, `"b":1`, `"b":0`, 1),
		"null-merge-token": strings.Replace(minimalTokenizerJSON, `[["a","b"]]`, `[[null,"b"]]`, 1),
		"mixed-merges":     strings.Replace(minimalTokenizerJSON, `[["a","b"]]`, `[["a","b"],"a b"]`, 1),
		"unknown-field":    strings.Replace(minimalTokenizerJSON, `"version":"1.0"`, `"typo":true`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			tok, err := bpe.ParseTokenizer([]byte(data))
			if tok != nil || !errors.Is(err, bpe.ErrInvalidTokenizer) {
				t.Fatalf("%v, %v", tok, err)
			}
		})
	}
	for _, c := range []struct {
		key, raw    string
		unsupported bool
	}{
		{"version", `"2.0"`, true}, {"version", `null`, false},
		{"normalizer", `{"type":"NFC"}`, true}, {"padding", `{}`, true}, {"truncation", `{}`, true},
		{"model", `{"type":"WordPiece","vocab":{},"merges":[]}`, true},
		{"model", `{"type":"BPE","vocab":{},"merges":[],"dropout":0.1}`, true},
		{"model", `{"type":"BPE","vocab":{},"merges":[],"unk_token":""}`, true},
		{"pre_tokenizer", `{"type":"Split","pattern":{"Regex":"[a-z]+"},"behavior":"Isolated","invert":false}`, false},
		{"pre_tokenizer", `{"type":"ByteLevel","unknown":true}`, false},
		{"pre_tokenizer", `{"type":"Unknown"}`, false},
		{"post_processor", `{"type":"Sequence","processors":[]}`, true},
		{"post_processor", `{"type":"ByteLevel","trim_offsets":null}`, false},
		{"added_tokens", `null`, false}, {"added_tokens", `[{}]`, false},
		{"added_tokens", `[{"id":99,"content":"<tag>","special":true,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false}]`, false},
	} {
		t.Run(c.key+"/"+c.raw, func(t *testing.T) {
			tok, err := bpe.ParseTokenizer(tokenizerJSON(t, map[string]json.RawMessage{c.key: json.RawMessage(c.raw)}))
			if tok != nil || !errors.Is(err, bpe.ErrInvalidTokenizer) || (c.unsupported && !errors.Is(err, bpe.ErrUnsupportedConfiguration)) {
				t.Fatalf("%v, %v", tok, err)
			}
		})
	}
	data := tokenizerJSON(t, map[string]json.RawMessage{
		"pre_tokenizer": json.RawMessage(`{"type":"Sequence","pretokenizers":[{"type":"Metaspace","prepend_scheme":"first"}]}`),
		"added_tokens":  json.RawMessage(`[{"id":5,"content":"<tag>","special":true,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false}]`),
	})
	if _, err := bpe.ParseTokenizer(data); !errors.Is(err, bpe.ErrUnsupportedConfiguration) {
		t.Fatal(err)
	}
}

func TestParseTokenizerOwnershipAndConcurrency(t *testing.T) {
	data := tokenizerJSON(t, map[string]json.RawMessage{"pre_tokenizer": json.RawMessage(`{"type":"ByteLevel","add_prefix_space":false}`)})
	tok, err := bpe.ParseTokenizer(data)
	if err != nil {
		t.Fatal(err)
	}
	for i := range data {
		data[i] = 0
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 50 {
				e, err := tok.Encode("ab")
				if err != nil || !slices.Equal(e.IDs, []uint32{2}) {
					t.Error(e, err)
					return
				}
				ids, err := tok.AppendIDs([]uint32{99}, "ab")
				if err != nil || !slices.Equal(ids, []uint32{99, 2}) {
					t.Error(ids, err)
					return
				}
				_, _ = tok.Encode("bbbb")
				if !slices.Equal(e.Tokens, []string{"ab"}) || !slices.Equal(e.Offsets, []bpe.Offset{{0, 2}}) {
					t.Error("retained borrowed output", e)
					return
				}
			}
		})
	}
	wg.Wait()
}

func TestTokenizerJSONHFFixtures(t *testing.T) {
	data, err := os.ReadFile("testdata/hf_tokenizer_json_v0.23.2.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Version                            string `json:"tokenizers_version"`
		Offsets                            string
		Normalization, Padding, Truncation bool
		Source                             struct{ Path, SHA256 string }
		Model                              json.RawMessage
		Pipelines                          []struct {
			Name  string
			Pre   json.RawMessage `json:"pre_tokenizer"`
			Post  json.RawMessage `json:"post_processor"`
			Added json.RawMessage `json:"added_tokens"`
			Cases []struct {
				Text      string
				Second    *string
				Special   bool `json:"add_special_tokens"`
				IDs       []uint32
				Tokens    []string
				Offsets   [][2]int
				Types     []uint32 `json:"type_ids"`
				Sequences []*int   `json:"sequence_ids"`
				Mask      []uint32 `json:"special_tokens_mask"`
			}
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Version != "0.23.2" || fixture.Offsets != "byte" || fixture.Normalization || fixture.Padding || fixture.Truncation || len(fixture.Pipelines) != 28 {
		t.Fatal("unexpected metadata")
	}
	source, err := os.ReadFile(fixture.Source.Path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(source)) != fixture.Source.SHA256 {
		t.Fatal("source hash mismatch")
	}
	for _, p := range fixture.Pipelines {
		t.Run(p.Name, func(t *testing.T) {
			config := tokenizerJSON(t, map[string]json.RawMessage{"model": fixture.Model, "pre_tokenizer": p.Pre, "post_processor": p.Post, "added_tokens": p.Added})
			tok, err := bpe.ParseTokenizer(config)
			if err != nil {
				t.Fatal(err)
			}
			for i, c := range p.Cases {
				t.Run(fmt.Sprint(i), func(t *testing.T) {
					options := bpe.EncodeOptions{AddSpecialTokens: c.Special}
					var e bpe.Encoding
					if c.Second == nil {
						e, err = tok.EncodeWithOptions(c.Text, options)
					} else {
						e, err = tok.EncodePairWithOptions(c.Text, *c.Second, options)
					}
					if err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(e.IDs, c.IDs) || !slices.Equal(e.Tokens, c.Tokens) || !slices.Equal(e.TypeIDs, c.Types) || !slices.Equal(e.SpecialTokensMask, c.Mask) || len(e.Offsets) != len(c.Offsets) || len(e.SequenceIDs) != len(c.Sequences) {
						t.Fatalf("%q: %+v, want %+v", c.Text, e, c)
					}
					for j, o := range e.Offsets {
						seq := -1
						if c.Sequences[j] != nil {
							seq = *c.Sequences[j]
						}
						if [2]int{o.Start, o.End} != c.Offsets[j] || e.SequenceIDs[j] != seq {
							t.Fatalf("%q token %d: offset %v sequence %d, want %v / %d", c.Text, j, o, e.SequenceIDs[j], c.Offsets[j], seq)
						}
					}
					if c.Second == nil {
						ids, err := tok.AppendIDsWithOptions([]uint32{999}, c.Text, options)
						if err != nil || !slices.Equal(ids, append([]uint32{999}, c.IDs...)) {
							t.Fatal(ids, err)
						}
					}
				})
			}
		})
	}
}

func FuzzParseTokenizer(f *testing.F) {
	for _, seed := range []string{minimalTokenizerJSON, `null`, `{}`, `{"model":{"vocab":{"\ud800":0},"merges":[]}}`} {
		f.Add(seed, "ab 猫")
	}
	f.Fuzz(func(t *testing.T, config, text string) {
		if len(config) > 32768 || len(text) > 4096 {
			t.Skip()
		}
		tok, err := bpe.ParseTokenizer([]byte(config))
		if err != nil {
			return
		}
		e, encodeErr := tok.Encode(text)
		ids, idErr := tok.AppendIDs(nil, text)
		if (encodeErr == nil) != (idErr == nil) || (encodeErr == nil && !slices.Equal(e.IDs, ids)) {
			t.Fatal("Encode/AppendIDs disagreement", e, encodeErr, ids, idErr)
		}
	})
}

func ExampleOpenTokenizer() {
	tokenizer, err := bpe.OpenTokenizer("testdata/gpt2/tokenizer.json")
	if err != nil {
		panic(err)
	}
	e, err := tokenizer.Encode("Hello, world!")
	if err != nil {
		panic(err)
	}
	fmt.Println(e.IDs)
	// Output: [15496 11 995 0]
}
