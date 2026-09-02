package bpe_test

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"sync"
	"testing"

	"github.com/gosimd/bpe"
)

func TestTokenizerDefaultAddsPrefixSpace(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"ĠHello": 1}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizer(model)
	if err != nil {
		t.Fatal(err)
	}

	encoding, err := tokenizer.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{1}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, want)
	}
	if want := []bpe.Offset{{Start: 0, End: 5}}; !reflect.DeepEqual(encoding.Offsets, want) {
		t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, want)
	}
}

func TestTokenizerByteLevelSplittingAndOffsets(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{
			"Hello":  1,
			",":      2,
			"Ġworld": 3,
			"!":      4,
			"a":      5,
			"Ġ":      6,
			"Ġb":     7,
			"it":     8,
			"'s":     9,
		}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		input       string
		wantIDs     []uint32
		wantOffsets []bpe.Offset
	}{
		{
			input:       "Hello, world!",
			wantIDs:     []uint32{1, 2, 3, 4},
			wantOffsets: []bpe.Offset{{0, 5}, {5, 6}, {6, 12}, {12, 13}},
		},
		{
			input:       "a  b",
			wantIDs:     []uint32{5, 6, 7},
			wantOffsets: []bpe.Offset{{0, 1}, {1, 2}, {2, 4}},
		},
		{
			input:       "it's",
			wantIDs:     []uint32{8, 9},
			wantOffsets: []bpe.Offset{{0, 2}, {2, 4}},
		},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			encoding, err := tokenizer.Encode(test.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(encoding.IDs, test.wantIDs) {
				t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, test.wantIDs)
			}
			if !reflect.DeepEqual(encoding.Offsets, test.wantOffsets) {
				t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, test.wantOffsets)
			}
		})
	}
}

func TestTokenizerMapsOffsetsBackFromByteLevelUTF8(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"Ã©": 1}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{})
	if err != nil {
		t.Fatal(err)
	}

	encoding, err := tokenizer.Encode("é")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{1}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, want)
	}
	if want := []bpe.Offset{{Start: 0, End: 2}}; !reflect.DeepEqual(encoding.Offsets, want) {
		t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, want)
	}

	ids, err := tokenizer.AppendIDs([]uint32{99}, "é")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{99, 1}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("AppendIDs() = %v, want %v", ids, want)
	}
}

func TestTokenizerAddedTokenBypassesByteLevelAndBPE(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"a": 1, "b": 2}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddedTokens: []bpe.AddedToken{bpe.NewAddedToken("<|end|>", 100, true)},
	})
	if err != nil {
		t.Fatal(err)
	}

	encoding, err := tokenizer.Encode("a<|end|>b")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{1, 100, 2}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, want)
	}
	if want := []bpe.Offset{{0, 1}, {1, 8}, {8, 9}}; !reflect.DeepEqual(encoding.Offsets, want) {
		t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, want)
	}

	ids, err := tokenizer.AppendIDs([]uint32{9}, "a<|end|>b")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{9, 1, 100, 2}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("AppendIDs() = %v, want %v", ids, want)
	}
}

func TestTokenizerAddsPrefixSpaceToEveryOrdinaryAddedTokenSegment(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"Ġa": 1, "Ġb": 2}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddPrefixSpace: true,
		AddedTokens:    []bpe.AddedToken{bpe.NewAddedToken("<x>", 10, true)},
	})
	if err != nil {
		t.Fatal(err)
	}

	encoding, err := tokenizer.Encode("a<x>b")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{1, 10, 2}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, want)
	}
	if want := []bpe.Offset{{0, 1}, {1, 4}, {4, 5}}; !reflect.DeepEqual(encoding.Offsets, want) {
		t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, want)
	}
}

func TestTokenizerAddedTokenOptions(t *testing.T) {
	model := mustModel(t, map[string]uint32{}, nil, bpe.ModelOptions{})
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddedTokens: []bpe.AddedToken{
			{ID: 10, Content: "<mask>", SingleWord: true, LStrip: true, RStrip: true},
			{ID: 11, Content: "<mask>long"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	input := "A<mask>  <mask>\t <mask>long"
	encoding, err := tokenizer.Encode(input)
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{10, 11}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, want)
	}
	if want := []bpe.Offset{{7, 17}, {17, len(input)}}; !reflect.DeepEqual(encoding.Offsets, want) {
		t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, want)
	}
}

func TestTokenizerAddedTokenCanReuseSameModelEntry(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"<|endoftext|>": 7}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddedTokens: []bpe.AddedToken{bpe.NewAddedToken("<|endoftext|>", 7, true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.Encode("<|endoftext|>")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{7}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, want)
	}
}

func TestTokenizerRejectsInvalidAddedTokens(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"model": 1, "other": 2}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tests := []struct {
		name   string
		tokens []bpe.AddedToken
	}{
		{name: "empty", tokens: []bpe.AddedToken{{ID: 10}}},
		{name: "invalid UTF-8", tokens: []bpe.AddedToken{{ID: 10, Content: string([]byte{0xff})}}},
		{name: "duplicate content", tokens: []bpe.AddedToken{{ID: 10, Content: "x"}, {ID: 11, Content: "x"}}},
		{name: "duplicate ID", tokens: []bpe.AddedToken{{ID: 10, Content: "x"}, {ID: 10, Content: "y"}}},
		{name: "model content mismatch", tokens: []bpe.AddedToken{{ID: 2, Content: "model"}}},
		{name: "model ID collision", tokens: []bpe.AddedToken{{ID: 1, Content: "added"}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{AddedTokens: test.tokens})
			if !errors.Is(err, bpe.ErrInvalidAddedToken) {
				t.Fatalf("NewTokenizerWithOptions() error = %v, want ErrInvalidAddedToken", err)
			}
		})
	}
}

func TestNewAddedTokenDefaults(t *testing.T) {
	regular := bpe.NewAddedToken("regular", 1, false)
	if !regular.Normalized || regular.Special {
		t.Fatalf("regular token defaults = %#v", regular)
	}
	special := bpe.NewAddedToken("special", 2, true)
	if special.Normalized || !special.Special {
		t.Fatalf("special token defaults = %#v", special)
	}
}

func TestTokenizerCopiesAddedTokensAndIsConcurrentSafe(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"a": 1}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	added := []bpe.AddedToken{bpe.NewAddedToken("<x>", 10, true)}
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{AddedTokens: added})
	if err != nil {
		t.Fatal(err)
	}
	added[0].ID = 99
	added[0].Content = "changed"

	const goroutines = 16
	var group sync.WaitGroup
	failures := make(chan []uint32, goroutines)
	for range goroutines {
		group.Add(1)
		go func() {
			defer group.Done()
			encoding, encodeErr := tokenizer.Encode("a<x>")
			if encodeErr != nil || !reflect.DeepEqual(encoding.IDs, []uint32{1, 10}) {
				failures <- encoding.IDs
			}
		}()
	}
	group.Wait()
	close(failures)
	for ids := range failures {
		t.Fatalf("concurrent Encode() IDs = %v, want [1 10]", ids)
	}
}

func TestHFAddedTokenFixtures(t *testing.T) {
	type fixtureToken struct {
		ID         uint32 `json:"id"`
		Content    string `json:"content"`
		SingleWord bool   `json:"single_word"`
		LStrip     bool   `json:"lstrip"`
		RStrip     bool   `json:"rstrip"`
		Normalized bool   `json:"normalized"`
		Special    bool   `json:"special"`
	}
	type fixtureCase struct {
		Name           string            `json:"name"`
		Vocab          map[string]uint32 `json:"vocab"`
		AddPrefixSpace bool              `json:"add_prefix_space"`
		AddedTokens    []fixtureToken    `json:"added_tokens"`
		Input          string            `json:"input"`
		IDs            []uint32          `json:"ids"`
		Offsets        [][2]int          `json:"offsets"`
	}
	var fixture struct {
		Upstream string        `json:"upstream"`
		Version  string        `json:"version"`
		Cases    []fixtureCase `json:"cases"`
	}
	data, err := os.ReadFile("testdata/hf_added_tokens_v0.22.2.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Upstream != "huggingface/tokenizers" || fixture.Version != "0.22.2" {
		t.Fatalf("unexpected fixture source %q %q", fixture.Upstream, fixture.Version)
	}

	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			model := mustModel(t, test.Vocab, nil, bpe.ModelOptions{IgnoreMerges: true})
			added := make([]bpe.AddedToken, len(test.AddedTokens))
			for i, token := range test.AddedTokens {
				added[i] = bpe.AddedToken{
					ID:         token.ID,
					Content:    token.Content,
					SingleWord: token.SingleWord,
					LStrip:     token.LStrip,
					RStrip:     token.RStrip,
					Normalized: token.Normalized,
					Special:    token.Special,
				}
			}
			tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
				AddPrefixSpace: test.AddPrefixSpace,
				AddedTokens:    added,
			})
			if err != nil {
				t.Fatal(err)
			}
			encoding, err := tokenizer.Encode(test.Input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(encoding.IDs, test.IDs) {
				t.Fatalf("IDs = %v, want %v", encoding.IDs, test.IDs)
			}
			wantOffsets := make([]bpe.Offset, len(test.Offsets))
			for i, offset := range test.Offsets {
				wantOffsets[i] = bpe.Offset{Start: offset[0], End: offset[1]}
			}
			if !reflect.DeepEqual(encoding.Offsets, wantOffsets) {
				t.Fatalf("offsets = %#v, want %#v", encoding.Offsets, wantOffsets)
			}
		})
	}
}

func TestTokenizerValidation(t *testing.T) {
	if _, err := bpe.NewTokenizer(nil); !errors.Is(err, bpe.ErrNilModel) {
		t.Fatalf("NewTokenizer(nil) error = %v, want ErrNilModel", err)
	}
	if _, err := bpe.NewTokenizer(&bpe.Model{}); !errors.Is(err, bpe.ErrNilModel) {
		t.Fatalf("NewTokenizer(zero Model) error = %v, want ErrNilModel", err)
	}

	model := mustModel(t, map[string]uint32{"a": 1}, nil, bpe.ModelOptions{})
	tokenizer, err := bpe.NewTokenizer(model)
	if err != nil {
		t.Fatal(err)
	}
	invalid := string([]byte{0xff})
	if _, err := tokenizer.Encode(invalid); !errors.Is(err, bpe.ErrInvalidUTF8) {
		t.Fatalf("Encode() error = %v, want ErrInvalidUTF8", err)
	}
	dst := []uint32{9}
	got, err := tokenizer.AppendIDs(dst, invalid)
	if !errors.Is(err, bpe.ErrInvalidUTF8) {
		t.Fatalf("AppendIDs() error = %v, want ErrInvalidUTF8", err)
	}
	if !reflect.DeepEqual(got, dst) {
		t.Fatalf("AppendIDs() = %v, want unchanged %v", got, dst)
	}
}

type postProcessorFixtureEncoding struct {
	IDs               []uint32 `json:"ids"`
	TypeIDs           []uint32 `json:"type_ids"`
	Tokens            []string `json:"tokens"`
	Offsets           [][2]int `json:"offsets"`
	SpecialTokensMask []uint32 `json:"special_tokens_mask"`
	SequenceIDs       []int    `json:"sequence_ids"`
}

type postProcessorFixtureSpecial struct {
	Identifier string   `json:"identifier"`
	IDs        []uint32 `json:"ids"`
	Tokens     []string `json:"tokens"`
}

func TestHFPostProcessorFixtures(t *testing.T) {
	type templateCase struct {
		Name             string                        `json:"name"`
		Vocab            map[string]uint32             `json:"vocab"`
		AddPrefixSpace   bool                          `json:"add_prefix_space"`
		Single           string                        `json:"single"`
		Pair             string                        `json:"pair"`
		SpecialTokens    []postProcessorFixtureSpecial `json:"special_tokens"`
		First            string                        `json:"first"`
		Second           *string                       `json:"second"`
		AddSpecialTokens bool                          `json:"add_special_tokens"`
		Encoding         postProcessorFixtureEncoding  `json:"encoding"`
	}
	type byteLevelCase struct {
		Name               string                       `json:"name"`
		Vocab              map[string]uint32            `json:"vocab"`
		PreAddPrefixSpace  bool                         `json:"pre_add_prefix_space"`
		PostAddPrefixSpace bool                         `json:"post_add_prefix_space"`
		TrimOffsets        bool                         `json:"trim_offsets"`
		First              string                       `json:"first"`
		Second             *string                      `json:"second"`
		Encoding           postProcessorFixtureEncoding `json:"encoding"`
	}
	var fixture struct {
		Upstream       string          `json:"upstream"`
		Version        string          `json:"version"`
		TemplateCases  []templateCase  `json:"template_cases"`
		ByteLevelCases []byteLevelCase `json:"byte_level_cases"`
	}
	data, err := os.ReadFile("testdata/hf_post_processors_v0.22.2.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Upstream != "huggingface/tokenizers" || fixture.Version != "0.22.2" {
		t.Fatalf("unexpected fixture source %q %q", fixture.Upstream, fixture.Version)
	}

	for _, test := range fixture.TemplateCases {
		t.Run("Template/"+test.Name, func(t *testing.T) {
			model := mustModel(t, test.Vocab, nil, bpe.ModelOptions{IgnoreMerges: true})
			specials := make([]bpe.TemplateSpecialToken, len(test.SpecialTokens))
			for i, special := range test.SpecialTokens {
				specials[i] = bpe.TemplateSpecialToken{
					Identifier: special.Identifier,
					IDs:        special.IDs,
					Tokens:     special.Tokens,
				}
			}
			tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
				AddPrefixSpace: test.AddPrefixSpace,
				PostProcessor: bpe.TemplateProcessing{
					Single:        test.Single,
					Pair:          test.Pair,
					SpecialTokens: specials,
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			options := bpe.EncodeOptions{AddSpecialTokens: test.AddSpecialTokens}
			var got bpe.Encoding
			if test.Second == nil {
				got, err = tokenizer.EncodeWithOptions(test.First, options)
			} else {
				got, err = tokenizer.EncodePairWithOptions(test.First, *test.Second, options)
			}
			if err != nil {
				t.Fatal(err)
			}
			assertFixtureEncoding(t, got, test.Encoding)
		})
	}

	for _, test := range fixture.ByteLevelCases {
		t.Run("ByteLevel/"+test.Name, func(t *testing.T) {
			model := mustModel(t, test.Vocab, nil, bpe.ModelOptions{IgnoreMerges: true})
			tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
				AddPrefixSpace: test.PreAddPrefixSpace,
				PostProcessor: bpe.ByteLevelPostProcessor{
					AddPrefixSpace: test.PostAddPrefixSpace,
					TrimOffsets:    test.TrimOffsets,
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			var got bpe.Encoding
			if test.Second == nil {
				got, err = tokenizer.Encode(test.First)
			} else {
				got, err = tokenizer.EncodePair(test.First, *test.Second)
			}
			if err != nil {
				t.Fatal(err)
			}
			assertFixtureEncoding(t, got, test.Encoding)
		})
	}
}

func assertFixtureEncoding(t *testing.T, got bpe.Encoding, want postProcessorFixtureEncoding) {
	t.Helper()
	wantOffsets := make([]bpe.Offset, len(want.Offsets))
	for i, offset := range want.Offsets {
		wantOffsets[i] = bpe.Offset{Start: offset[0], End: offset[1]}
	}
	if !reflect.DeepEqual(got.IDs, want.IDs) {
		t.Errorf("IDs = %v, want %v", got.IDs, want.IDs)
	}
	if !reflect.DeepEqual(got.TypeIDs, want.TypeIDs) {
		t.Errorf("TypeIDs = %v, want %v", got.TypeIDs, want.TypeIDs)
	}
	if !reflect.DeepEqual(got.Tokens, want.Tokens) {
		t.Errorf("Tokens = %v, want %v", got.Tokens, want.Tokens)
	}
	if !reflect.DeepEqual(got.Offsets, wantOffsets) {
		t.Errorf("Offsets = %#v, want %#v", got.Offsets, wantOffsets)
	}
	if !reflect.DeepEqual(got.SpecialTokensMask, want.SpecialTokensMask) {
		t.Errorf("SpecialTokensMask = %v, want %v", got.SpecialTokensMask, want.SpecialTokensMask)
	}
	if !reflect.DeepEqual(got.SequenceIDs, want.SequenceIDs) {
		t.Errorf("SequenceIDs = %v, want %v", got.SequenceIDs, want.SequenceIDs)
	}
}

func TestTemplateProcessingAppendIDsCopiesConfigurationAndIsConcurrentSafe(t *testing.T) {
	model := mustModel(t, map[string]uint32{"a": 1}, nil, bpe.ModelOptions{IgnoreMerges: true})
	special := bpe.NewTemplateSpecialToken("[S]", 10)
	configuration := bpe.TemplateProcessing{
		Single:        "[S] $A [S]",
		Pair:          "$A $B",
		SpecialTokens: []bpe.TemplateSpecialToken{special},
	}
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{PostProcessor: configuration})
	if err != nil {
		t.Fatal(err)
	}
	configuration.SpecialTokens[0].IDs[0] = 99
	configuration.SpecialTokens[0].Tokens[0] = "changed"

	got, err := tokenizer.AppendIDs([]uint32{7}, "a")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{7, 10, 1, 10}; !reflect.DeepEqual(got, want) {
		t.Fatalf("AppendIDs() = %v, want %v", got, want)
	}

	const goroutines = 16
	var group sync.WaitGroup
	failures := make(chan []uint32, goroutines)
	for range goroutines {
		group.Add(1)
		go func() {
			defer group.Done()
			encoding, encodeErr := tokenizer.Encode("a")
			if encodeErr != nil || !reflect.DeepEqual(encoding.IDs, []uint32{10, 1, 10}) {
				failures <- encoding.IDs
			}
		}()
	}
	group.Wait()
	close(failures)
	for ids := range failures {
		t.Fatalf("concurrent Encode() IDs = %v, want [10 1 10]", ids)
	}
}

func TestPostProcessorValidation(t *testing.T) {
	model := mustModel(t, map[string]uint32{"a": 1}, nil, bpe.ModelOptions{})
	tests := []struct {
		name      string
		processor bpe.PostProcessor
		want      error
	}{
		{name: "nil byte level", processor: (*bpe.ByteLevelPostProcessor)(nil), want: bpe.ErrInvalidPostProcessor},
		{name: "nil template", processor: (*bpe.TemplateProcessing)(nil), want: bpe.ErrInvalidPostProcessor},
		{name: "missing special", processor: bpe.TemplateProcessing{Single: "[S] $A"}, want: bpe.ErrInvalidTemplate},
		{name: "bad pair", processor: bpe.TemplateProcessing{Pair: "$A"}, want: bpe.ErrInvalidTemplate},
		{name: "duplicate special", processor: bpe.TemplateProcessing{SpecialTokens: []bpe.TemplateSpecialToken{bpe.NewTemplateSpecialToken("X", 1), bpe.NewTemplateSpecialToken("X", 2)}}, want: bpe.ErrInvalidTemplate},
		{name: "unequal special values", processor: bpe.TemplateProcessing{SpecialTokens: []bpe.TemplateSpecialToken{{Identifier: "X", IDs: []uint32{1}}}}, want: bpe.ErrInvalidTemplate},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{PostProcessor: test.processor})
			if !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestPostProcessorDefaults(t *testing.T) {
	byteLevel := bpe.NewByteLevelPostProcessor()
	if !byteLevel.AddPrefixSpace || !byteLevel.TrimOffsets {
		t.Fatalf("NewByteLevelPostProcessor() = %#v", byteLevel)
	}

	model := mustModel(t, map[string]uint32{"a": 1, "b": 2}, nil, bpe.ModelOptions{IgnoreMerges: true})
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{PostProcessor: bpe.TemplateProcessing{}})
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.EncodePair("a", "b")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{1, 2}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("default template IDs = %v, want %v", encoding.IDs, want)
	}
	if want := []uint32{0, 1}; !reflect.DeepEqual(encoding.TypeIDs, want) {
		t.Fatalf("default template type IDs = %v, want %v", encoding.TypeIDs, want)
	}
}
