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
	"unicode/utf8"

	"github.com/gosimd/bpe"
	"github.com/gosimd/bpe/internal/bytelevel"
)

func TestTokenizerRegexpHFFixtures(t *testing.T) {
	data, err := os.ReadFile("testdata/hf_regexp_pipeline_v0.23.2.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Version                            string `json:"tokenizers_version"`
		Offsets                            string
		Normalization, Padding, Truncation bool
		Source                             struct{ Path, SHA256 string }
		Cases                              []struct {
			Pattern, Processor, Text string
			Second                   *string
			Prefix                   bool `json:"add_prefix_space"`
			Special                  bool `json:"add_special_tokens"`
			Pieces                   []struct {
				Value   string
				Offsets [2]int
			}
			Encoding struct {
				IDs         []uint32
				Tokens      []string
				Offsets     [][2]int
				TypeIDs     []uint32 `json:"type_ids"`
				SequenceIDs []*int   `json:"sequence_ids"`
				SpecialMask []uint32 `json:"special_tokens_mask"`
			}
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Version != "0.23.2" || fixture.Offsets != "byte" || fixture.Normalization || fixture.Padding || fixture.Truncation || len(fixture.Cases) != 92 {
		t.Fatal("unexpected fixture metadata")
	}
	source, err := os.ReadFile(fixture.Source.Path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(source)) != fixture.Source.SHA256 {
		t.Fatal("source tokenizer hash mismatch")
	}
	model, err := bpe.OpenModel("testdata/gpt2/vocab.json", "testdata/gpt2/merges.txt", bpe.ModelOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range fixture.Cases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			p, err := bpe.NewRegexpPreTokenizer(c.Pattern)
			if err != nil {
				t.Fatal(err)
			}
			spans, err := p.Split(c.Text, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(spans) != len(c.Pieces) {
				t.Fatalf("pieces for %q: %v, want %v", c.Text, spans, c.Pieces)
			}
			for j, span := range spans {
				piece := c.Text[span.Start:span.End]
				if c.Prefix && !strings.HasPrefix(piece, " ") {
					piece = " " + piece
				}
				mapped := string(bytelevel.AppendMap(nil, piece))
				if mapped != c.Pieces[j].Value || [2]int{span.Start, span.End} != c.Pieces[j].Offsets {
					t.Fatalf("piece %d: %q %v, want %+v", j, mapped, span, c.Pieces[j])
				}
			}
			added := bpe.NewAddedToken("<tag>", 50257, true)
			added.LStrip, added.RStrip = true, true
			options := bpe.TokenizerOptions{PreTokenizer: p, AddPrefixSpace: c.Prefix, AddedTokens: []bpe.AddedToken{added}}
			switch c.Processor {
			case "bytelevel":
				options.PostProcessor = bpe.ByteLevelPostProcessor{AddPrefixSpace: c.Prefix, TrimOffsets: true}
			case "template":
				options.PostProcessor = bpe.TemplateProcessing{Single: "[S] $A", Pair: "[S] $A $B:1 [S]:1", SpecialTokens: []bpe.TemplateSpecialToken{bpe.NewTemplateSpecialToken("[S]", 50258)}}
			case "none":
			default:
				t.Fatal("unexpected processor")
			}
			tokenizer, err := bpe.NewTokenizerWithOptions(model, options)
			if err != nil {
				t.Fatal(err)
			}
			call := bpe.EncodeOptions{AddSpecialTokens: c.Special}
			var got bpe.Encoding
			if c.Second != nil {
				got, err = tokenizer.EncodePairWithOptions(c.Text, *c.Second, call)
			} else {
				got, err = tokenizer.EncodeWithOptions(c.Text, call)
			}
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.IDs, c.Encoding.IDs) || !slices.Equal(got.Tokens, c.Encoding.Tokens) || !slices.Equal(got.TypeIDs, c.Encoding.TypeIDs) || !slices.Equal(got.SpecialTokensMask, c.Encoding.SpecialMask) {
				t.Fatalf("%q: encoding %+v, want %+v", c.Text, got, c.Encoding)
			}
			if len(got.Offsets) != len(c.Encoding.Offsets) || len(got.SequenceIDs) != len(c.Encoding.SequenceIDs) {
				t.Fatal("metadata length mismatch")
			}
			for j, span := range got.Offsets {
				sequence := -1
				if c.Encoding.SequenceIDs[j] != nil {
					sequence = *c.Encoding.SequenceIDs[j]
				}
				if [2]int{span.Start, span.End} != c.Encoding.Offsets[j] || got.SequenceIDs[j] != sequence {
					t.Fatalf("%q: token %d offset/sequence %v/%d, want %v/%d", c.Text, j, span, got.SequenceIDs[j], c.Encoding.Offsets[j], sequence)
				}
			}
			if c.Second == nil {
				dst := make([]uint32, 1, len(got.IDs)+1)
				dst[0] = 999
				for range 3 {
					ids, err := tokenizer.AppendIDsWithOptions(dst[:1], c.Text, call)
					if err != nil || ids[0] != 999 || !slices.Equal(ids[1:], got.IDs) || &ids[0] != &dst[0] {
						t.Fatalf("AppendIDs: %v, %v", ids, err)
					}
				}
			}
		})
	}
}

func TestTokenizerRegexpNoSecondSplit(t *testing.T) {
	model := mustModel(t, map[string]uint32{"ab!": 1, "Ġab!": 2, "1": 3, "Ġ1": 4}, nil, bpe.ModelOptions{IgnoreMerges: true})
	for _, pattern := range []string{"[0-9]", "[0123456789]"} {
		p, err := bpe.NewRegexpPreTokenizer(pattern)
		if err != nil {
			t.Fatal(err)
		}
		tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{PreTokenizer: p, AddPrefixSpace: true})
		if err != nil {
			t.Fatal(err)
		}
		// Mutating the caller's wrapper must not change compiled Tokenizer state.
		*p = bpe.RegexpPreTokenizer{}
		got, err := tokenizer.Encode("ab!1")
		if err != nil || !slices.Equal(got.IDs, []uint32{2, 4}) || !slices.Equal(got.Offsets, []bpe.Offset{{0, 3}, {3, 4}}) {
			t.Fatalf("no second split/per-piece prefix: %+v %v", got, err)
		}
		ids, err := tokenizer.AppendIDs(nil, "ab!1")
		if err != nil || !slices.Equal(ids, got.IDs) {
			t.Fatal(ids, err)
		}
	}
	if _, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{PreTokenizer: new(bpe.RegexpPreTokenizer)}); !errors.Is(err, bpe.ErrUninitializedPreTokenizer) {
		t.Fatal(err)
	}
}

func TestTokenizerRegexpConcurrentAndInvalidUTF8(t *testing.T) {
	model := mustModel(t, map[string]uint32{"Ġab": 1, "Ġ1": 2, "Ġ!": 3}, nil, bpe.ModelOptions{IgnoreMerges: true})
	for _, pattern := range []string{"[0-9]", "[0123456789]"} {
		p, err := bpe.NewRegexpPreTokenizer(pattern)
		if err != nil {
			t.Fatal(err)
		}
		tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{PreTokenizer: p, AddPrefixSpace: true})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tokenizer.EncodePair("ab1!", "\xff"); !errors.Is(err, bpe.ErrInvalidUTF8) {
			t.Fatal(err)
		}
		backing := []uint32{90, 91, 92, 93}
		before := slices.Clone(backing)
		if got, err := tokenizer.AppendIDs(backing[:1], "ab1!\xff"); !errors.Is(err, bpe.ErrInvalidUTF8) || len(got) != 1 || !slices.Equal(backing, before) {
			t.Fatalf("error not atomic: %v %v", got, err)
		}
		want, err := tokenizer.Encode("ab1!")
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		for range 8 {
			wg.Go(func() {
				dst := make([]uint32, 0, 8)
				for range 100 {
					got, err := tokenizer.Encode("ab1!")
					if err != nil || !reflect.DeepEqual(got, want) {
						t.Errorf("concurrent encode: %+v %v", got, err)
						return
					}
					dst, err = tokenizer.AppendIDs(dst[:0], "ab1!")
					if err != nil || !slices.Equal(dst, want.IDs) {
						t.Errorf("concurrent append: %v %v", dst, err)
						return
					}
				}
			})
		}
		wg.Wait()
	}
}

func FuzzTokenizerRegexp(f *testing.F) {
	vocab := make(map[string]uint32, 256)
	for i := range 256 {
		vocab[string(bytelevel.AppendMap(nil, string([]byte{byte(i)})))] = uint32(i)
	}
	model, err := bpe.NewModel(vocab, nil, bpe.ModelOptions{})
	if err != nil {
		f.Fatal(err)
	}
	var tokenizers []*bpe.Tokenizer
	for _, pattern := range []string{"[0-9]", "[a-z]+", "", `\b`} {
		p, err := bpe.NewRegexpPreTokenizer(pattern)
		if err != nil {
			f.Fatal(err)
		}
		for _, prefix := range []bool{false, true} {
			tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
				PreTokenizer: p, AddPrefixSpace: prefix,
				AddedTokens:   []bpe.AddedToken{bpe.NewAddedToken("<tag>", 256, true)},
				PostProcessor: bpe.TemplateProcessing{Single: "[S] $A", SpecialTokens: []bpe.TemplateSpecialToken{bpe.NewTemplateSpecialToken("[S]", 257)}},
			})
			if err != nil {
				f.Fatal(err)
			}
			tokenizers = append(tokenizers, tokenizer)
		}
	}
	for _, text := range []string{"", "ab12猫<tag>3", "é\u0301 عربي\r\n 🙂", "\xff"} {
		f.Add(text, uint8(0))
	}
	f.Fuzz(func(t *testing.T, text string, index uint8) {
		tokenizer := tokenizers[int(index)%len(tokenizers)]
		got, err := tokenizer.Encode(text)
		ids, idErr := tokenizer.AppendIDs([]uint32{999}, text)
		if !utf8.ValidString(text) {
			if !errors.Is(err, bpe.ErrInvalidUTF8) || !errors.Is(idErr, bpe.ErrInvalidUTF8) || !slices.Equal(ids, []uint32{999}) {
				t.Fatal("UTF-8 contract")
			}
			return
		}
		if err != nil || idErr != nil || ids[0] != 999 || !slices.Equal(ids[1:], got.IDs) {
			t.Fatalf("encode/append mismatch: %+v %v / %v %v", got, err, ids, idErr)
		}
		for _, span := range got.Offsets {
			if span.Start < 0 || span.End < span.Start || span.End > len(text) || !utf8.ValidString(text[span.Start:span.End]) {
				t.Fatalf("invalid raw span %v in %q", span, text)
			}
		}
	})
}

func BenchmarkTokenizerRegexp(b *testing.B) {
	model, err := bpe.NewModel(map[string]uint32{"Ġhello": 1, "Ġ1": 2, "Ġ2": 3, "Ġ3": 4, "Ġworld": 5}, nil, bpe.ModelOptions{IgnoreMerges: true})
	if err != nil {
		b.Fatal(err)
	}
	const text = "hello123world"
	for _, c := range []struct{ name, pattern string }{{"pretok", "[0-9]"}, {"regexp", "[0123456789]"}} {
		p, err := bpe.NewRegexpPreTokenizer(c.pattern)
		if err != nil {
			b.Fatal(err)
		}
		tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{PreTokenizer: p, AddPrefixSpace: true})
		if err != nil {
			b.Fatal(err)
		}
		want := []uint32{1, 2, 3, 4, 5}
		b.Run(c.name+"/Encode", func(b *testing.B) {
			got, err := tokenizer.Encode(text)
			if err != nil || !slices.Equal(got.IDs, want) {
				b.Fatal(got, err)
			}
			b.ReportAllocs()
			b.SetBytes(int64(len(text)))
			b.ResetTimer()
			for range b.N {
				got, err = tokenizer.Encode(text)
				if err != nil {
					b.Fatal(err)
				}
			}
			if !slices.Equal(got.IDs, want) {
				b.Fatal(got)
			}
		})
		b.Run(c.name+"/AppendIDs", func(b *testing.B) {
			dst := make([]uint32, 0, len(want))
			dst, err = tokenizer.AppendIDs(dst, text)
			if err != nil || !slices.Equal(dst, want) {
				b.Fatal(dst, err)
			}
			b.ReportAllocs()
			b.SetBytes(int64(len(text)))
			b.ResetTimer()
			for range b.N {
				dst, err = tokenizer.AppendIDs(dst[:0], text)
				if err != nil {
					b.Fatal(err)
				}
			}
			if !slices.Equal(dst, want) {
				b.Fatal(dst)
			}
		})
	}
}

func ExampleTokenizer_regexp() {
	model, err := bpe.NewModel(map[string]uint32{"ab": 1, "1": 2, "2": 3}, nil, bpe.ModelOptions{IgnoreMerges: true})
	if err != nil {
		panic(err)
	}
	p, err := bpe.NewRegexpPreTokenizer(`[0-9]`)
	if err != nil {
		panic(err)
	}
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{PreTokenizer: p, AddPrefixSpace: false})
	if err != nil {
		panic(err)
	}
	encoding, err := tokenizer.Encode("ab12")
	if err != nil {
		panic(err)
	}
	fmt.Println(encoding.IDs)
	fmt.Println(encoding.Offsets)
	// Output:
	// [1 2 3]
	// [{0 2} {2 3} {3 4}]
}
