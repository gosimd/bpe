package bpe_test

import (
	"reflect"
	"slices"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/gosimd/bpe"
)

func TestModelAppendIDsReusesDestination(t *testing.T) {
	model := newAppendModel(t)
	dst := make([]uint32, 0, 8)
	var err error

	dst, err = model.AppendIDs(dst, "abababab")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{2, 2, 2, 2}; !reflect.DeepEqual(dst, want) {
		t.Fatalf("AppendIDs() = %v, want %v", dst, want)
	}

	before := &dst[:cap(dst)][0]
	dst, err = model.AppendIDs(dst[:0], "abababab")
	if err != nil {
		t.Fatal(err)
	}
	if after := &dst[:cap(dst)][0]; after != before {
		t.Fatal("AppendIDs() did not reuse destination capacity")
	}
}

func TestTokenizerAppendIDsReusesDestinationAndIsConcurrentSafe(t *testing.T) {
	model, err := bpe.NewModel(
		map[string]uint32{"Ġhello": 0, "Ġworld": 1, "!": 2},
		nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddPrefixSpace: true,
		AddedTokens: []bpe.AddedToken{
			bpe.NewAddedToken("<s>", 100, true),
		},
		PostProcessor: bpe.TemplateProcessing{
			Single:        "[BOS] $A [EOS]",
			Pair:          "$A $B",
			SpecialTokens: []bpe.TemplateSpecialToken{bpe.NewTemplateSpecialToken("[BOS]", 101), bpe.NewTemplateSpecialToken("[EOS]", 102)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	dst := make([]uint32, 0, 8)
	dst, err = tokenizer.AppendIDs(dst, "hello<s>world!")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{101, 0, 100, 1, 2, 102}; !reflect.DeepEqual(dst, want) {
		t.Fatalf("AppendIDs() = %v, want %v", dst, want)
	}

	before := &dst[:cap(dst)][0]
	dst, err = tokenizer.AppendIDs(dst[:0], "hello<s>world!")
	if err != nil {
		t.Fatal(err)
	}
	if after := &dst[:cap(dst)][0]; after != before {
		t.Fatal("AppendIDs() did not reuse destination capacity")
	}

	const goroutines = 16
	var group sync.WaitGroup
	failures := make(chan []uint32, goroutines)
	for range goroutines {
		group.Add(1)
		go func() {
			defer group.Done()
			ids, appendErr := tokenizer.AppendIDs(make([]uint32, 0, 8), "hello<s>world!")
			if appendErr != nil || !reflect.DeepEqual(ids, []uint32{101, 0, 100, 1, 2, 102}) {
				failures <- ids
			}
		}()
	}
	group.Wait()
	close(failures)
	for ids := range failures {
		t.Fatalf("concurrent AppendIDs() = %v", ids)
	}
}

func FuzzModelAppendIDsMatchesTokenize(f *testing.F) {
	model := newAppendModel(f)
	f.Add("abababab")
	f.Add("abba")
	f.Add("")
	f.Fuzz(func(t *testing.T, piece string) {
		if !utf8.ValidString(piece) {
			return
		}
		tokens, err := model.Tokenize(piece)
		if err != nil {
			t.Fatal(err)
		}
		ids, err := model.AppendIDs(nil, piece)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]uint32, len(tokens))
		for i, token := range tokens {
			want[i] = token.ID
		}
		if !slices.Equal(ids, want) {
			t.Fatalf("AppendIDs() = %v, Tokenize() IDs = %v", ids, want)
		}
	})
}

func FuzzTokenizerAppendIDsMatchesEncode(f *testing.F) {
	model, err := bpe.NewModel(
		map[string]uint32{"<unk>": 0},
		nil,
		bpe.ModelOptions{UnknownToken: "<unk>", FuseUnknown: false},
	)
	if err != nil {
		f.Fatal(err)
	}
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddPrefixSpace: true,
		AddedTokens:    []bpe.AddedToken{bpe.NewAddedToken("<s>", 1, true)},
		PostProcessor: bpe.TemplateProcessing{
			Single:        "[BOS] $A [EOS]",
			Pair:          "$A $B",
			SpecialTokens: []bpe.TemplateSpecialToken{bpe.NewTemplateSpecialToken("[BOS]", 2), bpe.NewTemplateSpecialToken("[EOS]", 3)},
		},
	})
	if err != nil {
		f.Fatal(err)
	}
	f.Add("hello<s>world", true)
	f.Add("é\ttext", false)
	f.Add("", true)
	f.Fuzz(func(t *testing.T, text string, addSpecialTokens bool) {
		if !utf8.ValidString(text) {
			return
		}
		options := bpe.EncodeOptions{AddSpecialTokens: addSpecialTokens}
		encoding, err := tokenizer.EncodeWithOptions(text, options)
		if err != nil {
			t.Fatal(err)
		}
		ids, err := tokenizer.AppendIDsWithOptions([]uint32{99}, text, options)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ids[1:], encoding.IDs) {
			t.Fatalf("AppendIDsWithOptions() = %v, EncodeWithOptions() IDs = %v", ids[1:], encoding.IDs)
		}
	})
}

func BenchmarkModelAppendIDs(b *testing.B) {
	model := newAppendModel(b)
	dst := make([]uint32, 0, 8)
	dst, _ = model.AppendIDs(dst, "abababab")

	b.ReportAllocs()
	b.SetBytes(8)
	b.ResetTimer()
	for range b.N {
		dst, _ = model.AppendIDs(dst[:0], "abababab")
	}
}

func BenchmarkTokenizerAppendIDs(b *testing.B) {
	model, err := bpe.NewModel(
		map[string]uint32{"Ġhello": 0, "Ġworld": 1, "!": 2},
		nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	if err != nil {
		b.Fatal(err)
	}
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{AddPrefixSpace: true})
	if err != nil {
		b.Fatal(err)
	}
	const input = "hello world!"
	dst := make([]uint32, 0, 8)
	dst, _ = tokenizer.AppendIDs(dst, input)
	templateTokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddPrefixSpace: true,
		AddedTokens:    []bpe.AddedToken{bpe.NewAddedToken("<s>", 100, true)},
		PostProcessor: bpe.TemplateProcessing{
			Single:        "[BOS] $A [EOS]",
			Pair:          "$A $B",
			SpecialTokens: []bpe.TemplateSpecialToken{bpe.NewTemplateSpecialToken("[BOS]", 101), bpe.NewTemplateSpecialToken("[EOS]", 102)},
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	const templateInput = "hello<s>world!"
	templateDst := make([]uint32, 0, 8)
	templateDst, _ = templateTokenizer.AppendIDs(templateDst, templateInput)

	b.Run("AppendIDs", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(input)))
		for range b.N {
			dst, _ = tokenizer.AppendIDs(dst[:0], input)
		}
	})
	b.Run("AppendIDsTemplateAdded", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(templateInput)))
		for range b.N {
			templateDst, _ = templateTokenizer.AppendIDs(templateDst[:0], templateInput)
		}
	})
	b.Run("Encode", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(input)))
		for range b.N {
			if _, encodeErr := tokenizer.Encode(input); encodeErr != nil {
				b.Fatal(encodeErr)
			}
		}
	})
}

type testFataler interface {
	Helper()
	Fatal(...any)
}

func newAppendModel(t testFataler) *bpe.Model {
	t.Helper()
	model, err := bpe.NewModel(
		map[string]uint32{"a": 0, "b": 1, "ab": 2},
		[]bpe.Merge{{Left: "a", Right: "b"}},
		bpe.ModelOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	return model
}
