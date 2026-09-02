package bpe_test

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/gosimd/bpe"
)

func TestModelUsesGlobalMergePriority(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{
			"a": 0, "b": 1, "c": 2,
			"ab": 3, "bc": 4, "abc": 5,
		},
		[]bpe.Merge{
			{Left: "b", Right: "c"},
			{Left: "a", Right: "b"},
			{Left: "a", Right: "bc"},
		},
		bpe.ModelOptions{},
	)

	tokens, err := model.Tokenize("abc")
	if err != nil {
		t.Fatal(err)
	}
	want := []bpe.Token{{ID: 5, Value: "abc", Offset: bpe.Offset{Start: 0, End: 3}}}
	if !reflect.DeepEqual(tokens, want) {
		t.Fatalf("Tokenize() = %#v, want %#v", tokens, want)
	}
}

func TestModelPrefixAndSuffix(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{
			"a":       0,
			"##b</w>": 1,
			"ab</w>":  2,
		},
		[]bpe.Merge{{Left: "a", Right: "##b</w>"}},
		bpe.ModelOptions{
			ContinuingSubwordPrefix: "##",
			EndOfWordSuffix:         "</w>",
		},
	)

	tokens, err := model.Tokenize("ab")
	if err != nil {
		t.Fatal(err)
	}
	want := []bpe.Token{{ID: 2, Value: "ab</w>", Offset: bpe.Offset{Start: 0, End: 2}}}
	if !reflect.DeepEqual(tokens, want) {
		t.Fatalf("Tokenize() = %#v, want %#v", tokens, want)
	}
}

func TestModelUnknownsAndByteFallback(t *testing.T) {
	t.Run("fused unknown", func(t *testing.T) {
		model := mustModel(t,
			map[string]uint32{"<unk>": 0, "a": 1, "b": 2}, nil,
			bpe.ModelOptions{UnknownToken: "<unk>", FuseUnknown: true},
		)
		tokens, err := model.Tokenize("accb")
		if err != nil {
			t.Fatal(err)
		}
		want := []bpe.Token{
			{ID: 1, Value: "a", Offset: bpe.Offset{Start: 0, End: 1}},
			{ID: 0, Value: "<unk>", Offset: bpe.Offset{Start: 1, End: 3}},
			{ID: 2, Value: "b", Offset: bpe.Offset{Start: 3, End: 4}},
		}
		if !reflect.DeepEqual(tokens, want) {
			t.Fatalf("Tokenize() = %#v, want %#v", tokens, want)
		}
	})

	t.Run("byte fallback", func(t *testing.T) {
		model := mustModel(t,
			map[string]uint32{"<0xC3>": 1, "<0xA9>": 2}, nil,
			bpe.ModelOptions{ByteFallback: true},
		)
		tokens, err := model.Tokenize("é")
		if err != nil {
			t.Fatal(err)
		}
		want := []bpe.Token{
			{ID: 1, Value: "<0xC3>", Offset: bpe.Offset{Start: 0, End: 1}},
			{ID: 2, Value: "<0xA9>", Offset: bpe.Offset{Start: 1, End: 2}},
		}
		if !reflect.DeepEqual(tokens, want) {
			t.Fatalf("Tokenize() = %#v, want %#v", tokens, want)
		}
	})
}

func TestModelIgnoreMergesAndAppendIDs(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"a": 0, "b": 1, "ab": 2}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)

	tokens, err := model.Tokenize("ab")
	if err != nil {
		t.Fatal(err)
	}
	wantTokens := []bpe.Token{{ID: 2, Value: "ab", Offset: bpe.Offset{Start: 0, End: 2}}}
	if !reflect.DeepEqual(tokens, wantTokens) {
		t.Fatalf("Tokenize() = %#v, want %#v", tokens, wantTokens)
	}

	ids, err := model.AppendIDs([]uint32{99}, "ab")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{99, 2}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("AppendIDs() = %v, want %v", ids, want)
	}
}

func TestModelParsingAndValidation(t *testing.T) {
	model, err := bpe.ParseModel(
		[]byte(`{"a":0,"b":1,"ab":2}`),
		[]byte("#version: 0.2\na b\n"),
		bpe.ModelOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := model.AppendIDs(nil, "ab")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{2}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("AppendIDs() = %v, want %v", ids, want)
	}

	_, err = bpe.ParseModel([]byte(`{"a": 1.0}`), nil, bpe.ModelOptions{})
	if !errors.Is(err, bpe.ErrInvalidVocabulary) {
		t.Fatalf("ParseModel(non-integer ID) error = %v, want ErrInvalidVocabulary", err)
	}

	_, err = bpe.NewModel(map[string]uint32{"a": 0, "b": 0}, nil, bpe.ModelOptions{})
	if !errors.Is(err, bpe.ErrDuplicateTokenID) {
		t.Fatalf("NewModel(duplicate ID) error = %v, want ErrDuplicateTokenID", err)
	}

	_, err = bpe.NewModel(map[string]uint32{"a": 0, "b": 1}, []bpe.Merge{{Left: "a", Right: "b"}}, bpe.ModelOptions{})
	if !errors.Is(err, bpe.ErrInvalidMerge) {
		t.Fatalf("NewModel(missing merge result) error = %v, want ErrInvalidMerge", err)
	}

	_, err = bpe.NewModel(map[string]uint32{"a": 0}, nil, bpe.ModelOptions{UnknownToken: "<unk>"})
	if !errors.Is(err, bpe.ErrUnknownTokenNotInVocab) {
		t.Fatalf("NewModel(missing unknown token) error = %v, want ErrUnknownTokenNotInVocab", err)
	}
}

func TestOpenModel(t *testing.T) {
	directory := t.TempDir()
	vocabPath := filepath.Join(directory, "vocab.json")
	mergesPath := filepath.Join(directory, "merges.txt")
	if err := os.WriteFile(vocabPath, []byte(`{"a":0,"b":1,"ab":2}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mergesPath, []byte("#version: 0.2\na b\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	model, err := bpe.OpenModel(vocabPath, mergesPath, bpe.ModelOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ids, err := model.AppendIDs(nil, "ab")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{2}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("AppendIDs() = %v, want %v", ids, want)
	}
}

func TestModelRejectsInvalidUTF8WithoutChangingDestination(t *testing.T) {
	model := mustModel(t, map[string]uint32{"a": 0}, nil, bpe.ModelOptions{})
	invalid := string([]byte{0xff})

	if _, err := model.Tokenize(invalid); !errors.Is(err, bpe.ErrInvalidUTF8) {
		t.Fatalf("Tokenize() error = %v, want ErrInvalidUTF8", err)
	}
	dst := []uint32{7}
	got, err := model.AppendIDs(dst, invalid)
	if !errors.Is(err, bpe.ErrInvalidUTF8) {
		t.Fatalf("AppendIDs() error = %v, want ErrInvalidUTF8", err)
	}
	if !reflect.DeepEqual(got, dst) {
		t.Fatalf("AppendIDs() destination = %v, want unchanged %v", got, dst)
	}
}

func TestModelCopiesInputAndIsConcurrentSafe(t *testing.T) {
	vocab := map[string]uint32{"a": 0, "b": 1, "ab": 2}
	merges := []bpe.Merge{{Left: "a", Right: "b"}}
	model := mustModel(t, vocab, merges, bpe.ModelOptions{})
	vocab["a"] = 99
	merges[0] = bpe.Merge{Left: "b", Right: "a"}

	const goroutines = 16
	var group sync.WaitGroup
	errs := make(chan error, goroutines)
	for range goroutines {
		group.Add(1)
		go func() {
			defer group.Done()
			ids, err := model.AppendIDs(nil, "ab")
			if err != nil {
				errs <- err
				return
			}
			if !reflect.DeepEqual(ids, []uint32{2}) {
				errs <- errors.New("unexpected IDs")
			}
		}()
	}
	group.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

func mustModel(t *testing.T, vocab map[string]uint32, merges []bpe.Merge, options bpe.ModelOptions) *bpe.Model {
	t.Helper()
	model, err := bpe.NewModel(vocab, merges, options)
	if err != nil {
		t.Fatal(err)
	}
	return model
}
