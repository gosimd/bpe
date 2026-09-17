package bpe_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/gosimd/bpe"
	"github.com/gosimd/pretok"
)

func TestRegexpPreTokenizerHFFixtures(t *testing.T) {
	data, err := os.ReadFile("testdata/hf_regexp_v0.23.2.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Version  string `json:"tokenizers_version"`
		Behavior string `json:"behavior"`
		Invert   bool   `json:"invert"`
		Offsets  string `json:"offsets"`
		Cases    []struct {
			Name, Pattern, Text string
			Pieces              []struct {
				Value   string
				Offsets [2]int
			}
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Version != "0.23.2" || fixture.Behavior != "Isolated" || fixture.Invert || fixture.Offsets != "byte" || len(fixture.Cases) != 80 {
		t.Fatal("unexpected fixture metadata/count")
	}
	for i, c := range fixture.Cases {
		t.Run(fmt.Sprintf("%s/%d", c.Name, i), func(t *testing.T) {
			p, err := bpe.NewRegexpPreTokenizer(c.Pattern)
			if err != nil {
				t.Fatal(err)
			}
			got, err := p.Split(c.Text, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(c.Pieces) {
				t.Fatalf("%q: got %v, want %v", c.Text, got, c.Pieces)
			}
			for j, span := range got {
				want := c.Pieces[j]
				if [2]int{span.Start, span.End} != want.Offsets || c.Text[span.Start:span.End] != want.Value {
					t.Fatalf("%q: piece %d = %v, want %+v", c.Text, j, span, want)
				}
			}
		})
	}
}

func TestRegexpPreTokenizerErrorsAndReuse(t *testing.T) {
	for _, pattern := range []string{"[", "(?=x)", `(a)\1`} {
		p, err := bpe.NewRegexpPreTokenizer(pattern)
		if p != nil || !errors.Is(err, bpe.ErrUnsupportedPattern) {
			t.Fatalf("%q: %v, %v", pattern, p, err)
		}
	}
	p, err := bpe.NewRegexpPreTokenizer("[0-9]")
	if err != nil {
		t.Fatal(err)
	}
	backing := []bpe.Offset{{Start: 90, End: 99}, {Start: 70, End: 80}, {Start: 50, End: 60}, {Start: 30, End: 40}, {Start: 10, End: 20}}
	before := slices.Clone(backing)
	for _, c := range []struct {
		p    *bpe.RegexpPreTokenizer
		text string
		want error
	}{
		{nil, "", bpe.ErrUninitializedPreTokenizer}, {new(bpe.RegexpPreTokenizer), "\xff", bpe.ErrUninitializedPreTokenizer},
		{p, "1valid\xff", bpe.ErrInvalidUTF8},
	} {
		got, err := c.p.Split(c.text, backing[:1])
		if !errors.Is(err, c.want) || len(got) != 1 || &got[0] != &backing[0] || !slices.Equal(backing, before) {
			t.Fatalf("error changed destination: %v %v", got, err)
		}
	}
	got, err := p.Split("α1猫", backing[:1])
	if err != nil {
		t.Fatal(err)
	}
	want := []bpe.Offset{{Start: 90, End: 99}, {Start: 0, End: 2}, {Start: 2, End: 3}, {Start: 3, End: 6}}
	if !slices.Equal(got, want) || &got[0] != &backing[0] {
		t.Fatalf("append/reuse: %v", got)
	}
	saved := slices.Clone(got)
	if _, err := p.Split("other input", nil); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, saved) {
		t.Fatal("output retained or overwritten")
	}
	empty, err := p.Split("", got)
	if err != nil || !slices.Equal(empty, got) {
		t.Fatal("empty input changed destination")
	}
}

func TestRegexpPreTokenizerConcurrent(t *testing.T) {
	p, err := bpe.NewRegexpPreTokenizer("[0-9]")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			dst := make([]bpe.Offset, 0, 8)
			for range 100 {
				var err error
				dst, err = p.Split("a12猫", dst[:0])
				if err != nil || !slices.Equal(dst, []bpe.Offset{{0, 1}, {1, 2}, {2, 3}, {3, 6}}) {
					t.Errorf("concurrent split: %v %v", dst, err)
					return
				}
			}
		})
	}
	wg.Wait()
}

func FuzzRegexpPreTokenizer(f *testing.F) {
	patterns := []string{"[0-9]", "[0-9][0-9][0-9]", `\p{N}{1,3}`, `\s+$`}
	for _, text := range []string{"", "abc123猫", "١٢٣４\r\n", "a1\xff", "e\u0301🙂"} {
		f.Add(text, uint8(0))
	}
	f.Fuzz(func(t *testing.T, text string, index uint8) {
		pattern := patterns[int(index)%len(patterns)]
		p, err := bpe.NewRegexpPreTokenizer(pattern)
		if err != nil {
			t.Fatal(err)
		}
		upstream, err := pretok.Compile(pattern)
		if err != nil {
			t.Fatal(err)
		}
		got, err := p.Split(text, nil)
		want, refErr := upstream.Split(text, nil)
		if !utf8.ValidString(text) {
			if !errors.Is(err, bpe.ErrInvalidUTF8) || !errors.Is(refErr, pretok.ErrInvalidUTF8) || got != nil {
				t.Fatal("UTF-8 contract")
			}
			return
		}
		if err != nil || refErr != nil || len(got) != len(want) {
			t.Fatalf("%q: %v %v / %v %v", text, got, err, want, refErr)
		}
		end := 0
		for i, span := range got {
			if span.Start != want[i].Start || span.End != want[i].End || span.Start != end || span.End <= span.Start || !utf8.ValidString(text[span.Start:span.End]) {
				t.Fatalf("%q: invalid span %v", text, span)
			}
			end = span.End
		}
		if end != len(text) {
			t.Fatal("input not covered")
		}
	})
}

func BenchmarkRegexpPreTokenizerSplit(b *testing.B) {
	p, err := bpe.NewRegexpPreTokenizer("[0-9]")
	if err != nil {
		b.Fatal(err)
	}
	text := "hello 123 мир 456 日本語"
	dst := make([]bpe.Offset, 0, 16)
	b.ReportAllocs()
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for range b.N {
		dst, err = p.Split(text, dst[:0])
		if err != nil {
			b.Fatal(err)
		}
	}
	if len(dst) != 9 {
		b.Fatal(dst)
	}
}

func ExampleRegexpPreTokenizer_Split() {
	p, err := bpe.NewRegexpPreTokenizer("[0-9]")
	if err != nil {
		panic(err)
	}
	text := "ab12猫"
	spans, err := p.Split(text, nil)
	if err != nil {
		panic(err)
	}
	for _, span := range spans {
		fmt.Printf("%q [%d,%d)\n", text[span.Start:span.End], span.Start, span.End)
	}
	// Output:
	// "ab" [0,2)
	// "1" [2,3)
	// "2" [3,4)
	// "猫" [4,7)
}
