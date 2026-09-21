package pretokenize

import (
	"errors"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestByteLevelMapping(t *testing.T) {
	for _, c := range []struct{ input, want string }{
		{"\x00 \x7f", "ĀĠġ"}, {" ′", "ĠâĢ²"}, {"é", "Ã©"},
	} {
		p, err := NewByteLevel(false, false)
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		err = p.Walk(c.input, func(piece Piece) error {
			count++
			if piece.Text() != c.want {
				t.Fatalf("%q -> %q, want %q", c.input, piece.Text(), c.want)
			}
			start, end, err := piece.OriginalRange(0, len(piece.Text()))
			if start != 0 || end != len(c.input) {
				t.Fatal("wrong whole-piece range", start, end)
			}
			return err
		})
		if err != nil || count != 1 {
			t.Fatal(count, err)
		}
	}
}

func TestByteLevelPrefixAlignment(t *testing.T) {
	for _, useRegex := range []bool{false, true} {
		p, err := NewByteLevel(true, useRegex)
		if err != nil {
			t.Fatal(err)
		}
		for _, text := range []string{"a", "猫", "\u0301", "🙂"} {
			err = p.Walk(text, func(piece Piece) error {
				for i := range len(piece.Text()) {
					start, end, err := piece.OriginalRange(i, i+1)
					if err != nil || start != 0 || end != len(text) {
						t.Fatalf("%q byte %d: [%d,%d), %v", text, i, start, end, err)
					}
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestPipelineCallbacks(t *testing.T) {
	p, err := NewByteLevel(false, true)
	if err != nil {
		t.Fatal(err)
	}
	sentinel := errors.New("stop")
	for _, textOnly := range []bool{false, true} {
		calls := 0
		emit := func(string) error { calls++; return sentinel }
		if textOnly {
			err = p.WalkTextUnsafe("hello, world", emit)
		} else {
			err = p.Walk("hello, world", func(v Piece) error { return emit(v.Text()) })
		}
		if err != sentinel || calls != 1 {
			t.Fatal("cancellation", calls, err)
		}
		calls = 0
		if textOnly {
			err = p.WalkTextUnsafe("ok\xff", emit)
		} else {
			err = p.Walk("ok\xff", func(v Piece) error { return emit(v.Text()) })
		}
		if !errors.Is(err, ErrInvalidUTF8) || calls != 0 {
			t.Fatal("validation", calls, err)
		}
	}
}

func FuzzByteLevelPipeline(f *testing.F) {
	for _, s := range []string{"Hello, world!", "é\n  test", "it's", "\xff"} {
		f.Add(s)
	}
	p, err := NewByteLevel(false, true)
	if err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, text string) {
		var aligned, plain []string
		previous := 0
		err := p.Walk(text, func(piece Piece) error {
			start, end, err := piece.OriginalRange(0, len(piece.Text()))
			if err != nil || start != previous || end <= start || end > len(text) || !utf8.ValidString(text[start:end]) {
				t.Fatalf("invalid range [%d,%d) / %q: %v", start, end, text, err)
			}
			previous = end
			aligned = append(aligned, strings.Clone(piece.Text()))
			return nil
		})
		refErr := p.WalkTextUnsafe(text, func(piece string) error { plain = append(plain, strings.Clone(piece)); return nil })
		if !utf8.ValidString(text) {
			if !errors.Is(err, ErrInvalidUTF8) || !errors.Is(refErr, ErrInvalidUTF8) || len(aligned) != 0 || len(plain) != 0 {
				t.Fatal("UTF8 contract")
			}
			return
		}
		if err != nil || refErr != nil || !slices.Equal(aligned, plain) || previous != len(text) {
			t.Fatal("walk/text parity", err, refErr)
		}
	})
}
