package bytelevel

import (
	"strings"
	"testing"
)

func TestSplitMatchesGPT2Pattern(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{input: "Hello, world!", want: []string{"Hello", ",", " world", "!"}},
		{input: "a  b", want: []string{"a", " ", " b"}},
		{input: "it's", want: []string{"it", "'s"}},
		{input: "  ", want: []string{"  "}},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			pieces := Split(test.input)
			got := make([]string, len(pieces))
			for i, piece := range pieces {
				got[i] = test.input[piece.Start:piece.End]
			}
			if len(got) != len(test.want) {
				t.Fatalf("Split(%q) = %q, want %q", test.input, got, test.want)
			}
			for i := range got {
				if got[i] != test.want[i] {
					t.Fatalf("Split(%q) = %q, want %q", test.input, got, test.want)
				}
			}
		})
	}
}

func TestMapUsesGPT2ByteAlphabetAndPreservesBoundaries(t *testing.T) {
	mapped := Map("\x00 \x7f", 0, func(offset int) int { return offset })
	if got, want := mapped.Text, "ĀĠġ"; got != want {
		t.Fatalf("Map() text = %q, want %q", got, want)
	}
	start, end := mapped.RawRange(0, 2)
	if got, want := [2]int{start, end}, [2]int{0, 1}; got != want {
		t.Fatalf("RawRange(first byte) = %v, want %v", got, want)
	}
	start, end = mapped.RawRange(0, len(mapped.Text))
	if got, want := [2]int{start, end}, [2]int{0, 3}; got != want {
		t.Fatalf("RawRange(full piece) = %v, want %v", got, want)
	}
}

// Hugging Face tokenizers 0.23.2 aligns every ByteLevel byte derived from a
// UTF-8 scalar with the scalar's complete original byte range.
func TestMapRawRangeExpandsPartialUTF8Scalar(t *testing.T) {
	mapped := Map(" ′", 0, func(offset int) int { return offset })
	if got, want := mapped.Text, "ĠâĢ²"; got != want {
		t.Fatalf("Map() text = %q, want %q", got, want)
	}
	start, end := mapped.RawRange(0, 6)
	if got, want := [2]int{start, end}, [2]int{0, 4}; got != want {
		t.Fatalf("RawRange(0, 6) = %v, want %v", got, want)
	}
	start, end = mapped.RawRange(6, 8)
	if got, want := [2]int{start, end}, [2]int{1, 4}; got != want {
		t.Fatalf("RawRange(6, 8) = %v, want %v", got, want)
	}
}

func TestAppendSplitAndMapReuseDestinations(t *testing.T) {
	pieces := make([]Piece, 0, 4)
	pieces = AppendSplit(pieces, "Hello, world!")
	wantPieces := Split("Hello, world!")
	if len(pieces) != len(wantPieces) {
		t.Fatalf("AppendSplit() = %#v, want %#v", pieces, wantPieces)
	}
	for i := range pieces {
		if pieces[i] != wantPieces[i] {
			t.Fatalf("AppendSplit() = %#v, want %#v", pieces, wantPieces)
		}
	}

	mapped := make([]byte, 1, 16)
	mapped[0] = 'x'
	mapped = AppendMap(mapped[:0], "\x00 \x7f")
	if got, want := string(mapped), "ĀĠġ"; got != want {
		t.Fatalf("AppendMap() = %q, want %q", got, want)
	}
}

func FuzzSplitAndMap(f *testing.F) {
	f.Add("Hello, world!")
	f.Add("é\n  test")
	f.Add("it's")

	f.Fuzz(func(t *testing.T, input string) {
		input = strings.ToValidUTF8(input, "�")
		pieces := Split(input)
		end := 0
		for _, piece := range pieces {
			if piece.Start != end || piece.Start > piece.End || piece.End > len(input) {
				t.Fatalf("invalid piece %#v for %q", piece, input)
			}
			mapped := Map(input[piece.Start:piece.End], piece.Start, func(offset int) int { return offset })
			start, mappedEnd := mapped.RawRange(0, len(mapped.Text))
			if got := [2]int{start, mappedEnd}; got != [2]int{piece.Start, piece.End} {
				t.Fatalf("mapped range = %v, want %v", got, [2]int{piece.Start, piece.End})
			}
			end = piece.End
		}
		if end != len(input) {
			t.Fatalf("pieces cover [0,%d), want [0,%d)", end, len(input))
		}
	})
}
