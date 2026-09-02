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
	if got, want := mapped.RawOffset(0), 0; got != want {
		t.Fatalf("RawOffset(0) = %d, want %d", got, want)
	}
	if got, want := mapped.RawOffset(2), 1; got != want {
		t.Fatalf("RawOffset(2) = %d, want %d", got, want)
	}
	if got, want := mapped.RawOffset(len(mapped.Text)), 3; got != want {
		t.Fatalf("RawOffset(end) = %d, want %d", got, want)
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
			if got := mapped.RawOffset(0); got != piece.Start {
				t.Fatalf("start offset = %d, want %d", got, piece.Start)
			}
			if got := mapped.RawOffset(len(mapped.Text)); got != piece.End {
				t.Fatalf("end offset = %d, want %d", got, piece.End)
			}
			end = piece.End
		}
		if end != len(input) {
			t.Fatalf("pieces cover [0,%d), want [0,%d)", end, len(input))
		}
	})
}
