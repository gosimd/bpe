package addedtoken

import (
	"testing"
	"unicode/utf8"
)

func TestSplitUsesLeftmostLongest(t *testing.T) {
	matcher := New([]Definition{
		{ID: 1, Content: "<x>"},
		{ID: 2, Content: "<x>long"},
	})
	got := matcher.Split("a<x>longb")
	want := []Segment{
		{Start: 0, End: 1},
		{Start: 1, End: 8, ID: 2, Added: true},
		{Start: 8, End: 9},
	}
	assertSegments(t, got, want)
}

func TestSplitNonNormalizedPhasePrecedesNormalizedPhase(t *testing.T) {
	matcher := New([]Definition{
		{ID: 1, Content: "a"},
		{ID: 2, Content: "ab", Normalized: true},
	})
	got := matcher.Split("ab")
	want := []Segment{
		{Start: 0, End: 1, ID: 1, Added: true},
		{Start: 1, End: 2},
	}
	assertSegments(t, got, want)
}

func TestSplitSingleWordUsesUnicodeWordBoundaries(t *testing.T) {
	matcher := New([]Definition{{ID: 7, Content: "<mask>", SingleWord: true}})
	input := "<mask>, A<mask> <mask>ony <mask> ◌̰<mask>"
	got := matcher.Split(input)

	var offsets [][2]int
	for _, segment := range got {
		if segment.Added {
			offsets = append(offsets, [2]int{segment.Start, segment.End})
		}
	}
	want := [][2]int{{0, 6}, {26, 32}}
	if len(offsets) != len(want) {
		t.Fatalf("added offsets = %v, want %v", offsets, want)
	}
	for i := range want {
		if offsets[i] != want[i] {
			t.Fatalf("added offsets = %v, want %v", offsets, want)
		}
	}
}

func TestSplitStripsUnicodeWhitespace(t *testing.T) {
	matcher := New([]Definition{{
		ID: 9, Content: "<mask>", LStrip: true, RStrip: true,
	}})
	input := "hi \t<mask>\u2000 there"
	got := matcher.Split(input)
	want := []Segment{
		{Start: 0, End: 2},
		{Start: 2, End: 14, ID: 9, Added: true},
		{Start: 14, End: len(input)},
	}
	assertSegments(t, got, want)
}

func TestSplitWithScratchMatchesSplit(t *testing.T) {
	matcher := New([]Definition{
		{ID: 1, Content: "<raw>", LStrip: true},
		{ID: 2, Content: "é", Normalized: true},
	})
	const input = "a <raw>b é"
	first := make([]Segment, 0, 8)
	second := make([]Segment, 0, 8)
	got, spare := matcher.SplitWithScratch(input, first, second)
	want := matcher.Split(input)
	assertSegments(t, got, want)

	got, _ = matcher.SplitWithScratch(input, got, spare)
	assertSegments(t, got, want)
}

func FuzzSplitCoversInput(f *testing.F) {
	f.Add("a <x> b")
	f.Add("é\t<x>\u2000")
	f.Fuzz(func(t *testing.T, input string) {
		if !utf8.ValidString(input) {
			t.Skip()
		}
		matcher := New([]Definition{
			{ID: 1, Content: "<x>", LStrip: true, RStrip: true},
			{ID: 2, Content: "é", SingleWord: true, Normalized: true},
		})
		segments := matcher.Split(input)
		position := 0
		for _, segment := range segments {
			if segment.Start != position {
				t.Fatalf("gap or overlap at %d: %#v", position, segments)
			}
			if segment.End < segment.Start || segment.End > len(input) {
				t.Fatalf("invalid segment %#v for input length %d", segment, len(input))
			}
			if !utf8.ValidString(input[segment.Start:segment.End]) {
				t.Fatalf("segment is not aligned to UTF-8 boundaries: %#v", segment)
			}
			position = segment.End
		}
		if position != len(input) {
			t.Fatalf("segments end at %d, want %d: %#v", position, len(input), segments)
		}
	})
}

func assertSegments(t *testing.T, got, want []Segment) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("Split() = %#v, want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Split() = %#v, want %#v", got, want)
		}
	}
}
