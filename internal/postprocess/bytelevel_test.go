package postprocess

import (
	"reflect"
	"testing"
)

func TestByteLevelTrimOffsets(t *testing.T) {
	tokens := []Token{
		{Value: "Ġ", Start: 0, End: 1},
		{Value: "ĠĠĠĠHelloĠĠ", Start: 0, End: 11},
		{Value: "ĠĠHello", Start: 11, End: 18},
		{Value: "HelloĠĠ", Start: 18, End: 25},
		{Value: "ĠĠĠĠ", Start: 25, End: 29},
	}

	got := NewByteLevel(true, true).Process(tokens, nil, false, true)
	want := []Token{
		{Value: "Ġ", Start: 0, End: 0, SequenceID: 0},
		{Value: "ĠĠĠĠHelloĠĠ", Start: 4, End: 9, SequenceID: 0},
		{Value: "ĠĠHello", Start: 13, End: 18, SequenceID: 0},
		{Value: "HelloĠĠ", Start: 18, End: 23, SequenceID: 0},
		{Value: "ĠĠĠĠ", Start: 29, End: 29, SequenceID: 0},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Process() = %#v, want %#v", got, want)
	}
}

func TestByteLevelPrefixSpaceAndPairAreIndependent(t *testing.T) {
	first := []Token{{ID: 1, Value: "Ġone", Start: 0, End: 3}}
	second := []Token{{ID: 2, Value: "Ġtwo", Start: 0, End: 3}}

	got := NewByteLevel(true, true).Process(first, second, true, true)
	want := []Token{
		{ID: 1, Value: "Ġone", Start: 0, End: 3, SequenceID: 0},
		{ID: 2, Value: "Ġtwo", Start: 0, End: 3, TypeID: 1, SequenceID: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Process() = %#v, want %#v", got, want)
	}
}

func TestByteLevelCanDisableTrimming(t *testing.T) {
	got := NewByteLevel(false, false).Process(
		[]Token{{Value: "ĠwordĠ", Start: 0, End: 6}}, nil, false, true,
	)
	if got[0].Start != 0 || got[0].End != 6 {
		t.Fatalf("offset = (%d, %d), want (0, 6)", got[0].Start, got[0].End)
	}
}

func FuzzByteLevelTrimOffsets(f *testing.F) {
	f.Add("ĠHelloĠ", 0, 7, true)
	f.Add("\tword\n", 4, 10, false)
	f.Add("ĠĠĠ", 2, 5, true)
	f.Fuzz(func(t *testing.T, value string, start, end int, addPrefixSpace bool) {
		if start < 0 || end < start || end > 1<<30 {
			return
		}
		got := NewByteLevel(addPrefixSpace, true).Process(
			[]Token{{Value: value, Start: start, End: end}}, nil, false, true,
		)
		if got[0].Start < start || got[0].Start > got[0].End || got[0].End > end {
			t.Fatalf("trimmed offset (%d, %d) is outside (%d, %d)", got[0].Start, got[0].End, start, end)
		}
	})
}
