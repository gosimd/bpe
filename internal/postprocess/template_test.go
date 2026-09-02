package postprocess

import (
	"reflect"
	"testing"
)

func TestTemplateSinglePairAndSpecialToggle(t *testing.T) {
	processor, err := NewTemplate(
		"[CLS] $A [SEP]",
		"[CLS] $A [SEP] $B:1 [SEP]:1",
		[]SpecialToken{
			{Identifier: "[CLS]", IDs: []uint32{101}, Tokens: []string{"[CLS]"}},
			{Identifier: "[SEP]", IDs: []uint32{102}, Tokens: []string{"[SEP]"}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	first := []Token{{ID: 10, Value: "first", Start: 0, End: 5}}
	second := []Token{{ID: 20, Value: "second", Start: 0, End: 6}}

	got := processor.Process(first, second, true, true)
	want := []Token{
		{ID: 101, Value: "[CLS]", Special: true, SequenceID: NoSequence},
		{ID: 10, Value: "first", Start: 0, End: 5, SequenceID: 0},
		{ID: 102, Value: "[SEP]", Special: true, SequenceID: NoSequence},
		{ID: 20, Value: "second", Start: 0, End: 6, TypeID: 1, SequenceID: 1},
		{ID: 102, Value: "[SEP]", TypeID: 1, Special: true, SequenceID: NoSequence},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Process(pair) = %#v, want %#v", got, want)
	}

	got = processor.Process(first, second, true, false)
	want = []Token{
		{ID: 10, Value: "first", Start: 0, End: 5, SequenceID: 0},
		{ID: 20, Value: "second", Start: 0, End: 6, TypeID: 1, SequenceID: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Process(pair, no specials) = %#v, want %#v", got, want)
	}
}

func TestTemplateMultiTokenSpecialAndRepeatedSequence(t *testing.T) {
	processor, err := NewTemplate(
		"X:7 $A:3 X $A:4",
		"$A $B",
		[]SpecialToken{{Identifier: "X", IDs: []uint32{10, 11}, Tokens: []string{"x", "y"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	got := processor.Process([]Token{{ID: 1, Value: "a"}}, nil, false, true)
	if want := []uint32{10, 11, 1, 10, 11, 1}; !reflect.DeepEqual(tokenIDs(got), want) {
		t.Fatalf("IDs = %v, want %v", tokenIDs(got), want)
	}
	if want := []uint32{7, 7, 3, 0, 0, 4}; !reflect.DeepEqual(tokenTypeIDs(got), want) {
		t.Fatalf("type IDs = %v, want %v", tokenTypeIDs(got), want)
	}
}

func TestTemplateValidation(t *testing.T) {
	tests := []struct {
		name   string
		single string
		pair   string
	}{
		{name: "bad piece", single: "$C", pair: "$A $B"},
		{name: "bad type", single: "$A:no", pair: "$A $B"},
		{name: "single B", single: "$B", pair: "$A $B"},
		{name: "pair missing B", single: "$A", pair: "$A"},
		{name: "missing special", single: "[CLS] $A", pair: "$A $B"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewTemplate(test.single, test.pair, nil); err == nil {
				t.Fatal("NewTemplate() unexpectedly succeeded")
			}
		})
	}
}

func tokenIDs(tokens []Token) []uint32 {
	result := make([]uint32, len(tokens))
	for i, token := range tokens {
		result[i] = token.ID
	}
	return result
}

func tokenTypeIDs(tokens []Token) []uint32 {
	result := make([]uint32, len(tokens))
	for i, token := range tokens {
		result[i] = token.TypeID
	}
	return result
}

func FuzzTemplateParser(f *testing.F) {
	f.Add("[CLS] $A [SEP]", "[CLS] $A [SEP] $B:1 [SEP]:1")
	f.Add("$0", "$A:0 $B:1")
	f.Add("X:7 $A:3 X", "X $B:4 $A")
	f.Fuzz(func(t *testing.T, single, pair string) {
		processor, err := NewTemplate(single, pair, []SpecialToken{
			{Identifier: "[CLS]", IDs: []uint32{10}, Tokens: []string{"[CLS]"}},
			{Identifier: "[SEP]", IDs: []uint32{11}, Tokens: []string{"[SEP]"}},
			{Identifier: "X", IDs: []uint32{12, 13}, Tokens: []string{"x", "y"}},
		})
		if err != nil {
			return
		}
		got := processor.Process(
			[]Token{{ID: 1, Value: "a", Start: 0, End: 1}},
			[]Token{{ID: 2, Value: "b", Start: 0, End: 1}},
			true,
			true,
		)
		for _, token := range got {
			if token.Start < 0 || token.End < token.Start {
				t.Fatalf("invalid token offsets: %#v", token)
			}
		}
	})
}
