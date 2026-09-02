package bpe_test

import (
	"errors"
	"reflect"
	"testing"

	"github.com/gosimd/bpe"
)

func TestTokenizerDefaultAddsPrefixSpace(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"ĠHello": 1}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizer(model)
	if err != nil {
		t.Fatal(err)
	}

	encoding, err := tokenizer.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{1}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, want)
	}
	if want := []bpe.Offset{{Start: 0, End: 5}}; !reflect.DeepEqual(encoding.Offsets, want) {
		t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, want)
	}
}

func TestTokenizerByteLevelSplittingAndOffsets(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{
			"Hello":  1,
			",":      2,
			"Ġworld": 3,
			"!":      4,
			"a":      5,
			"Ġ":      6,
			"Ġb":     7,
			"it":     8,
			"'s":     9,
		}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		input       string
		wantIDs     []uint32
		wantOffsets []bpe.Offset
	}{
		{
			input:       "Hello, world!",
			wantIDs:     []uint32{1, 2, 3, 4},
			wantOffsets: []bpe.Offset{{0, 5}, {5, 6}, {6, 12}, {12, 13}},
		},
		{
			input:       "a  b",
			wantIDs:     []uint32{5, 6, 7},
			wantOffsets: []bpe.Offset{{0, 1}, {1, 2}, {2, 4}},
		},
		{
			input:       "it's",
			wantIDs:     []uint32{8, 9},
			wantOffsets: []bpe.Offset{{0, 2}, {2, 4}},
		},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			encoding, err := tokenizer.Encode(test.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(encoding.IDs, test.wantIDs) {
				t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, test.wantIDs)
			}
			if !reflect.DeepEqual(encoding.Offsets, test.wantOffsets) {
				t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, test.wantOffsets)
			}
		})
	}
}

func TestTokenizerMapsOffsetsBackFromByteLevelUTF8(t *testing.T) {
	model := mustModel(t,
		map[string]uint32{"Ã©": 1}, nil,
		bpe.ModelOptions{IgnoreMerges: true},
	)
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{})
	if err != nil {
		t.Fatal(err)
	}

	encoding, err := tokenizer.Encode("é")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{1}; !reflect.DeepEqual(encoding.IDs, want) {
		t.Fatalf("Encode() IDs = %v, want %v", encoding.IDs, want)
	}
	if want := []bpe.Offset{{Start: 0, End: 2}}; !reflect.DeepEqual(encoding.Offsets, want) {
		t.Fatalf("Encode() offsets = %#v, want %#v", encoding.Offsets, want)
	}

	ids, err := tokenizer.AppendIDs([]uint32{99}, "é")
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{99, 1}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("AppendIDs() = %v, want %v", ids, want)
	}
}

func TestTokenizerValidation(t *testing.T) {
	if _, err := bpe.NewTokenizer(nil); !errors.Is(err, bpe.ErrNilModel) {
		t.Fatalf("NewTokenizer(nil) error = %v, want ErrNilModel", err)
	}
	if _, err := bpe.NewTokenizer(&bpe.Model{}); !errors.Is(err, bpe.ErrNilModel) {
		t.Fatalf("NewTokenizer(zero Model) error = %v, want ErrNilModel", err)
	}

	model := mustModel(t, map[string]uint32{"a": 1}, nil, bpe.ModelOptions{})
	tokenizer, err := bpe.NewTokenizer(model)
	if err != nil {
		t.Fatal(err)
	}
	invalid := string([]byte{0xff})
	if _, err := tokenizer.Encode(invalid); !errors.Is(err, bpe.ErrInvalidUTF8) {
		t.Fatalf("Encode() error = %v, want ErrInvalidUTF8", err)
	}
	dst := []uint32{9}
	got, err := tokenizer.AppendIDs(dst, invalid)
	if !errors.Is(err, bpe.ErrInvalidUTF8) {
		t.Fatalf("AppendIDs() error = %v, want ErrInvalidUTF8", err)
	}
	if !reflect.DeepEqual(got, dst) {
		t.Fatalf("AppendIDs() = %v, want unchanged %v", got, dst)
	}
}
