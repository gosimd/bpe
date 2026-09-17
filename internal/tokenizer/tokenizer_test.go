package tokenizer

import (
	"reflect"
	"testing"

	"github.com/gosimd/bpe/internal/addedtoken"
	"github.com/gosimd/bpe/internal/bytelevel"
	"github.com/gosimd/bpe/internal/model"
	"github.com/gosimd/bpe/internal/postprocess"
	"github.com/gosimd/bpe/internal/pretokenize"
)

func TestAppendIDsWithScratchReusesPipelineBuffers(t *testing.T) {
	compiled, err := model.New(
		map[string]uint32{"Ġhello": 0, "Ġworld": 1, "!": 2},
		nil,
		model.Options{IgnoreMerges: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	processor, err := postprocess.NewTemplate(
		"[BOS] $A [EOS]",
		"$A $B",
		[]postprocess.SpecialToken{
			{Identifier: "[BOS]", IDs: []uint32{101}, Tokens: []string{"[BOS]"}},
			{Identifier: "[EOS]", IDs: []uint32{102}, Tokens: []string{"[EOS]"}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer := New(
		compiled,
		true,
		[]addedtoken.Definition{{ID: 100, Content: "<s>"}},
		processor,
		nil,
	)
	var scratch Scratch
	dst := make([]uint32, 0, 8)
	dst, err = tokenizer.AppendIDsWithScratch(dst, "hello<s>world!", true, &scratch)
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{101, 0, 100, 1, 2, 102}; !reflect.DeepEqual(dst, want) {
		t.Fatalf("AppendIDsWithScratch() = %v, want %v", dst, want)
	}
	// Run twice so both alternating AddedToken segment buffers have reached the
	// capacity required by this input.
	dst, err = tokenizer.AppendIDsWithScratch(dst[:0], "hello<s>world!", true, &scratch)
	if err != nil {
		t.Fatal(err)
	}

	working := firstByte(t, scratch.working, "working")
	pieces := firstPiece(t, scratch.pieces)
	mapped := firstByte(t, scratch.mapped, "mapped")
	segments := firstSegment(t, scratch.segments)
	spare := firstSegment(t, scratch.spare)
	sequenceIDs := firstID(t, scratch.sequenceIDs)

	dst, err = tokenizer.AppendIDsWithScratch(dst[:0], "hello<s>world!", true, &scratch)
	if err != nil {
		t.Fatal(err)
	}
	if firstByte(t, scratch.working, "working") != working ||
		firstPiece(t, scratch.pieces) != pieces ||
		firstByte(t, scratch.mapped, "mapped") != mapped ||
		!samePair(firstSegment(t, scratch.segments), firstSegment(t, scratch.spare), segments, spare) ||
		firstID(t, scratch.sequenceIDs) != sequenceIDs {
		t.Fatal("pipeline scratch capacity was not reused")
	}
}

func TestRegexpAppendIDsReusesBuffers(t *testing.T) {
	compiled, err := model.New(map[string]uint32{"Ġhello": 0, "Ġ1": 1, "Ġworld": 2}, nil, model.Options{IgnoreMerges: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, pattern := range []string{"[0-9]", "[0123456789]"} {
		splitter, err := pretokenize.NewRegexp(pattern)
		if err != nil {
			t.Fatal(err)
		}
		tokenizer := New(compiled, true, nil, nil, splitter)
		var scratch Scratch
		dst, err := tokenizer.AppendIDsWithScratch(make([]uint32, 0, 3), "hello1world", false, &scratch)
		if err != nil {
			t.Fatal(err)
		}
		working, mapped := firstByte(t, scratch.working, "working"), firstByte(t, scratch.mapped, "mapped")
		for range 3 {
			dst, err = tokenizer.AppendIDsWithScratch(dst[:0], "hello1world", false, &scratch)
			if err != nil || !reflect.DeepEqual(dst, []uint32{0, 1, 2}) || working != firstByte(t, scratch.working, "working") || mapped != firstByte(t, scratch.mapped, "mapped") {
				t.Fatalf("scratch not reused: %v, %v", dst, err)
			}
		}
		if len(scratch.pieces) != 0 {
			t.Fatal("regexp path must not run GPT-2 splitting")
		}
	}
}

func samePair[T comparable](left, right, wantLeft, wantRight T) bool {
	return left == wantLeft && right == wantRight || left == wantRight && right == wantLeft
}

func firstByte(t *testing.T, values []byte, name string) *byte {
	t.Helper()
	if cap(values) == 0 {
		t.Fatalf("%s scratch is empty", name)
	}
	return &values[:cap(values)][0]
}

func firstPiece(t *testing.T, values []bytelevel.Piece) *bytelevel.Piece {
	t.Helper()
	if cap(values) == 0 {
		t.Fatal("piece scratch is empty")
	}
	return &values[:cap(values)][0]
}

func firstSegment(t *testing.T, values []addedtoken.Segment) *addedtoken.Segment {
	t.Helper()
	if cap(values) == 0 {
		t.Fatal("segment scratch is empty")
	}
	return &values[:cap(values)][0]
}

func firstID(t *testing.T, values []uint32) *uint32 {
	t.Helper()
	if cap(values) == 0 {
		t.Fatal("ID scratch is empty")
	}
	return &values[:cap(values)][0]
}
