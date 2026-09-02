package model

import (
	"math/rand"
	"sort"
	"testing"
)

func TestAppendIDsWithScratchReusesWorkingBuffers(t *testing.T) {
	compiled, err := New(
		map[string]uint32{"a": 0, "b": 1, "ab": 2},
		[]Merge{{Left: "a", Right: "b"}},
		Options{},
	)
	if err != nil {
		t.Fatal(err)
	}
	var scratch Scratch
	dst := make([]uint32, 0, 4)
	dst = compiled.AppendIDsWithScratch(dst, "abababab", &scratch)
	if len(scratch.symbols) == 0 || cap(scratch.candidates) == 0 {
		t.Fatalf("scratch was not populated: %#v", scratch)
	}
	symbols := &scratch.symbols[:cap(scratch.symbols)][0]
	candidates := &scratch.candidates[:cap(scratch.candidates)][0]

	dst = compiled.AppendIDsWithScratch(dst[:0], "abababab", &scratch)
	if got := &scratch.symbols[:cap(scratch.symbols)][0]; got != symbols {
		t.Fatal("symbol storage was not reused")
	}
	if got := &scratch.candidates[:cap(scratch.candidates)][0]; got != candidates {
		t.Fatal("candidate storage was not reused")
	}
}

func TestCandidateHeapOrdersRankThenPosition(t *testing.T) {
	random := rand.New(rand.NewSource(1))
	values := make([]candidate, 1000)
	var queue candidateHeap
	for index := range values {
		values[index] = candidate{
			rank:  uint32(random.Intn(20)),
			left:  random.Intn(100),
			right: index,
		}
		queue.push(values[index])
	}
	sort.SliceStable(values, func(i, j int) bool {
		return candidateLess(values[i], values[j])
	})
	for index, want := range values {
		got := queue.pop()
		if got.rank != want.rank || got.left != want.left {
			t.Fatalf("pop %d = (rank %d, left %d), want (rank %d, left %d)", index, got.rank, got.left, want.rank, want.left)
		}
	}
}
