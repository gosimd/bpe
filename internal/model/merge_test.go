package model

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

func mergeTestModel(t testing.TB) *Model {
	t.Helper()
	vocab := map[string]uint32{"x": 0}
	var rules []Merge
	var grow func(string, int)
	grow = func(s string, depth int) {
		if s != "" {
			vocab[s] = uint32(len(vocab))
			for i := range s {
				if i > 0 {
					rules = append(rules, Merge{Left: s[:i], Right: s[i:]})
				}
			}
		}
		if depth < 4 {
			for _, r := range "abé" {
				grow(s+string(r), depth+1)
			}
		}
	}
	grow("", 0)
	rng := rand.New(rand.NewPCG(12, 34))
	rng.Shuffle(len(rules), func(i, j int) { rules[i], rules[j] = rules[j], rules[i] })
	m, err := New(vocab, rules, Options{})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func mergeSurvivors(symbols []symbol) []Token {
	var result []Token
	if len(symbols) == 0 {
		return result
	}
	offset := 0
	for i := 0; i >= 0; i = symbols[i].next {
		s := symbols[i]
		result = append(result, Token{ID: s.id, Start: offset, End: offset + s.length})
		offset += s.length
	}
	return result
}

func checkMergeAgainstHeap(t testing.TB, m *Model, piece string) {
	t.Helper()
	a := m.initialSymbols(nil, piece)
	b := slices.Clone(a)
	var heap candidateHeap
	m.mergeHeap(a, &heap)
	m.mergeAll(b, &heap)
	if got, want := mergeSurvivors(b), mergeSurvivors(a); !slices.Equal(got, want) {
		t.Fatalf("piece %q: array=%v heap=%v", piece, got, want)
	}
}

func TestMergeBoundaries(t *testing.T) {
	m := mergeTestModel(t)
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128} {
		for _, pattern := range []string{"a", "x", "ab", "é", "abé"} {
			runes := []rune(strings.Repeat(pattern, n))
			checkMergeAgainstHeap(t, m, string(runes[:n]))
		}
	}
	// Different merge orders create ties, replaced candidates, and merged
	// endpoints whose IDs change while their slot indices stay the same.
	rng := rand.New(rand.NewPCG(56, 78))
	for range 3000 {
		var input strings.Builder
		for n := rng.IntN(70); n > 0; n-- {
			input.WriteRune([]rune("abéx?")[rng.IntN(5)])
		}
		checkMergeAgainstHeap(t, m, input.String())
	}
}

func TestMergeShortPieces(t *testing.T) {
	m, err := New(map[string]uint32{"a": 0, "b": 1, "é": 2, "ab": 3, "éa": 4},
		[]Merge{{Left: "a", Right: "b"}, {Left: "é", Right: "a"}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	// The direct two-symbol path does not pack ranks, including large ones.
	rule := m.merges[pair{left: 2, right: 0}]
	rule.rank = ^uint32(0)
	m.merges[pair{left: 2, right: 0}] = rule
	for _, tc := range []struct {
		piece string
		want  []Token
	}{
		{"", nil},
		{"é", []Token{{ID: 2, Start: 0, End: 2}}},
		{"ab", []Token{{ID: 3, Start: 0, End: 2}}},
		{"ba", []Token{{ID: 1, Start: 0, End: 1}, {ID: 0, Start: 1, End: 2}}},
		{"éa", []Token{{ID: 4, Start: 0, End: 3}}},
	} {
		t.Run(tc.piece, func(t *testing.T) {
			symbols := m.initialSymbols(nil, tc.piece)
			var queue candidateHeap
			m.mergeAll(symbols, &queue)
			if cap(queue) != 0 {
				t.Fatal("short path allocated heap candidates")
			}
			if got := mergeSurvivors(symbols); !slices.Equal(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			if got := m.Tokenize(tc.piece); !slices.Equal(got, tc.want) {
				t.Fatalf("Tokenize got %v, want %v", got, tc.want)
			}
		})
	}
}

func FuzzMergeMatchesHeap(f *testing.F) {
	m := mergeTestModel(f)
	for _, s := range []string{"", "abababa", "aaaaaa", "ééabé", strings.Repeat("ab", 17)} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, piece string) {
		if len(piece) > 512 || !utf8.ValidString(piece) {
			return
		}
		checkMergeAgainstHeap(t, m, piece)
	})
}

// Each op restores the same initial symbols into reusable storage, then merges.
func BenchmarkMergeShapes(b *testing.B) {
	m := mergeTestModel(b)
	for _, n := range []int{1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64} {
		for _, pattern := range []string{"ab", "x"} {
			b.Run(fmt.Sprintf("%s/%d", pattern, n), func(b *testing.B) {
				initial := m.initialSymbols(nil, strings.Repeat(pattern, n)[:n])
				symbols := make([]symbol, len(initial))
				queue := make(candidateHeap, 0, 2*n)
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					copy(symbols, initial)
					m.mergeAll(symbols, &queue)
				}
			})
		}
	}
}
