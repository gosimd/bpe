package model

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"testing"
)

// WorkloadAnalyzer is test-only. Generated methods are copies of the current
// production methods with merge-map reads replaced by counted reads. Nothing
// here is linked into the library, and these methods must never be timed.
type WorkloadAnalyzer struct {
	*Model
	queries map[pair]uint64
	sites   map[string]*LookupCount
	groups  map[uint32]int
	Report  WorkloadReport
}

type LookupCount struct {
	Queries uint64 `json:"queries"`
	Hits    uint64 `json:"hits"`
}

type Distribution struct {
	Count     uint64         `json:"count"`
	Sum       uint64         `json:"sum"`
	Mean      float64        `json:"mean"`
	P50       int            `json:"p50"`
	P90       int            `json:"p90"`
	P99       int            `json:"p99"`
	Max       int            `json:"max"`
	Histogram map[int]uint64 `json:"histogram"`
}

func (d *Distribution) add(n int, count uint64) {
	if d.Histogram == nil {
		d.Histogram = make(map[int]uint64)
	}
	d.Histogram[n] += count
	d.Count += count
	d.Sum += uint64(n) * count
}

func (d *Distribution) finish() {
	if d.Count == 0 {
		return
	}
	d.Mean = float64(d.Sum) / float64(d.Count)
	keys := make([]int, 0, len(d.Histogram))
	for n := range d.Histogram {
		keys = append(keys, n)
	}
	slices.Sort(keys)
	d.Max = keys[len(keys)-1]
	quantile := func(p float64) int {
		threshold := uint64(math.Ceil(p * float64(d.Count)))
		var seen uint64
		for _, n := range keys {
			seen += d.Histogram[n]
			if seen >= threshold {
				return n
			}
		}
		panic("unreachable quantile")
	}
	d.P50, d.P90, d.P99 = quantile(.5), quantile(.9), quantile(.99)
}

type PairFrequency struct {
	Left    uint32 `json:"left_id"`
	Right   uint32 `json:"right_id"`
	Queries uint64 `json:"queries"`
	Hit     bool   `json:"hit"`
}

type WorkloadReport struct {
	VocabEntries           int                     `json:"vocab_entries"`
	MergeRules             int                     `json:"merge_rules"`
	MinID                  uint32                  `json:"min_id"`
	MaxID                  uint32                  `json:"max_id"`
	MaxRank                uint32                  `json:"max_rank"`
	MergeOperandMaxID      uint32                  `json:"merge_operand_max_id"`
	RulesPerLeft           Distribution            `json:"rules_per_left"`
	Pieces                 uint64                  `json:"pieces"`
	PieceUTF8Bytes         Distribution            `json:"piece_utf8_bytes"`
	InitialSymbols         Distribution            `json:"initial_symbols"`
	FinalTokens            Distribution            `json:"final_tokens"`
	Merges                 Distribution            `json:"merges_per_piece"`
	NoMergePieces          uint64                  `json:"no_merge_pieces"`
	BypassPieces           uint64                  `json:"ignore_merges_bypass_pieces"`
	Paths                  map[string]uint64       `json:"paths"`
	Lookup                 LookupCount             `json:"lookup"`
	LookupSites            map[string]*LookupCount `json:"lookup_sites"`
	QueriesPerPiece        Distribution            `json:"queries_per_piece"`
	UniquePairs            uint64                  `json:"unique_queried_pairs"`
	UniqueHitPairs         uint64                  `json:"unique_hit_pairs"`
	PairFrequencies        Distribution            `json:"pair_query_frequencies"`
	QueriedLeftRuleCounts  Distribution            `json:"query_weighted_rules_per_left"`
	TopPairs               []PairFrequency         `json:"top_pairs"`
	Top1PercentQueryShare  float64                 `json:"top_1_percent_pair_query_share"`
	Top10PercentQueryShare float64                 `json:"top_10_percent_pair_query_share"`
}

func NewWorkloadAnalyzer(m *Model) *WorkloadAnalyzer {
	a := &WorkloadAnalyzer{Model: m, queries: make(map[pair]uint64), sites: make(map[string]*LookupCount), groups: make(map[uint32]int)}
	r := &a.Report
	r.VocabEntries, r.MergeRules, r.MinID = len(m.vocab), len(m.merges), ^uint32(0)
	r.Paths = make(map[string]uint64)
	for _, id := range m.vocab {
		r.MinID = min(r.MinID, id)
		r.MaxID = max(r.MaxID, id)
	}
	if len(m.vocab) == 0 {
		r.MinID = 0
	}
	for p, rule := range m.merges {
		a.groups[p.left]++
		r.MaxRank = max(r.MaxRank, rule.rank)
		r.MergeOperandMaxID = max(r.MergeOperandMaxID, p.left, p.right)
	}
	for _, n := range a.groups {
		r.RulesPerLeft.add(n, 1)
	}
	return a
}

func (m *WorkloadAnalyzer) lookup(p pair, site string) (merge, bool) {
	rule, ok := m.merges[p]
	m.queries[p]++
	c := m.sites[site]
	if c == nil {
		c = new(LookupCount)
		m.sites[site] = c
	}
	c.Queries++
	m.Report.Lookup.Queries++
	if ok {
		c.Hits++
		m.Report.Lookup.Hits++
	}
	return rule, ok
}

// Analyze checks the traced output against the unmodified implementation, then
// records one piece. The caller also checks the returned output against HF.
func (m *WorkloadAnalyzer) Analyze(piece string) ([]Token, error) {
	initial := len(m.initialSymbols(nil, piece))
	before := m.Report.Lookup.Queries
	got := m.Tokenize(piece)
	want := m.Model.Tokenize(piece)
	if !slices.Equal(got, want) {
		return nil, fmt.Errorf("instrumented output differs from production for %q", piece)
	}
	r := &m.Report
	r.Pieces++
	r.PieceUTF8Bytes.add(len(piece), 1)
	r.InitialSymbols.add(initial, 1)
	r.FinalTokens.add(len(got), 1)
	r.QueriesPerPiece.add(int(r.Lookup.Queries-before), 1)
	merges := initial - len(got)
	_, bypass := m.vocab[piece]
	bypass = piece != "" && m.options.IgnoreMerges && bypass
	path := "trivial"
	switch {
	case bypass:
		path = "ignore_merges"
		r.BypassPieces++
		merges = 0
	case initial == 2:
		path = "direct_two"
	case initial > 32 || (initial > 2 && len(m.merges) >= 1<<27-1):
		path = "heap"
	case initial > 2:
		path = "array"
	}
	r.Paths[path]++
	if merges < 0 {
		return nil, fmt.Errorf("negative merge count for %q", piece)
	}
	r.Merges.add(merges, 1)
	if merges == 0 {
		r.NoMergePieces++
	}
	return got, nil
}

func (m *WorkloadAnalyzer) Finish() WorkloadReport {
	r := m.Report
	r.LookupSites = m.sites
	r.UniquePairs = uint64(len(m.queries))
	pairs := make([]PairFrequency, 0, len(m.queries))
	for p, n := range m.queries {
		_, hit := m.merges[p]
		if hit {
			r.UniqueHitPairs++
		}
		r.PairFrequencies.add(int(n), 1)
		r.QueriedLeftRuleCounts.add(m.groups[p.left], n)
		pairs = append(pairs, PairFrequency{p.left, p.right, n, hit})
	}
	slices.SortFunc(pairs, func(a, b PairFrequency) int {
		if order := cmp.Compare(b.Queries, a.Queries); order != 0 {
			return order
		}
		if order := cmp.Compare(a.Left, b.Left); order != 0 {
			return order
		}
		return cmp.Compare(a.Right, b.Right)
	})
	share := func(percent float64) float64 {
		if r.Lookup.Queries == 0 {
			return 0
		}
		var n uint64
		for _, p := range pairs[:int(math.Ceil(percent*float64(len(pairs))))] {
			n += p.Queries
		}
		return float64(n) / float64(r.Lookup.Queries)
	}
	r.Top1PercentQueryShare, r.Top10PercentQueryShare = share(.01), share(.1)
	r.TopPairs = pairs[:min(20, len(pairs))]
	for _, d := range []*Distribution{&r.RulesPerLeft, &r.PieceUTF8Bytes, &r.InitialSymbols, &r.FinalTokens, &r.Merges, &r.QueriesPerPiece, &r.PairFrequencies, &r.QueriedLeftRuleCounts} {
		d.finish()
	}
	return r
}

func TestWorkloadAnalyzer(t *testing.T) {
	m, err := New(map[string]uint32{"a": 0, "b": 1, "ab": 2, "abab": 3, "x": 4}, []Merge{{"a", "b"}, {"ab", "ab"}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	a := NewWorkloadAnalyzer(m)
	for _, piece := range []string{"", "a", "ab", "ax", "abab", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"} {
		if _, err := a.Analyze(piece); err != nil {
			t.Fatal(err)
		}
	}
	r := a.Finish()
	// ab and ax: one lookup each; abab: three initial + two updated
	// pairs; the 33-symbol no-merge heap piece: 32 initial lookups.
	if r.Pieces != 6 || r.Merges.Sum != 4 || r.Lookup.Queries != 39 || r.Lookup.Hits != 4 || r.UniquePairs != 6 {
		t.Fatalf("unexpected trace: %+v", r)
	}
	if r.Paths["heap"] != 1 || r.Paths["array"] != 1 || r.Paths["direct_two"] != 2 || r.Paths["trivial"] != 2 {
		t.Fatal(r.Paths)
	}
	if r.InitialSymbols.P50 != 2 || r.InitialSymbols.Max != 33 {
		t.Fatal(r.InitialSymbols)
	}
}

func TestWorkloadAnalyzerSpecialPaths(t *testing.T) {
	for _, options := range []Options{{}, {IgnoreMerges: true}, {UnknownToken: "x", FuseUnknown: true}} {
		m, err := New(map[string]uint32{"a": 0, "b": 1, "ab": 2, "x": 3}, []Merge{{"a", "b"}}, options)
		if err != nil {
			t.Fatal(err)
		}
		a := NewWorkloadAnalyzer(m)
		for _, piece := range []string{"ab", "ababababababababababababababababab", "??a??"} {
			if _, err := a.Analyze(piece); err != nil {
				t.Fatal(err)
			}
		}
		r := a.Finish()
		if r.LookupSites["mergeHeap.2"] == nil || r.LookupSites["mergeHeap.2"].Hits == 0 {
			t.Fatal("heap revalidation not traced")
		}
		if options.IgnoreMerges && r.BypassPieces != 1 {
			t.Fatal("ignore-merges shortcut not recorded")
		}
		if r.QueriesPerPiece.Sum != r.Lookup.Queries {
			t.Fatal("query counts do not reconcile")
		}
	}
}
