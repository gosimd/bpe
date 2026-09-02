package model

import (
	"container/heap"
	"fmt"
	"unicode/utf8"
)

// Token is an internal token result. Start and End are byte offsets relative
// to the input piece.
type Token struct {
	ID    uint32
	Start int
	End   int
}

type symbol struct {
	id     uint32
	prev   int
	next   int
	length int
	alive  bool
}

type candidate struct {
	rank  uint32
	left  int
	right int
	newID uint32
}

type candidateHeap []candidate

func (h candidateHeap) Len() int { return len(h) }

func (h candidateHeap) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].left < h[j].left
}

func (h candidateHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *candidateHeap) Push(value any) {
	*h = append(*h, value.(candidate))
}

func (h *candidateHeap) Pop() any {
	old := *h
	last := len(old) - 1
	value := old[last]
	*h = old[:last]
	return value
}

// Tokenize applies exact rank-prioritized BPE merging to one valid UTF-8 piece.
func (m *Model) Tokenize(piece string) []Token {
	if piece == "" {
		return nil
	}
	if m.options.IgnoreMerges {
		if id, ok := m.vocab[piece]; ok {
			return []Token{{ID: id, Start: 0, End: len(piece)}}
		}
	}

	symbols := m.initialSymbols(piece)
	if len(symbols) == 0 {
		return nil
	}
	m.mergeAll(symbols)

	result := make([]Token, 0, len(symbols))
	offset := 0
	for index := 0; index >= 0; index = symbols[index].next {
		s := symbols[index]
		result = append(result, Token{ID: s.id, Start: offset, End: offset + s.length})
		offset += s.length
	}
	return result
}

// AppendIDs appends the IDs produced by Tokenize to dst.
func (m *Model) AppendIDs(dst []uint32, piece string) []uint32 {
	for _, token := range m.Tokenize(piece) {
		dst = append(dst, token.ID)
	}
	return dst
}

func (m *Model) initialSymbols(piece string) []symbol {
	symbols := make([]symbol, 0, utf8.RuneCountInString(piece))
	appendSymbol := func(id uint32, length int) {
		previous := len(symbols) - 1
		if previous >= 0 {
			symbols[previous].next = len(symbols)
		}
		symbols = append(symbols, symbol{
			id:     id,
			prev:   previous,
			next:   -1,
			length: length,
			alive:  true,
		})
	}

	var unknownID uint32
	unknownPending := false
	unknownLength := 0
	if m.options.UnknownToken != "" {
		unknownID = m.vocab[m.options.UnknownToken]
	}
	flushUnknown := func() {
		if unknownPending {
			appendSymbol(unknownID, unknownLength)
			unknownPending = false
			unknownLength = 0
		}
	}

	for start, r := range piece {
		width := utf8.RuneLen(r)
		end := start + width
		value := piece[start:end]
		if start != 0 && m.options.ContinuingSubwordPrefix != "" {
			value = m.options.ContinuingSubwordPrefix + value
		}
		if end == len(piece) && m.options.EndOfWordSuffix != "" {
			value += m.options.EndOfWordSuffix
		}

		if id, ok := m.vocab[value]; ok {
			flushUnknown()
			appendSymbol(id, width)
			continue
		}
		if m.options.ByteFallback {
			fallback, ok := m.byteFallbackIDs(value)
			if ok {
				// Upstream associates one byte of offset with each fallback token.
				for _, id := range fallback {
					appendSymbol(id, 1)
				}
				continue
			}
		}
		if m.options.UnknownToken == "" {
			continue
		}
		if unknownPending && m.options.FuseUnknown {
			unknownLength += width
			continue
		}
		flushUnknown()
		unknownPending = true
		unknownLength = width
	}
	flushUnknown()
	return symbols
}

func (m *Model) byteFallbackIDs(value string) ([]uint32, bool) {
	ids := make([]uint32, 0, len(value))
	for _, b := range []byte(value) {
		fallback := fmt.Sprintf("<0x%02X>", b)
		id, ok := m.vocab[fallback]
		if !ok {
			return nil, false
		}
		ids = append(ids, id)
	}
	return ids, true
}

func (m *Model) mergeAll(symbols []symbol) {
	queue := make(candidateHeap, 0, len(symbols))
	pushPair := func(left int) {
		right := symbols[left].next
		if right < 0 {
			return
		}
		if rule, ok := m.merges[pair{left: symbols[left].id, right: symbols[right].id}]; ok {
			heap.Push(&queue, candidate{rank: rule.rank, left: left, right: right, newID: rule.newID})
		}
	}
	for index := range symbols {
		pushPair(index)
	}
	heap.Init(&queue)

	for queue.Len() > 0 {
		candidate := heap.Pop(&queue).(candidate)
		left := &symbols[candidate.left]
		if !left.alive || left.next != candidate.right {
			continue
		}
		right := &symbols[candidate.right]
		if !right.alive {
			continue
		}
		rule, ok := m.merges[pair{left: left.id, right: right.id}]
		if !ok || rule.rank != candidate.rank || rule.newID != candidate.newID {
			continue
		}

		left.id = rule.newID
		left.length += right.length
		left.next = right.next
		right.alive = false
		if right.next >= 0 {
			symbols[right.next].prev = candidate.left
		}
		if left.prev >= 0 {
			pushPair(left.prev)
		}
		if left.next >= 0 {
			pushPair(candidate.left)
		}
	}
}
