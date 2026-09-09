package model

import (
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

// Scratch owns temporary model-tokenization storage. It contains no model
// references and can be reused sequentially with different models and pieces.
// A Scratch must not be used concurrently.
type Scratch struct {
	symbols    []symbol
	candidates candidateHeap
}

func candidateLess(left, right candidate) bool {
	if left.rank != right.rank {
		return left.rank < right.rank
	}
	return left.left < right.left
}

func (h *candidateHeap) push(value candidate) {
	*h = append(*h, value)
	child := len(*h) - 1
	for child > 0 {
		parent := (child - 1) / 2
		if !candidateLess((*h)[child], (*h)[parent]) {
			break
		}
		(*h)[parent], (*h)[child] = (*h)[child], (*h)[parent]
		child = parent
	}
}

func (h *candidateHeap) pop() candidate {
	last := len(*h) - 1
	value := (*h)[0]
	(*h)[0] = (*h)[last]
	*h = (*h)[:last]

	parent := 0
	for {
		left := parent*2 + 1
		if left >= len(*h) {
			break
		}
		child := left
		right := left + 1
		if right < len(*h) && candidateLess((*h)[right], (*h)[left]) {
			child = right
		}
		if !candidateLess((*h)[child], (*h)[parent]) {
			break
		}
		(*h)[parent], (*h)[child] = (*h)[child], (*h)[parent]
		parent = child
	}
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

	scratch := m.scratchPool.Get().(*Scratch)
	symbols := m.initialSymbols(scratch.symbols[:0], piece)
	scratch.symbols = symbols
	if len(symbols) == 0 {
		m.scratchPool.Put(scratch)
		return nil
	}
	m.mergeAll(symbols, &scratch.candidates)

	result := make([]Token, 0, len(symbols))
	offset := 0
	for index := 0; index >= 0; index = symbols[index].next {
		s := symbols[index]
		result = append(result, Token{ID: s.id, Start: offset, End: offset + s.length})
		offset += s.length
	}
	m.scratchPool.Put(scratch)
	return result
}

// AppendIDs appends IDs using scratch borrowed from the model's concurrent
// pool. The returned slice never aliases scratch storage.
func (m *Model) AppendIDs(dst []uint32, piece string) []uint32 {
	if piece == "" {
		return dst
	}
	if m.options.IgnoreMerges {
		if id, ok := m.vocab[piece]; ok {
			return append(dst, id)
		}
	}
	scratch := m.scratchPool.Get().(*Scratch)
	dst = m.AppendIDsWithScratch(dst, piece, scratch)
	m.scratchPool.Put(scratch)
	return dst
}

// AppendIDsWithScratch appends IDs without constructing Token results. It is
// used by the raw-text tokenizer to reuse one allocation across many pieces.
func (m *Model) AppendIDsWithScratch(dst []uint32, piece string, scratch *Scratch) []uint32 {
	if piece == "" {
		return dst
	}
	if m.options.IgnoreMerges {
		if id, ok := m.vocab[piece]; ok {
			return append(dst, id)
		}
	}

	symbols := m.initialSymbols(scratch.symbols[:0], piece)
	scratch.symbols = symbols
	if len(symbols) == 0 {
		return dst
	}
	// The ID-only path retains the heap and its reusable candidate storage.
	m.mergeHeap(symbols, &scratch.candidates)
	for index := 0; index >= 0; index = symbols[index].next {
		dst = append(dst, symbols[index].id)
	}
	return dst
}

func (m *Model) initialSymbols(symbols []symbol, piece string) []symbol {
	if additional := utf8.RuneCountInString(piece) - cap(symbols); additional > 0 {
		symbols = make([]symbol, 0, cap(symbols)+additional)
	}
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
			fallback := true
			for index := 0; index < len(value); index++ {
				if !m.byteFallbackPresent[value[index]] {
					fallback = false
					break
				}
			}
			if fallback {
				// Upstream associates one byte of offset with each fallback token.
				for index := 0; index < len(value); index++ {
					appendSymbol(m.byteFallbackIDs[value[index]], 1)
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

func (m *Model) mergeHeap(symbols []symbol, queue *candidateHeap) {
	*queue = (*queue)[:0]
	if additional := len(symbols) - cap(*queue); additional > 0 {
		*queue = make(candidateHeap, 0, cap(*queue)+additional)
	}
	pushPair := func(left int) {
		right := symbols[left].next
		if right < 0 {
			return
		}
		if rule, ok := m.merges[pair{left: symbols[left].id, right: symbols[right].id}]; ok {
			queue.push(candidate{rank: rule.rank, left: left, right: right, newID: rule.newID})
		}
	}
	for index := range symbols {
		pushPair(index)
	}
	for len(*queue) > 0 {
		candidate := queue.pop()
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
