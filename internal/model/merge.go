package model

const noPair = ^uint32(0)

func (m *Model) mergeAll(symbols []symbol, queue *candidateHeap) {
	if len(symbols) < 2 {
		return
	}
	if len(symbols) == 2 {
		// There is only one possible merge. No priority structure is needed,
		// and the rank need not fit the packed array key.
		if rule, ok := m.merges[pair{left: symbols[0].id, right: symbols[1].id}]; ok {
			symbols[0].id = rule.newID
			symbols[0].length += symbols[1].length
			symbols[0].next = -1
			symbols[1].alive = false
		}
		return
	}
	// Ranks are contiguous construction-time ordinals. Conservatively keep
	// every packed (rank << 5 | original position) below the sentinel.
	if len(symbols) > 32 || len(m.merges) >= 1<<27-1 {
		m.mergeHeap(symbols, queue)
		return
	}
	m.mergeArray(symbols)
}

func (m *Model) mergeArray(symbols []symbol) {
	var priorities, newIDs [32]uint32
	key := noPair
	for i := 0; i < len(symbols)-1; i++ {
		m.updateArrayPair(symbols, &priorities, &newIDs, i)
		key = min(key, priorities[i])
	}
	priorities[len(symbols)-1] = noPair
	// The initial scan also finds the first minimum: no-merge pieces return
	// without another scan. Slots remain fixed until the piece is complete.
	for {
		if key == noPair {
			return
		}
		leftIndex := int(key & 31)
		left := &symbols[leftIndex]
		rightIndex := left.next
		right := &symbols[rightIndex]
		left.id = newIDs[leftIndex]
		left.length += right.length
		left.next = right.next
		right.alive = false
		priorities[rightIndex] = noPair
		if right.next >= 0 {
			symbols[right.next].prev = leftIndex
		}
		// Only these two surviving left endpoints can have changed pairs.
		m.updateArrayPair(symbols, &priorities, &newIDs, leftIndex)
		if left.prev >= 0 {
			m.updateArrayPair(symbols, &priorities, &newIDs, left.prev)
		}
		key = minimumPair(&priorities, len(symbols))
	}
}

func (m *Model) updateArrayPair(symbols []symbol, priorities, newIDs *[32]uint32, left int) {
	priorities[left] = noPair
	right := symbols[left].next
	if right < 0 {
		return
	}
	if rule, ok := m.merges[pair{left: symbols[left].id, right: symbols[right].id}]; ok {
		priorities[left] = rule.rank<<5 | uint32(left)
		newIDs[left] = rule.newID
	}
}

func minimumPair(priorities *[32]uint32, n int) uint32 {
	best := noPair
	for i := 0; i < n; i++ {
		best = min(best, priorities[i])
	}
	return best
}
