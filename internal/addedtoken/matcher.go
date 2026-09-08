// Package addedtoken extracts configured AddedTokens before normalization and
// pre-tokenization. Its behavior follows Hugging Face Tokenizers v0.23.2.
package addedtoken

import (
	"unicode"
	"unicode/utf8"
)

// Definition is one validated added token.
type Definition struct {
	ID         uint32
	Content    string
	SingleWord bool
	LStrip     bool
	RStrip     bool
	Normalized bool
}

// Segment covers one half-open byte range in the original input. Added is true
// when ID must be emitted directly; otherwise the segment continues through the
// tokenizer pipeline.
type Segment struct {
	Start int
	End   int
	ID    uint32
	Added bool
}

type trieNode struct {
	next     map[byte]int
	terminal int
}

type trie struct {
	nodes  []trieNode
	tokens []Definition
}

// Matcher is immutable and safe for concurrent use.
type Matcher struct {
	raw        trie
	normalized trie
}

// Empty reports whether the matcher has no definitions.
func (m *Matcher) Empty() bool {
	return len(m.raw.tokens) == 0 && len(m.normalized.tokens) == 0
}

// New compiles definitions into the two matching phases used by Hugging Face.
func New(definitions []Definition) *Matcher {
	var raw, normalized []Definition
	for _, definition := range definitions {
		if definition.Normalized {
			normalized = append(normalized, definition)
		} else {
			raw = append(raw, definition)
		}
	}
	return &Matcher{
		raw:        newTrie(raw),
		normalized: newTrie(normalized),
	}
}

func newTrie(tokens []Definition) trie {
	compiled := trie{
		nodes:  []trieNode{{terminal: -1}},
		tokens: append([]Definition(nil), tokens...),
	}
	for index, token := range compiled.tokens {
		node := 0
		for _, b := range []byte(token.Content) {
			if compiled.nodes[node].next == nil {
				compiled.nodes[node].next = make(map[byte]int)
			}
			next, ok := compiled.nodes[node].next[b]
			if !ok {
				next = len(compiled.nodes)
				compiled.nodes[node].next[b] = next
				compiled.nodes = append(compiled.nodes, trieNode{terminal: -1})
			}
			node = next
		}
		compiled.nodes[node].terminal = index
	}
	return compiled
}

// Split extracts non-normalized tokens first, then normalized tokens from each
// remaining segment. There is currently no normalizer, so the second phase sees
// unchanged text while retaining Hugging Face's precedence rules.
func (m *Matcher) Split(text string) []Segment {
	segments, _ := m.SplitWithScratch(text, nil, nil)
	return segments
}

// SplitWithScratch is Split with two caller-owned alternating buffers. The
// returned spare buffer may be reused as the second argument on the next call.
func (m *Matcher) SplitWithScratch(text string, first, second []Segment) (segments, spare []Segment) {
	first = append(first[:0], Segment{Start: 0, End: len(text)})
	first, second = m.splitSegments(text, first, second[:0], &m.raw)
	first, second = m.splitSegments(text, first, second[:0], &m.normalized)
	return first, second
}

func (m *Matcher) splitSegments(text string, input, output []Segment, compiled *trie) (segments, spare []Segment) {
	if len(compiled.tokens) == 0 {
		return input, output
	}
	for _, segment := range input {
		if segment.Added || segment.Start == segment.End {
			output = append(output, segment)
			continue
		}
		output = compiled.appendSplit(output, text[segment.Start:segment.End], segment.Start)
	}
	return output, input[:0]
}

func (t *trie) split(text string) []Segment {
	return t.appendSplit(nil, text, 0)
}

func (t *trie) appendSplit(dst []Segment, text string, base int) []Segment {
	if text == "" {
		return append(dst, Segment{Start: base, End: base})
	}

	acceptedEnd := 0
	searchFrom := 0
	for searchFrom < len(text) {
		start, end, tokenIndex := t.findLeftmostLongest(text, searchFrom)
		if tokenIndex < 0 {
			break
		}
		// Hugging Face's non-overlapping automaton advances even when the
		// selected leftmost-longest candidate fails single_word validation.
		searchFrom = end
		token := t.tokens[tokenIndex]
		if token.SingleWord && !singleWordBoundary(text, start, end) {
			continue
		}

		if token.LStrip {
			start = max(stripWhitespaceLeft(text, start), acceptedEnd)
		}
		if token.RStrip {
			end = stripWhitespaceRight(text, end)
		}
		if start < acceptedEnd {
			continue
		}
		if acceptedEnd < start {
			dst = append(dst, Segment{Start: base + acceptedEnd, End: base + start})
		}
		dst = append(dst, Segment{Start: base + start, End: base + end, ID: token.ID, Added: true})
		acceptedEnd = end
	}
	if acceptedEnd < len(text) {
		dst = append(dst, Segment{Start: base + acceptedEnd, End: base + len(text)})
	}
	return dst
}

func (t *trie) findLeftmostLongest(text string, from int) (start, end, tokenIndex int) {
	for start = from; start < len(text); start++ {
		node := 0
		bestEnd := -1
		bestToken := -1
		for position := start; position < len(text); position++ {
			next, ok := t.nodes[node].next[text[position]]
			if !ok {
				break
			}
			node = next
			if t.nodes[node].terminal >= 0 {
				bestEnd = position + 1
				bestToken = t.nodes[node].terminal
			}
		}
		if bestToken >= 0 {
			return start, bestEnd, bestToken
		}
	}
	return len(text), len(text), -1
}

func singleWordBoundary(text string, start, end int) bool {
	if start > 0 {
		r, _ := utf8.DecodeLastRuneInString(text[:start])
		if isWord(r) {
			return false
		}
	}
	if end < len(text) {
		r, _ := utf8.DecodeRuneInString(text[end:])
		if isWord(r) {
			return false
		}
	}
	return true
}

// Rust regex's Unicode \w is Alphabetic, Mark, Decimal_Number,
// Connector_Punctuation, or Join_Control.
func isWord(r rune) bool {
	return unicode.IsLetter(r) || unicode.Is(unicode.Other_Alphabetic, r) ||
		unicode.IsMark(r) || unicode.Is(unicode.Nd, r) ||
		unicode.Is(unicode.Pc, r) || unicode.Is(unicode.Join_Control, r)
}

func stripWhitespaceLeft(text string, end int) int {
	position := end
	for position > 0 {
		r, size := utf8.DecodeLastRuneInString(text[:position])
		if !unicode.Is(unicode.White_Space, r) {
			break
		}
		position -= size
	}
	return position
}

func stripWhitespaceRight(text string, start int) int {
	position := start
	for position < len(text) {
		r, size := utf8.DecodeRuneInString(text[position:])
		if !unicode.Is(unicode.White_Space, r) {
			break
		}
		position += size
	}
	return position
}
