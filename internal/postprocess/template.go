package postprocess

import (
	"fmt"
	"strconv"
	"strings"
)

type sequence uint8

const (
	sequenceA sequence = iota
	sequenceB
)

type piece struct {
	sequence *sequence
	special  string
	typeID   uint32
}

// SpecialToken is an immutable template special-token definition.
type SpecialToken struct {
	Identifier string
	IDs        []uint32
	Tokens     []string
}

// Template implements Hugging Face TemplateProcessing.
type Template struct {
	single   []piece
	pair     []piece
	specials map[string]SpecialToken
}

func NewTemplate(single, pair string, specialTokens []SpecialToken) (*Template, error) {
	if single == "" {
		single = "$A"
	}
	if pair == "" {
		pair = "$A:0 $B:1"
	}

	parsedSingle, err := parseTemplate(single)
	if err != nil {
		return nil, fmt.Errorf("single template: %w", err)
	}
	parsedPair, err := parseTemplate(pair)
	if err != nil {
		return nil, fmt.Errorf("pair template: %w", err)
	}
	return newTemplate(parsedSingle, parsedPair, specialTokens)
}

// TemplatePiece represents a serialized template entry without round-tripping
// identifiers through the string shorthand (identifiers can contain : or $).
type TemplatePiece struct {
	SequenceID int // NoSequence for a special token, otherwise 0 or 1.
	Identifier string
	TypeID     uint32
}

func NewTemplatePieces(single, pair []TemplatePiece, specials []SpecialToken) (*Template, error) {
	convert := func(items []TemplatePiece) ([]piece, error) {
		out := make([]piece, len(items))
		for i, item := range items {
			out[i] = piece{special: item.Identifier, typeID: item.TypeID}
			if item.SequenceID != NoSequence {
				if item.SequenceID != 0 && item.SequenceID != 1 {
					return nil, fmt.Errorf("invalid sequence %d", item.SequenceID)
				}
				s := sequence(item.SequenceID)
				out[i].sequence = &s
			}
		}
		return out, nil
	}
	a, err := convert(single)
	if err != nil {
		return nil, err
	}
	b, err := convert(pair)
	if err != nil {
		return nil, err
	}
	return newTemplate(a, b, specials)
}

func newTemplate(parsedSingle, parsedPair []piece, specialTokens []SpecialToken) (*Template, error) {
	for _, item := range parsedSingle {
		if item.sequence != nil && *item.sequence == sequenceB {
			return nil, fmt.Errorf("single template cannot reference sequence B")
		}
	}
	hasA, hasB := false, false
	for _, item := range parsedPair {
		if item.sequence == nil {
			continue
		}
		if *item.sequence == sequenceA {
			hasA = true
		} else {
			hasB = true
		}
	}
	if !hasA || !hasB {
		return nil, fmt.Errorf("pair template must use both sequences")
	}

	specials := make(map[string]SpecialToken, len(specialTokens))
	for _, special := range specialTokens {
		special.IDs = append([]uint32(nil), special.IDs...)
		special.Tokens = append([]string(nil), special.Tokens...)
		specials[special.Identifier] = special
	}
	for _, template := range [][]piece{parsedSingle, parsedPair} {
		for _, item := range template {
			if item.sequence == nil {
				if _, ok := specials[item.special]; !ok {
					return nil, fmt.Errorf("missing special token with identifier %q", item.special)
				}
			}
		}
	}

	return &Template{single: parsedSingle, pair: parsedPair, specials: specials}, nil
}

func (*Template) AffectsIDs() bool { return true }

func (p *Template) AppendIDs(dst, first, second []uint32, pair, addSpecialTokens bool) []uint32 {
	template := p.single
	if pair {
		template = p.pair
	}
	for _, item := range template {
		if item.sequence != nil {
			if *item.sequence == sequenceA {
				dst = append(dst, first...)
			} else {
				dst = append(dst, second...)
			}
			continue
		}
		if addSpecialTokens {
			dst = append(dst, p.specials[item.special].IDs...)
		}
	}
	return dst
}

func (p *Template) Process(first, second []Token, pair, addSpecialTokens bool) []Token {
	template := p.single
	if pair {
		template = p.pair
	}
	result := make([]Token, 0, len(first)+len(second)+len(template))
	for _, item := range template {
		if item.sequence != nil {
			if *item.sequence == sequenceA {
				result = appendSequence(result, first, item.typeID, 0)
			} else {
				result = appendSequence(result, second, item.typeID, 1)
			}
			continue
		}
		if !addSpecialTokens {
			continue
		}
		special := p.specials[item.special]
		for index, id := range special.IDs {
			result = append(result, Token{
				ID:         id,
				Value:      special.Tokens[index],
				TypeID:     item.typeID,
				Special:    true,
				SequenceID: NoSequence,
			})
		}
	}
	return result
}

func parseTemplate(template string) ([]piece, error) {
	parts := strings.Split(template, " ")
	result := make([]piece, len(parts))
	for index, value := range parts {
		parsed, err := parsePiece(value)
		if err != nil {
			return nil, fmt.Errorf("piece %d %q: %w", index, value, err)
		}
		result[index] = parsed
	}
	return result, nil
}

func parsePiece(value string) (piece, error) {
	parts := strings.Split(value, ":")
	if len(parts) > 2 {
		return piece{}, fmt.Errorf("expected at most one type-ID separator")
	}
	typeID := uint32(0)
	if len(parts) == 2 {
		parsed, err := strconv.ParseUint(parts[1], 10, 32)
		if err != nil {
			return piece{}, fmt.Errorf("invalid type ID %q", parts[1])
		}
		typeID = uint32(parsed)
	}

	id := parts[0]
	if !strings.HasPrefix(id, "$") {
		return piece{special: id, typeID: typeID}, nil
	}
	sequenceID := sequenceA
	rest := strings.TrimPrefix(id, "$")
	switch rest {
	case "", "A", "a":
	case "B", "b":
		sequenceID = sequenceB
	default:
		parsed, err := strconv.ParseUint(rest, 10, 32)
		if err != nil {
			return piece{}, fmt.Errorf("invalid sequence identifier %q", id)
		}
		if len(parts) == 1 {
			typeID = uint32(parsed)
		}
	}
	return piece{sequence: &sequenceID, typeID: typeID}, nil
}
