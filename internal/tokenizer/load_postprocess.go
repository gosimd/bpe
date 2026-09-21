package tokenizer

import (
	"encoding/json"
	"fmt"

	"github.com/gosimd/bpe/internal/postprocess"
)

func parsePostProcessor(raw []byte) (postprocess.Processor, error) {
	if isNull(raw) {
		return nil, nil
	}
	o, err := readObject(raw)
	if err != nil {
		return nil, err
	}
	typ, err := field(o, "type", "", true)
	if err != nil {
		return nil, err
	}
	switch typ {
	case "ByteLevel":
		prefix, err := field(o, "add_prefix_space", true, false)
		if err != nil {
			return nil, err
		}
		trim, err := field(o, "trim_offsets", true, false)
		if err != nil {
			return nil, err
		}
		if _, err := field(o, "use_regex", true, false); err != nil {
			return nil, err
		}
		if err := o.done(); err != nil {
			return nil, err
		}
		return postprocess.NewByteLevel(prefix, trim), nil
	case "TemplateProcessing":
		single, err := parseTemplateItems(o.take("single"))
		if err != nil {
			return nil, fmt.Errorf("single: %w", err)
		}
		pair, err := parseTemplateItems(o.take("pair"))
		if err != nil {
			return nil, fmt.Errorf("pair: %w", err)
		}
		specials, err := readObject(o.take("special_tokens"))
		if err != nil {
			return nil, fmt.Errorf("special_tokens: %w", err)
		}
		defs := make([]postprocess.SpecialToken, 0, len(specials))
		for key, raw := range specials {
			s, err := readObject(raw)
			if err != nil {
				return nil, err
			}
			id, err := field(s, "id", "", true)
			if err != nil {
				return nil, err
			}
			if id == "" || id != key {
				return nil, fmt.Errorf("special_tokens key %q must equal its nonempty id %q", key, id)
			}
			ids, err := array[uint32](s.take("ids"))
			if err != nil {
				return nil, err
			}
			tokens, err := array[string](s.take("tokens"))
			if err != nil {
				return nil, err
			}
			if len(ids) != len(tokens) {
				return nil, fmt.Errorf("special token %q: IDs/tokens length mismatch", id)
			}
			if err := s.done(); err != nil {
				return nil, err
			}
			defs = append(defs, postprocess.SpecialToken{Identifier: id, IDs: ids, Tokens: tokens})
		}
		if err := o.done(); err != nil {
			return nil, err
		}
		return postprocess.NewTemplatePieces(single, pair, defs)
	default:
		return nil, fmt.Errorf("%w: post-processor type %q", ErrUnsupportedConfiguration, typ)
	}
}

func parseTemplateItems(raw []byte) ([]postprocess.TemplatePiece, error) {
	if isNull(raw) {
		return nil, fmt.Errorf("template array required")
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, err
	}
	out := make([]postprocess.TemplatePiece, len(entries))
	for i, raw := range entries {
		o, err := readObject(raw)
		if err != nil {
			return nil, err
		}
		if len(o) != 1 {
			return nil, fmt.Errorf("piece %d must contain one Sequence or SpecialToken", i)
		}
		seq, ok := o["Sequence"]
		kind := "Sequence"
		if !ok {
			seq, ok = o["SpecialToken"]
			kind = "SpecialToken"
		}
		if !ok {
			return nil, fmt.Errorf("piece %d: unknown template entry", i)
		}
		p, err := readObject(seq)
		if err != nil {
			return nil, err
		}
		id, err := field(p, "id", "", true)
		if err != nil {
			return nil, err
		}
		typeID, err := field(p, "type_id", uint32(0), true)
		if err != nil {
			return nil, err
		}
		if err := p.done(); err != nil {
			return nil, err
		}
		out[i] = postprocess.TemplatePiece{SequenceID: postprocess.NoSequence, Identifier: id, TypeID: typeID}
		if kind == "Sequence" {
			switch id {
			case "A":
				out[i].SequenceID = 0
			case "B":
				out[i].SequenceID = 1
			default:
				return nil, fmt.Errorf("invalid sequence %q", id)
			}
		}
	}
	return out, nil
}
