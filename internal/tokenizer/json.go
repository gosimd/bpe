package tokenizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"unicode/utf8"
)

var (
	ErrInvalidTokenizer         = errors.New("bpe: invalid tokenizer configuration")
	ErrUnsupportedConfiguration = errors.New("bpe: unsupported tokenizer configuration")
)

type object map[string]json.RawMessage

func isNull(v []byte) bool { return len(v) == 0 || bytes.Equal(bytes.TrimSpace(v), []byte("null")) }

// Objects consume known fields so misspelled or future options cannot silently
// change semantics. pre_tokenizer is handed to pretok as its original raw JSON.
func readObject(data []byte) (object, error) {
	var o object
	if err := json.Unmarshal(data, &o); err != nil {
		return nil, err
	}
	if o == nil {
		return nil, errors.New("expected an object")
	}
	return o, nil
}

func (o object) take(key string) json.RawMessage { v := o[key]; delete(o, key); return v }
func (o object) done() error {
	for key := range o {
		return fmt.Errorf("unknown field %q", key)
	}
	return nil
}

func field[T any](o object, key string, fallback T, required bool) (T, error) {
	v := o.take(key)
	if len(v) == 0 && !required {
		return fallback, nil
	}
	if isNull(v) {
		return fallback, fmt.Errorf("%s: missing or null value", key)
	}
	var result T
	if err := json.Unmarshal(v, &result); err != nil {
		return fallback, fmt.Errorf("%s: %w", key, err)
	}
	return result, nil
}

func array[T any](raw []byte) ([]T, error) {
	if isNull(raw) {
		return nil, errors.New("array required")
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, err
	}
	out := make([]T, len(entries))
	for i, v := range entries {
		if isNull(v) {
			return nil, fmt.Errorf("array entry %d cannot be null", i)
		}
		if err := json.Unmarshal(v, &out[i]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Validate before decoding: encoding/json otherwise silently replaces malformed
// UTF-8/surrogates and accepts duplicate object keys (including vocab entries).
func validateJSON(data []byte) error {
	if !utf8.Valid(data) {
		return errors.New("JSON is not valid UTF-8")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	var visit func(int) error
	visit = func(depth int) error {
		if depth > 80 {
			return errors.New("JSON nesting limit exceeded")
		}
		token, err := d.Token()
		if err != nil {
			return err
		}
		if delim, ok := token.(json.Delim); ok {
			switch delim {
			case '{':
				seen := map[string]bool{}
				for d.More() {
					key, err := d.Token()
					if err != nil {
						return err
					}
					s, ok := key.(string)
					if !ok {
						return errors.New("expected object key")
					}
					if seen[s] {
						return fmt.Errorf("duplicate key %q", s)
					}
					seen[s] = true
					if err := visit(depth + 1); err != nil {
						return err
					}
				}
			case '[':
				for d.More() {
					if err := visit(depth + 1); err != nil {
						return err
					}
				}
			default:
				return errors.New("unexpected JSON delimiter")
			}
			_, err = d.Token()
			return err
		}
		return nil
	}
	if err := visit(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing JSON data")
	}
	// Syntax is valid now, so every escape has all its required bytes.
	quoted := false
	for i := 0; i < len(data); i++ {
		if data[i] == '"' {
			quoted = !quoted
			continue
		}
		if !quoted || data[i] != '\\' {
			continue
		}
		i++
		if data[i] != 'u' {
			continue
		}
		n, _ := strconv.ParseUint(string(data[i+1:i+5]), 16, 16)
		i += 4
		if n >= 0xD800 && n <= 0xDBFF {
			if i+6 >= len(data) || data[i+1] != '\\' || data[i+2] != 'u' {
				return errors.New("unpaired surrogate")
			}
			low, err := strconv.ParseUint(string(data[i+3:i+7]), 16, 16)
			if err != nil || low < 0xDC00 || low > 0xDFFF {
				return errors.New("unpaired surrogate")
			}
			i += 6
		} else if n >= 0xDC00 && n <= 0xDFFF {
			return errors.New("unpaired surrogate")
		}
	}
	return nil
}
