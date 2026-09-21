package tokenizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
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
