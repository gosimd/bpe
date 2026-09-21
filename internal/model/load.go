package model

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/gosimd/bpe/internal/jsoncheck"
)

// ParseVocabulary parses the object used by Hugging Face vocab.json files.
// IDs must be unsigned decimal integers that fit uint32.
func ParseVocabulary(data []byte) (map[string]uint32, error) {
	if err := jsoncheck.Validate(data); err != nil {
		return nil, fmt.Errorf("%w: parse JSON: %v", ErrInvalidVocabulary, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("%w: parse JSON: %v", ErrInvalidVocabulary, err)
	}
	if raw == nil {
		return nil, fmt.Errorf("%w: expected an object", ErrInvalidVocabulary)
	}

	vocab := make(map[string]uint32, len(raw))
	for token, rawID := range raw {
		id, err := strconv.ParseUint(strings.TrimSpace(string(rawID)), 10, 32)
		if err != nil {
			return nil, fmt.Errorf("%w: token %q has a non-uint32 ID", ErrInvalidVocabulary, token)
		}
		vocab[token] = uint32(id)
	}
	return vocab, nil
}

// ParseMerges parses the two-token, rank-ordered records used by Hugging Face
// merges.txt files. #version headers are ignored, like upstream tokenizers.
func ParseMerges(data []byte) ([]Merge, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 1024), 16*1024*1024)

	merges := make([]Merge, 0)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if strings.HasPrefix(line, "#version") {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("%w: line %d must contain exactly two tokens separated by one space", ErrInvalidMerge, lineNumber)
		}
		merges = append(merges, Merge{Left: parts[0], Right: parts[1]})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("%w: read merges: %v", ErrInvalidMerge, err)
	}
	return merges, nil
}
