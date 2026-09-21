package bpe

import (
	"fmt"
	"os"

	"github.com/gosimd/bpe/internal/tokenizer"
)

// ParseTokenizer compiles a Hugging Face tokenizer.json into an immutable,
// concurrency-safe tokenizer. The input bytes are not retained. It supports
// deterministic BPE, pretok pre-tokenization, AddedTokens, TemplateProcessing
// and ByteLevel post-processing.
//
// Non-null normalizer, padding and truncation, unsupported processors and
// unsafe pipeline combinations are rejected. Decoder metadata is accepted but
// never executed: this API provides encoding only. See docs/Tokenizer.md.
func ParseTokenizer(data []byte) (*Tokenizer, error) {
	impl, err := tokenizer.ParseJSON(data)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{impl: impl}, nil
}

// OpenTokenizer reads a local tokenizer.json and compiles it with ParseTokenizer.
// It does not access the network or load model weights or sibling files.
func OpenTokenizer(filename string) (*Tokenizer, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer %q: %w", filename, err)
	}
	t, err := ParseTokenizer(data)
	if err != nil {
		return nil, fmt.Errorf("tokenizer %q: %w", filename, err)
	}
	return t, nil
}
