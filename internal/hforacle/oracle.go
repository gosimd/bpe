// Package hforacle reads locally generated Hugging Face differential fixtures.
// It is test infrastructure rather than part of the public bpe package.
package hforacle

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	// ReferenceEnvironment names the opt-in full-corpus oracle path.
	ReferenceEnvironment = "BPE_HF_REFERENCE"
	// RepositoryEnvironment names the repository root used to locate pinned
	// tokenizer and corpus inputs from package-specific test working directories.
	RepositoryEnvironment = "BPE_HF_REPOSITORY"

	wantSchema            = "gosimd-bpe-hf-fixture-v2"
	wantUpstream          = "huggingface/tokenizers"
	wantTokenizersVersion = "0.23.2"
	maximumJSONLineBytes  = 64 << 20
)

// FileMetadata identifies one oracle input.
type FileMetadata struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// CorpusMetadata identifies the JSONL corpus used by the oracle.
type CorpusMetadata struct {
	Path      string `json:"path"`
	SHA256    string `json:"sha256"`
	Records   int    `json:"records"`
	UTF8Bytes int    `json:"utf8_bytes"`
}

// Mode records the upstream encoding settings used to build the fixture.
type Mode struct {
	AddSpecialTokens bool   `json:"add_special_tokens"`
	Padding          bool   `json:"padding"`
	Truncation       bool   `json:"truncation"`
	Offsets          string `json:"offsets"`
	Records          string `json:"records"`
}

// Metadata is the first line of a three-level oracle fixture.
type Metadata struct {
	Kind              string         `json:"kind"`
	Schema            string         `json:"schema"`
	Upstream          string         `json:"upstream"`
	TokenizersVersion string         `json:"tokenizers_version"`
	RustToolchain     string         `json:"rust_toolchain"`
	Tokenizer         FileMetadata   `json:"tokenizer"`
	Corpus            CorpusMetadata `json:"corpus"`
	Mode              Mode           `json:"mode"`
}

// PieceModelOutput is the Hugging Face model output for one pre-tokenized
// piece. Offsets are byte offsets relative to Piece.Value.
type PieceModelOutput struct {
	Source       string   `json:"source"`
	IDs          []uint32 `json:"ids"`
	LocalOffsets [][2]int `json:"local_offsets"`
}

// Piece is one transformed pre-token and its record-relative byte range.
type Piece struct {
	Index   int              `json:"index"`
	Value   string           `json:"value"`
	Offsets [2]int           `json:"offsets"`
	Model   PieceModelOutput `json:"model"`
}

// Encoding is the final Hugging Face Encoding for one corpus record.
type Encoding struct {
	IDs               []uint32 `json:"ids"`
	TypeIDs           []uint32 `json:"type_ids"`
	Offsets           [][2]int `json:"offsets"`
	SequenceIDs       []*int   `json:"sequence_ids"`
	SpecialTokensMask []uint32 `json:"special_tokens_mask"`
}

// Record links the pre-tokenizer, model, and final Encoding oracle levels.
type Record struct {
	Kind             string   `json:"kind"`
	Index            int      `json:"index"`
	InputByteRange   [2]int   `json:"input_byte_range"`
	OutputTokenRange [2]int   `json:"output_token_range"`
	PreTokenization  []Piece  `json:"pre_tokenization"`
	Encoding         Encoding `json:"encoding"`
}

type corpusRecord struct {
	Text string `json:"text"`
}

// Case is one oracle record paired with its original corpus text.
type Case struct {
	Text   string
	Record Record
}

// Reader streams a reference JSONL and its source corpus in lockstep.
type Reader struct {
	referenceFile    *os.File
	corpusFile       *os.File
	referenceScanner *bufio.Scanner
	corpusScanner    *bufio.Scanner
	nextIndex        int

	Metadata Metadata
}

// ValidateTokenizer requires path to match the tokenizer.json hashed into the
// oracle metadata.
func (m Metadata) ValidateTokenizer(path string) error {
	hash, err := fileSHA256(path)
	if err != nil {
		return fmt.Errorf("hash tokenizer %q: %w", path, err)
	}
	if hash != m.Tokenizer.SHA256 {
		return fmt.Errorf("tokenizer SHA-256 is %s, oracle requires %s", hash, m.Tokenizer.SHA256)
	}
	return nil
}

// Open opens and validates a three-level oracle and its pinned source corpus.
func Open(referencePath, corpusPath string) (*Reader, error) {
	referenceFile, err := os.Open(referencePath)
	if err != nil {
		return nil, fmt.Errorf("open Hugging Face reference %q: %w", referencePath, err)
	}
	referenceScanner := newScanner(referenceFile)
	if !referenceScanner.Scan() {
		referenceFile.Close()
		if err := referenceScanner.Err(); err != nil {
			return nil, fmt.Errorf("read Hugging Face reference metadata: %w", err)
		}
		return nil, errors.New("Hugging Face reference is empty")
	}

	var metadata Metadata
	if err := decodeLine(referenceScanner.Bytes(), &metadata); err != nil {
		referenceFile.Close()
		return nil, fmt.Errorf("decode Hugging Face reference metadata: %w", err)
	}
	if err := validateMetadata(metadata); err != nil {
		referenceFile.Close()
		return nil, err
	}

	corpusHash, err := fileSHA256(corpusPath)
	if err != nil {
		referenceFile.Close()
		return nil, fmt.Errorf("hash corpus %q: %w", corpusPath, err)
	}
	if corpusHash != metadata.Corpus.SHA256 {
		referenceFile.Close()
		return nil, fmt.Errorf("corpus SHA-256 is %s, oracle requires %s", corpusHash, metadata.Corpus.SHA256)
	}

	corpusFile, err := os.Open(corpusPath)
	if err != nil {
		referenceFile.Close()
		return nil, fmt.Errorf("open corpus %q: %w", corpusPath, err)
	}
	return &Reader{
		referenceFile:    referenceFile,
		corpusFile:       corpusFile,
		referenceScanner: referenceScanner,
		corpusScanner:    newScanner(corpusFile),
		Metadata:         metadata,
	}, nil
}

// Next returns the next corpus text and corresponding oracle record.
func (r *Reader) Next() (Case, bool, error) {
	referenceOK := r.referenceScanner.Scan()
	corpusOK := r.corpusScanner.Scan()
	if !referenceOK || !corpusOK {
		if err := r.referenceScanner.Err(); err != nil {
			return Case{}, false, fmt.Errorf("read Hugging Face reference record %d: %w", r.nextIndex, err)
		}
		if err := r.corpusScanner.Err(); err != nil {
			return Case{}, false, fmt.Errorf("read corpus record %d: %w", r.nextIndex, err)
		}
		if referenceOK != corpusOK {
			return Case{}, false, fmt.Errorf("reference and corpus record counts differ at record %d", r.nextIndex)
		}
		if r.nextIndex != r.Metadata.Corpus.Records {
			return Case{}, false, fmt.Errorf("read %d records, metadata requires %d", r.nextIndex, r.Metadata.Corpus.Records)
		}
		return Case{}, false, nil
	}

	var record Record
	if err := decodeLine(r.referenceScanner.Bytes(), &record); err != nil {
		return Case{}, false, fmt.Errorf("decode Hugging Face reference record %d: %w", r.nextIndex, err)
	}
	if record.Kind != "record" {
		return Case{}, false, fmt.Errorf("reference item %d has kind %q, want %q", r.nextIndex, record.Kind, "record")
	}
	if record.Index != r.nextIndex {
		return Case{}, false, fmt.Errorf("reference record index is %d, want %d", record.Index, r.nextIndex)
	}

	if len(r.corpusScanner.Bytes()) == 0 {
		return Case{}, false, fmt.Errorf("corpus record %d is a physically blank line", r.nextIndex)
	}
	var source corpusRecord
	if err := decodeLine(r.corpusScanner.Bytes(), &source); err != nil {
		return Case{}, false, fmt.Errorf("decode corpus record %d: %w", r.nextIndex, err)
	}
	if record.InputByteRange != [2]int{0, len(source.Text)} {
		return Case{}, false, fmt.Errorf("record %d input byte range is %v for %d-byte text", r.nextIndex, record.InputByteRange, len(source.Text))
	}

	r.nextIndex++
	return Case{Text: source.Text, Record: record}, true, nil
}

// Close closes the reference and corpus files.
func (r *Reader) Close() error {
	return errors.Join(r.referenceFile.Close(), r.corpusFile.Close())
}

func validateMetadata(metadata Metadata) error {
	if metadata.Kind != "metadata" {
		return fmt.Errorf("reference first item has kind %q, want %q", metadata.Kind, "metadata")
	}
	if metadata.Schema != wantSchema {
		return fmt.Errorf("reference schema is %q, want %q", metadata.Schema, wantSchema)
	}
	if metadata.Upstream != wantUpstream {
		return fmt.Errorf("reference upstream is %q, want %q", metadata.Upstream, wantUpstream)
	}
	if metadata.TokenizersVersion != wantTokenizersVersion {
		return fmt.Errorf("reference tokenizers version is %q, want %q", metadata.TokenizersVersion, wantTokenizersVersion)
	}
	mode := metadata.Mode
	if mode.AddSpecialTokens || mode.Padding || mode.Truncation || mode.Offsets != "byte" || mode.Records != "independent" {
		return fmt.Errorf("reference encoding mode is unsupported: %+v", mode)
	}
	if metadata.Corpus.Records < 0 || metadata.Corpus.UTF8Bytes < 0 {
		return errors.New("reference corpus metadata contains a negative count")
	}
	return nil
}

func newScanner(reader io.Reader) *bufio.Scanner {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64<<10), maximumJSONLineBytes)
	return scanner
}

func decodeLine(line []byte, value any) error {
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("JSON line contains more than one value")
		}
		return err
	}
	return nil
}

func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}
