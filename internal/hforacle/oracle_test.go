package hforacle

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReaderStreamsReferenceAndCorpusInLockstep(t *testing.T) {
	directory := t.TempDir()
	corpusPath := filepath.Join(directory, "corpus.jsonl")
	corpus := "{\"text\":\"hello\"}\n{\"text\":\"\"}\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o600); err != nil {
		t.Fatal(err)
	}
	hash, err := fileSHA256(corpusPath)
	if err != nil {
		t.Fatal(err)
	}

	referencePath := filepath.Join(directory, "reference.jsonl")
	reference := strings.Join([]string{
		`{"kind":"metadata","schema":"gosimd-bpe-hf-fixture-v2","upstream":"huggingface/tokenizers","tokenizers_version":"0.23.2","rust_toolchain":"1.98.1","tokenizer":{"path":"tokenizer.json","sha256":"unused"},"corpus":{"path":"corpus.jsonl","sha256":"` + hash + `","records":2,"utf8_bytes":5},"mode":{"add_special_tokens":false,"padding":false,"truncation":false,"offsets":"byte","records":"independent"}}`,
		`{"kind":"record","index":0,"input_byte_range":[0,5],"output_token_range":[0,1],"pre_tokenization":[{"index":0,"value":"hello","offsets":[0,5],"model":{"source":"bpe","ids":[1],"local_offsets":[[0,5]]}}],"encoding":{"ids":[1],"type_ids":[0],"offsets":[[0,5]],"sequence_ids":[0],"special_tokens_mask":[0]}}`,
		`{"kind":"record","index":1,"input_byte_range":[0,0],"output_token_range":[1,1],"pre_tokenization":[],"encoding":{"ids":[],"type_ids":[],"offsets":[],"sequence_ids":[],"special_tokens_mask":[]}}`,
		"",
	}, "\n")
	if err := os.WriteFile(referencePath, []byte(reference), 0o600); err != nil {
		t.Fatal(err)
	}

	reader, err := Open(referencePath, corpusPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	})

	first, ok, err := reader.Next()
	if err != nil || !ok {
		t.Fatalf("Next() = (%+v, %v, %v)", first, ok, err)
	}
	if first.Text != "hello" || first.Record.Index != 0 || first.Record.PreTokenization[0].Model.IDs[0] != 1 {
		t.Fatalf("first case = %+v", first)
	}
	second, ok, err := reader.Next()
	if err != nil || !ok {
		t.Fatalf("Next() = (%+v, %v, %v)", second, ok, err)
	}
	if second.Text != "" || second.Record.Index != 1 {
		t.Fatalf("second case = %+v", second)
	}
	_, ok, err = reader.Next()
	if err != nil || ok {
		t.Fatalf("final Next() = (_, %v, %v), want (_, false, nil)", ok, err)
	}
}

func TestReaderRejectsCorpusMismatch(t *testing.T) {
	directory := t.TempDir()
	corpusPath := filepath.Join(directory, "corpus.jsonl")
	if err := os.WriteFile(corpusPath, []byte("{\"text\":\"changed\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	referencePath := filepath.Join(directory, "reference.jsonl")
	reference := `{"kind":"metadata","schema":"gosimd-bpe-hf-fixture-v2","upstream":"huggingface/tokenizers","tokenizers_version":"0.23.2","rust_toolchain":"1.98.1","tokenizer":{"path":"tokenizer.json","sha256":"unused"},"corpus":{"path":"corpus.jsonl","sha256":"0000000000000000000000000000000000000000000000000000000000000000","records":1,"utf8_bytes":7},"mode":{"add_special_tokens":false,"padding":false,"truncation":false,"offsets":"byte","records":"independent"}}` + "\n"
	if err := os.WriteFile(referencePath, []byte(reference), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := Open(referencePath, corpusPath)
	if err == nil || !strings.Contains(err.Error(), "corpus SHA-256") {
		t.Fatalf("Open() error = %v, want corpus SHA-256 mismatch", err)
	}
}
