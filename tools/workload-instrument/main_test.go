package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGenerateAndCheck(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "internal/model")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"encode.go", "merge.go"} {
		data, err := os.ReadFile(filepath.Join("../../internal/model", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0644); err != nil {
			t.Fatal(err)
		}
	}
	if err := run(root, false); err != nil {
		t.Fatal(err)
	}
	if err := run(root, true); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "workload_generated_test.go")
	if err := os.WriteFile(path, []byte("package model\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := run(root, true); err == nil {
		t.Fatal("accepted stale instrumentation")
	}
}
