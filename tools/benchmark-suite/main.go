// Command benchmark-suite prepares local corpora and explicitly runs the pinned
// Rust oracle. It is never invoked by ordinary go test.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/gosimd/bpe/internal/benchsuite"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	root := flag.String("root", ".", "repository root")
	suitePath := flag.String("suite", "benchmarks/suite.json", "catalogue path relative to root")
	models := flag.String("models", "", "comma-separated model names; empty selects all")
	corpora := flag.String("corpora", "", "comma-separated corpus names; empty selects all")
	prepareOnly := flag.Bool("prepare-only", false, "prepare corpus JSONL without Rust")
	reference := flag.String("reference-bin", "tools/hf-reference/target/release/hf-reference", "prebuilt pinned Rust executable relative to root")
	hfOutput := flag.String("hf-output", "", "optional HF timing output directory (relative to root); runs all four HF modes")
	environment := flag.String("environment", "", "CPU/model, OS version and execution conditions; required with -hf-output")
	warmup := flag.Int("warmup", 3, "HF warm-up passes")
	iterations := flag.Int("iterations", 10, "HF measured passes")
	flag.Parse()
	if flag.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments")
	}
	if *hfOutput != "" && strings.TrimSpace(*environment) == "" {
		return fmt.Errorf("HF timings require -environment (make: HF_ENVIRONMENT): CPU/model, OS version, power mode and background-load/affinity conditions")
	}
	if *warmup < 0 || *iterations <= 0 {
		return fmt.Errorf("warmup must be nonnegative and iterations positive")
	}
	absRoot, err := filepath.Abs(*root)
	if err != nil {
		return err
	}
	resolve := func(path string) string {
		if filepath.IsAbs(path) {
			return path
		}
		return filepath.Join(absRoot, path)
	}
	suite, err := benchsuite.Load(resolve(*suitePath))
	if err != nil {
		return err
	}
	suite, err = suite.Select(*models, *corpora)
	if err != nil {
		return err
	}
	if *prepareOnly && *hfOutput != "" {
		return fmt.Errorf("-prepare-only cannot be combined with -hf-output")
	}
	if !*prepareOnly {
		for _, m := range suite.Models {
			if _, err := benchsuite.ModelOptions(resolve(m.Tokenizer), m.Pipeline); err != nil {
				return fmt.Errorf("%s: %w", m.Name, err)
			}
			if err := benchsuite.ValidateModelFiles(absRoot, m); err != nil {
				return err
			}
		}
	}
	for _, c := range suite.Corpora {
		if err := benchsuite.PrepareCorpus(absRoot, c); err != nil {
			return fmt.Errorf("%s: %w", c.Name, err)
		}
		fmt.Printf("Prepared %s: %s\n", c.Name, c.CorpusPath(absRoot))
	}
	if *prepareOnly {
		return nil
	}
	invoke := func(args ...string) error {
		cmd := exec.Command(resolve(*reference), args...)
		cmd.Dir = absRoot
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		return cmd.Run()
	}
	for _, m := range suite.Models {
		for _, c := range suite.Corpora {
			oracle := benchsuite.ReferencePath(absRoot, m, c)
			if err := os.MkdirAll(filepath.Dir(oracle), 0755); err != nil {
				return err
			}
			fmt.Printf("Oracle %s/%s\n", m.Name, c.Name)
			if err := invoke("generate", "--tokenizer", resolve(m.Tokenizer), "--corpus", c.CorpusPath(absRoot), "--output", oracle); err != nil {
				return err
			}
			if *hfOutput == "" {
				continue
			}
			dir := filepath.Join(resolve(*hfOutput), m.Name)
			if err := os.MkdirAll(dir, 0755); err != nil {
				return err
			}
			if err := invoke("benchmark", "--tokenizer", resolve(m.Tokenizer), "--vocab", resolve(m.Vocab), "--merges", resolve(m.Merges), "--corpus", c.CorpusPath(absRoot), "--fixture", oracle,
				"--output", filepath.Join(dir, c.Name+".json"), "--threads", "1", "--warmup", fmt.Sprint(*warmup), "--iterations", fmt.Sprint(*iterations),
				"--environment", *environment+"; benchmark-suite: "+m.Name+"/"+c.Name+"; independent records; sequential single-thread execution"); err != nil {
				return err
			}
		}
	}
	return nil
}
