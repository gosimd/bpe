CARGO ?= cargo
GO ?= go
SUITE ?= benchmarks/suite.json
MODELS ?=
CORPORA ?=
BENCHTIME ?= 1x
COUNT ?= 10
HF_RESULT_DIR ?= testdata/generated/hf-benchmarks
HF_WARMUP ?= 3
HF_ITERATIONS ?= 10
HF_ENVIRONMENT ?=

HF_REFERENCE_MANIFEST := tools/hf-reference/Cargo.toml

.PHONY: benchmark-go-reference generate-fixtures test-reference
generate-fixtures: generate-suite-fixtures

test-reference: generate-fixtures
	BPE_BENCH_SUITE=$(abspath $(SUITE)) BPE_HF_REPOSITORY=$(CURDIR) \
	BPE_BENCH_MODELS='$(MODELS)' BPE_BENCH_CORPORA='$(CORPORA)' \
	$(GO) test -count=1 -run '^TestHFSuite' ./internal/bytelevel .

benchmark-go-reference: benchmark-suite

# Explicit tooling targets. Neither Go tests nor benchmarks invoke Rust.
.PHONY: prepare-corpora generate-suite-fixtures benchmark-suite benchmark-hf-suite
prepare-corpora:
	$(GO) run ./tools/benchmark-suite -suite '$(SUITE)' \
		-models '$(MODELS)' -corpora '$(CORPORA)' -prepare-only

generate-suite-fixtures:
	$(CARGO) build --locked --release --manifest-path $(HF_REFERENCE_MANIFEST)
	$(GO) run ./tools/benchmark-suite -suite '$(SUITE)' \
		-models '$(MODELS)' -corpora '$(CORPORA)'

benchmark-suite: generate-suite-fixtures
	BPE_BENCH_SUITE=$(abspath $(SUITE)) BPE_HF_REPOSITORY=$(CURDIR) \
	BPE_BENCH_MODELS='$(MODELS)' BPE_BENCH_CORPORA='$(CORPORA)' GOMAXPROCS=1 \
	$(GO) test -run '^$$' -bench '^BenchmarkBPE$$' \
		-benchmem -benchtime=$(BENCHTIME) -count=$(COUNT) -cpu=1 .

benchmark-hf-suite:
	$(CARGO) build --locked --release --manifest-path $(HF_REFERENCE_MANIFEST)
	$(GO) run ./tools/benchmark-suite -suite '$(SUITE)' \
		-models '$(MODELS)' -corpora '$(CORPORA)' -hf-output '$(HF_RESULT_DIR)' \
		-warmup $(HF_WARMUP) -iterations $(HF_ITERATIONS) -environment '$(HF_ENVIRONMENT)'
