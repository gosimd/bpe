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
WORKLOAD_DIR ?= testdata/generated/model-workload

HF_REFERENCE_MANIFEST := tools/hf-reference/Cargo.toml

.PHONY: generate-tokenizer-fixtures
generate-tokenizer-fixtures:
	$(CARGO) run --locked --release --manifest-path $(HF_REFERENCE_MANIFEST) \
		--example tokenizer_json -- testdata/hf_tokenizer_json_v0.23.2.json

.PHONY: generate-bytelevel-fixtures
generate-bytelevel-fixtures:
	$(CARGO) run --locked --release --manifest-path $(HF_REFERENCE_MANIFEST) \
		--example bytelevel_prefix -- testdata/hf_bytelevel_prefix_v0.23.2.json

.PHONY: benchmark-go-reference generate-fixtures test-reference
generate-fixtures: generate-suite-fixtures

test-reference: generate-fixtures
	BPE_BENCH_SUITE=$(abspath $(SUITE)) BPE_HF_REPOSITORY=$(CURDIR) \
	BPE_BENCH_MODELS='$(MODELS)' BPE_BENCH_CORPORA='$(CORPORA)' \
	$(GO) test -count=1 -run '^TestHFSuite' ./internal/pretokenize .

benchmark-go-reference: benchmark-suite

# Test-only counted copy of the current merge methods; never a timed backend.
.PHONY: analyze-model-workload
analyze-model-workload:
	$(GO) run ./tools/workload-instrument -check
	BPE_BENCH_SUITE=$(abspath $(SUITE)) BPE_HF_REPOSITORY=$(CURDIR) \
	BPE_BENCH_MODELS='$(MODELS)' BPE_BENCH_CORPORA='$(CORPORA)' \
	BPE_WORKLOAD_OUTPUT=$(abspath $(WORKLOAD_DIR)) \
	$(GO) test -count=1 -run '^TestModelWorkload$$' -v ./internal/model

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
