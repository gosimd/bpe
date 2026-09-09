CARGO ?= cargo
GO ?= go

HF_REFERENCE_MANIFEST := tools/hf-reference/Cargo.toml
HF_REFERENCE := $(CARGO) run --locked --release --manifest-path $(HF_REFERENCE_MANIFEST) --
HF_GPT2_TOKENIZER := testdata/gpt2/tokenizer.json
HF_WIKITEXT2_CORPUS := testdata/wikitext2/validation.jsonl
HF_GPT2_FIXTURE := testdata/generated/hf_gpt2_wikitext2_v0.23.2.jsonl

.PHONY: generate-fixtures test-reference
generate-fixtures:
	$(HF_REFERENCE) generate \
		--tokenizer $(HF_GPT2_TOKENIZER) \
		--corpus $(HF_WIKITEXT2_CORPUS) \
		--output $(HF_GPT2_FIXTURE)

test-reference: generate-fixtures
	BPE_HF_REFERENCE=$(abspath $(HF_GPT2_FIXTURE)) \
	BPE_HF_REPOSITORY=$(CURDIR) \
	$(GO) test -count=1 -run '^TestHFGPT2WikiText2' ./internal/bytelevel .
